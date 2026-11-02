"""Кружочек для телеграма: Мика говорит готовый сценарий. Сервер Мики не нужен — голос из mika_voice.Voice,
страницу аватара раздаёт встроенный веб-сервер, Chrome без окна рисует её в режиме ?render=1.
Кадры снимаются по одному по виртуальным часам — ролик ровный, сколько бы ни рисовался кадр.
Квадрат 640×640, H.264 + AAC: так его принимает sendVideoNote.

python video/note.py "Сценарий с метками [greet] жестов" [out.mp4] [--dist 1.9] [--fps 30]
python video/note.py --file script.txt [out.mp4]

Из кода: notes = Notes(); notes.render("сценарий", "out.mp4") — модели грузятся один раз."""
import argparse, asyncio, base64, functools, json, os, subprocess, sys, tempfile, threading, time, urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import torch
os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]  # cuDNN для whisper
import websockets

from mika_voice import Voice

CHROME = os.environ.get("CHROME", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
FFMPEG = os.environ.get("FFMPEG", r"D:\work\ffmpeg\bin\ffmpeg.exe")
VOICE_MODEL = os.environ.get("VOICE_MODEL", os.path.join(ROOT, "models", "masha-calm-tts"))
VOICE_REF = os.environ.get("VOICE_REF", os.path.join(ROOT, "voices", "masha.wav"))
SIZE, LIMIT = 640, 60  # у кружка не больше 60 с
MAX_BYTES = 12 * 1024 * 1024  # и не больше 12 МБ: файл крупнее sendVideoNote отвергает
BUDGET_BYTES, AUDIO_KBIT = 11_000_000, 128  # с запасом: потолок битрейта считаем от этого бюджета
WARMUP = 45  # кадров до начала: волосы и одежда успевают осесть
NO_PROXY = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # в окружении бывает внешний прокси


class _Static(SimpleHTTPRequestHandler):
    """Только static/ — страница, модель аватара, движения; остальное (веса голоса, данные) не отдаём."""

    def send_head(self):
        if not self.path.startswith("/static/"):
            self.send_error(404)
            return None
        return super().send_head()

    def log_message(self, *a):
        pass


class Page:
    """Вкладка Chrome по DevTools: выполнить выражение и вернуть значение."""

    def __init__(self, ws):
        self.ws, self.n = ws, 0
        self.errors = []  # ошибки и предупреждения страницы — для разбора, если она не загрузилась

    async def call(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(await self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m["result"]
            p = m.get("params", {})
            if m.get("method") == "Runtime.exceptionThrown":
                d = p["exceptionDetails"]
                self.errors.append(d.get("exception", {}).get("description") or d.get("text"))
            elif m.get("method") == "Runtime.consoleAPICalled" and p.get("type") in ("error", "warning"):
                self.errors.append(" ".join(str(a.get("value", a.get("description", ""))) for a in p.get("args", [])))
            elif m.get("method") == "Log.entryAdded" and p["entry"].get("level") in ("error", "warning"):
                self.errors.append(f"{p['entry'].get('text')} {p['entry'].get('url', '')}")

    async def eval(self, js):
        r = await self.call("Runtime.evaluate", expression=js, awaitPromise=True, returnByValue=True)
        if "exceptionDetails" in r:
            raise RuntimeError(r["exceptionDetails"].get("exception", {}).get("description") or r["exceptionDetails"]["text"])
        return r["result"].get("value")


def _free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Notes:
    def __init__(self, voice=None):
        self.voice = voice or Voice(VOICE_MODEL, VOICE_REF)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(_Static, directory=ROOT))
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def render(self, script, out, dist=1.9, fps=30, query=""):
        """Сценарий -> mp4 в out. Возвращает длительность, с."""
        t = time.perf_counter()
        clip = self.voice.script(script)
        print(f"голос: {clip['duration']:.1f} с за {time.perf_counter() - t:.1f} с", flush=True)
        if clip["duration"] > LIMIT:
            print(f"внимание: дольше {LIMIT} с — кружком телеграм такое не примет", flush=True)
        url = f"http://127.0.0.1:{self.http.server_port}/static/index.html?render=1&dist={dist}" + (f"&{query}" if query else "")
        asyncio.run(self._frames(clip, url, out, fps))
        return clip["duration"]

    async def _frames(self, clip, url, out, fps):
        work = tempfile.mkdtemp(prefix="mika-note-")
        raw = os.path.join(work, "voice.f32")
        clip["audio"].astype("<f4").tofile(raw)
        port = _free_port()
        chrome = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={work}",
                                   f"--window-size={SIZE},{SIZE}", "--force-device-scale-factor=1", "--hide-scrollbars",
                                   "--no-proxy-server", "--enable-gpu", "--ignore-gpu-blocklist",
                                   *os.environ.get("CHROME_FLAGS", "--use-angle=d3d11").split(), "about:blank"],
                                  stdout=subprocess.DEVNULL, stderr=open(os.path.join(work, "chrome.log"), "wb"))
        try:
            async with websockets.connect(self._target(port), proxy=None, max_size=None) as ws:
                page = Page(ws)
                # сначала слушаем ошибки, потом открываем страницу — иначе ранние ошибки модуля не увидеть
                await page.call("Runtime.enable")
                await page.call("Log.enable")
                # Окно headless-Chrome не равно странице: из --window-size он вычитает место под несуществующую панель
                # (вышло 640×497 на .90 и 624×545 здесь), кадр растягивался, а в mp4 уходили неквадратные пиксели —
                # телеграм рисовал кружок с чёрными полосами. Размер страницы задаём явно.
                await page.call("Emulation.setDeviceMetricsOverride", width=SIZE, height=SIZE, deviceScaleFactor=1, mobile=False)
                await page.call("Page.navigate", url=url)
                for _ in range(600):  # модель и движения грузятся несколько секунд
                    if await page.eval("typeof mikaReady === 'function' && mikaReady()"):
                        break
                    await asyncio.sleep(0.1)
                else:
                    state = await page.eval("JSON.stringify({ready: document.readyState, mika: typeof window.mika,"
                                            " vrm: !!(window.mika && mika.vrm), idle: window.mika && mika.motion.idle.length,"
                                            " caption: (document.getElementById('caption') || {}).textContent})")
                    raise RuntimeError(f"страница аватара не загрузилась: {state}; ошибки страницы: {page.errors[-8:]}")
                canvas = await page.eval("(() => { const c = document.getElementById('stage'); return [c.width, c.height]; })()")
                if canvas != [SIZE, SIZE]:
                    raise RuntimeError(f"кадр не квадратный: {canvas[0]}×{canvas[1]} вместо {SIZE}×{SIZE}")
                await page.eval(f"mikaLoad({json.dumps(clip['lines'], ensure_ascii=False)})")
                # Мерцающие звёзды сжимаются плохо: без потолка ролик длиннее ~40 с не влезал в MAX_BYTES.
                kbit = int(BUDGET_BYTES * 8 / 1000 / clip["duration"]) - AUDIO_KBIT
                enc = subprocess.Popen([FFMPEG, "-y", "-loglevel", "error",
                                        "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
                                        "-f", "f32le", "-ar", str(clip["rate"]), "-ac", "1", "-i", raw,
                                        "-vf", f"scale={SIZE}:{SIZE},setsar=1", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                                        "-maxrate", f"{kbit}k", "-bufsize", f"{kbit}k",
                                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", f"{AUDIO_KBIT}k",
                                        "-shortest", "-movflags", "+faststart", out], stdin=subprocess.PIPE)
                for _ in range(WARMUP):
                    await page.eval(f"mikaFrame(0, {1 / fps}, false)")
                frames, t = int(clip["duration"] * fps + 0.999), time.perf_counter()
                for i in range(frames):
                    enc.stdin.write(base64.b64decode(await page.eval(f"mikaFrame({i / fps}, {1 / fps})")))
                enc.stdin.close()
                if enc.wait():
                    raise RuntimeError("ffmpeg не собрал ролик")
                if os.path.getsize(out) > MAX_BYTES:
                    raise RuntimeError(f"ролик {os.path.getsize(out)} байт: кружком телеграм такое не примет")
                print(f"кадры: {frames} за {time.perf_counter() - t:.1f} с -> {out}", flush=True)
        finally:
            chrome.kill()
            chrome.wait()

    @staticmethod
    def _target(port):
        for _ in range(200):
            try:
                tabs = json.load(NO_PROXY.open(f"http://127.0.0.1:{port}/json"))
                return next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"]
            except Exception:
                time.sleep(0.1)
        raise RuntimeError("Chrome не поднялся")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--file")
    ap.add_argument("--dist", type=float, default=1.9, help="камера: 1.3 — лицо, 3 — в полный рост")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--query", default="", help="добавка к адресу страницы, например try=hi:vroid/VRMA_05.vrma")
    a = ap.parse_args()
    if a.file:  # с --file первый свободный аргумент — это уже имя ролика
        a.out, text = a.out or a.text, open(a.file, encoding="utf-8").read().strip()
    else:
        text = a.text
    if not text:
        sys.exit("нужен текст или --file")
    out = os.path.abspath(a.out or os.path.join(ROOT, "video", "out", time.strftime("note-%Y%m%d-%H%M%S.mp4")))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    Notes().render(text, out, a.dist, a.fps, a.query)


if __name__ == "__main__":
    main()
