"""Сервис кружочков: HTTP-обёртка над video/note.py для очереди ai-api-stuff.

Свой контейнер, потому что голосу нужны свои версии библиотек (Qwen3-TTS не грузится с transformers
из ai-api-stuff). Карту делим так же, как с ComfyUI: очередь ai-api-stuff решает, на какой карте задача,
перед ней освобождает память у ComfyUI, а когда карта нужна другим — зовёт /free.

  POST /render {"script": "...", "device": 0, "dist": 1.9} -> video/mp4 (640×640, ≤60 с)
  POST /free   {"device": 0} или {}                       -> модели выгружены, память карты отдана
  GET  /health

На каждую карту — свой процесс (CUDA_VISIBLE_DEVICES), модели в нём живут между задачами.
Освободить память изнутри живого процесса нельзя (CUDA держит контекст), поэтому /free его убивает."""
import multiprocessing as mp
import os
import queue
import sys
import tempfile
import threading

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import Response

TIMEOUT = 300  # с на один кружок с загрузкой моделей; обычно ~30 с

ctx = mp.get_context("spawn")
_start_lock = threading.Lock()


def _loop(jobs, results):
    """Процесс карты: модели грузятся на первой задаче и живут, пока процесс не убьют."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from note import Notes

    notes = None
    while True:
        job = jobs.get()
        try:
            notes = notes or Notes()
            out = os.path.join(tempfile.mkdtemp(prefix="note-"), "note.mp4")
            duration = notes.render(job["script"], out, dist=job.get("dist", 1.9))
            with open(out, "rb") as f:
                results.put({"mp4": f.read(), "duration": duration})
            os.unlink(out)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.put({"error": f"{type(e).__name__}: {e}"})


class Card:
    def __init__(self, device):
        self.device, self.proc, self.lock = device, None, threading.Lock()

    def _start(self):
        self.jobs, self.results = ctx.Queue(), ctx.Queue()
        self.proc = ctx.Process(target=_loop, args=(self.jobs, self.results), daemon=True)
        with _start_lock:  # окружение у процесса одно, а карт несколько
            prev = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.device)
            try:
                self.proc.start()
            finally:
                if prev is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = prev

    def render(self, job):
        with self.lock:
            if not self.proc or not self.proc.is_alive():
                self._start()
            self.jobs.put(job)
            waited = 0
            while True:
                try:
                    return self.results.get(timeout=2)
                except queue.Empty:
                    waited += 2
                    if not self.proc.is_alive():
                        code = self.proc.exitcode
                        self.proc = None
                        return {"error": f"процесс карты умер (код {code})"}
                    if waited >= TIMEOUT:
                        self._kill()
                        return {"error": f"кружок не готов за {TIMEOUT} с"}

    def _kill(self):
        if self.proc and self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(10)
            if self.proc.is_alive():
                self.proc.kill()
                self.proc.join(10)
        self.proc = None

    def free(self):
        with self.lock:  # идущий кружок дорисуется, потом процесс умрёт
            busy = self.proc is not None
            self._kill()
            return busy


cards: dict[int, Card] = {}
cards_lock = threading.Lock()


def card(device):
    with cards_lock:
        return cards.setdefault(device, Card(device))


app = FastAPI()


@app.get("/health")
def health():
    return {"ok": True, "loaded": [d for d, c in cards.items() if c.proc and c.proc.is_alive()]}


@app.post("/render")
def render(script: str = Body(...), device: int = Body(0), dist: float = Body(1.9)):
    if not script.strip():
        raise HTTPException(400, "пустой сценарий")
    res = card(device).render({"script": script, "dist": dist})
    if "error" in res:
        raise HTTPException(500, res["error"])
    return Response(res["mp4"], media_type="video/mp4", headers={"X-Duration": f"{res['duration']:.2f}"})


@app.post("/free")
def free(device: int | None = Body(None, embed=True)):
    targets = [device] if device is not None else list(cards)
    return {"freed": [d for d in targets if card(d).free()]}
