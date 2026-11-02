"""Сравнение на слух: Маша-Qwen (как сейчас на сервере) и Маша-Vosk (дообученная на фонемах).
Фразы — dataset/lines.py; звук Qwen берём готовый из static/samples/stress/*_old.wav (stress_ab.py).
Vosk читает текст через vosk_front: словарь + ударения омографов от RUAccent.
python vosk_ab.py <папка модели vosk> [метка] [папка]  ->  static/samples/vosk_masha/ (или <папка>)"""
import html, os, shutil, sys, time
from vosk_tts import Model, Synth

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from dataset.accent import load as load_accent  # noqa: E402
from dataset.lines import LINES  # noqa: E402
from dataset.vosk_front import Front  # noqa: E402

MODEL = sys.argv[1]
TAG = sys.argv[2] if len(sys.argv) > 2 else os.path.basename(MODEL.rstrip("/\\"))
OUT = os.path.join(ROOT, "static", "samples", sys.argv[3] if len(sys.argv) > 3 else "vosk_masha")
QWEN = os.path.join(ROOT, "static", "samples", "stress")

os.makedirs(OUT, exist_ok=True)
front = Front(load_accent(), os.path.join(MODEL, "dictionary"))
synth = Synth(Model(model_path=MODEL))
texts = [front(s) for s in LINES]
t = time.time()
for i, text in enumerate(texts):
    synth.synth(text, os.path.join(OUT, f"{i:02d}_vosk.wav"), speaker_id=0)
    shutil.copy(os.path.join(QWEN, f"{i:02d}_old.wav"), os.path.join(OUT, f"{i:02d}_qwen.wav"))
print(f"{len(LINES)} фраз за {time.time() - t:.1f} с")
rows = "".join(
    f'<section><p>{html.escape(LINES[i])}<br><small>{html.escape(texts[i])}</small></p>'
    f'<div class="ab"><label>сейчас (Qwen)<audio controls preload="none" src="{i:02d}_qwen.wav"></audio></label>'
    f'<label>Vosk<audio controls preload="none" src="{i:02d}_vosk.wav"></audio></label></div></section>'
    for i in range(len(LINES)))
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Маша: Qwen и Vosk</title><style>
:root{{--bg:#1b1730;--card:#262040;--text:#ece8f5;--muted:#a49cc0}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:820px;margin:0 auto;padding:24px 16px 48px}} small{{color:var(--muted)}}
section{{background:var(--card);border-radius:12px;padding:12px 16px;margin-bottom:10px}} p{{margin:0 0 8px}}
.ab{{display:flex;gap:16px;flex-wrap:wrap}} label{{display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:13px}}
audio{{width:300px;max-width:100%;height:36px}}
</style></head><body><main><h1>Маша: сейчас и на фонемах ({html.escape(TAG)})</h1>
<p><small>Под фразой — что получает Vosk: обычные слова как есть (ударение из словаря), омографы и незнакомые — с «+» от RUAccent.</small></p>
{rows}</main></body></html>""")
print("страница:", os.path.join(OUT, "index.html"))
