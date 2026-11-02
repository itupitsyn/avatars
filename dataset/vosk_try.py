"""Проба Vosk-TTS как есть: те же фразы, что в stress_ab.py, всеми голосами готовой модели.
Звук и страничка -> static/samples/vosk/. python vosk_try.py [папка модели]"""
import html, os, sys, time
from vosk_tts import Model, Synth

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from dataset.lines import LINES  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "models", "vosk", "vosk-model-tts-ru-0.7-multi")
OUT = os.path.join(ROOT, "static", "samples", "vosk")
VOICES = {0: "F01", 1: "F02", 2: "F03", 3: "M01", 4: "M02"}

os.makedirs(OUT, exist_ok=True)
synth = Synth(Model(model_path=MODEL))
t = time.time()
for i, text in enumerate(LINES):
    for sid in VOICES:
        synth.synth(text, os.path.join(OUT, f"{i:02d}_{sid}.wav"), speaker_id=sid)
print(f"{len(LINES) * len(VOICES)} фраз за {time.time() - t:.1f} с")
rows = "".join(f'<section><p>{html.escape(text)}</p><div class="ab">' + "".join(
    f'<label>{VOICES[s]}<audio controls preload="none" src="{i:02d}_{s}.wav"></audio></label>' for s in VOICES)
    + "</div></section>" for i, text in enumerate(LINES))
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Vosk-TTS как есть</title><style>
:root{{--bg:#1b1730;--card:#262040;--text:#ece8f5;--muted:#a49cc0}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:980px;margin:0 auto;padding:24px 16px 48px}}
section{{background:var(--card);border-radius:12px;padding:12px 16px;margin-bottom:10px}} p{{margin:0 0 8px}}
.ab{{display:flex;gap:12px;flex-wrap:wrap}} label{{display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:13px}}
audio{{width:170px;height:34px}}
</style></head><body><main><h1>Vosk-TTS как есть, до обучения</h1>{rows}</main></body></html>""")
