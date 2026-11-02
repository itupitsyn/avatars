"""Сравнение на слух: Маша без меток (masha-tts) и Маша на ударениях (masha-stress-tts, текст через RUAccent).
Тот же режим, что у сервера: только отпечаток голоса masha.wav. Звук и страничка -> static/samples/stress/.
Запуск: .venv/Scripts/python.exe dataset/stress_ab.py"""
import html, os, sys
import torch
os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]
import numpy as np, soundfile as sf
from faster_qwen3_tts import FasterQwen3TTS

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
from dataset.accent import load as load_accent

OUT = os.path.join(ROOT, "static", "samples", "stress")
REF = os.path.join(ROOT, "voices", "masha.wav")
REF_TEXT = open(REF[:-4] + ".txt", encoding="utf-8").read().strip()
from dataset.lines import LINES  # noqa: E402
SAMPLING = dict(temperature=0.7, xvec_only=True)

os.makedirs(OUT, exist_ok=True)
acc = load_accent()
marked = [acc.process_all(s) for s in LINES]
del acc
for tag, model, texts in [("old", "masha-tts", LINES), ("new", "masha-stress-tts", marked)]:
    tts = FasterQwen3TTS.from_pretrained(os.path.join(ROOT, "models", model))
    for i, text in enumerate(texts):
        wavs, sr = tts.generate_voice_clone(text=text, language="Russian", ref_audio=REF, ref_text=REF_TEXT,
                                            max_new_tokens=int(24 + len(LINES[i]) * 1.5), **SAMPLING)
        sf.write(os.path.join(OUT, f"{i:02d}_{tag}.wav"), np.clip(np.asarray(wavs[0], np.float32), -1, 1), sr)
        print(tag, i, text, flush=True)
    del tts
    torch.cuda.empty_cache()

rows = "".join(
    f'<section><p>{html.escape(LINES[i])}<br><small>{html.escape(marked[i])}</small></p>'
    f'<div class="ab"><label>сейчас<audio controls preload="none" src="{i:02d}_old.wav"></audio></label>'
    f'<label>с ударениями<audio controls preload="none" src="{i:02d}_new.wav"></audio></label></div></section>'
    for i in range(len(LINES)))
page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ударения Маши</title><style>
:root{{--bg:#1b1730;--card:#262040;--text:#ece8f5;--muted:#a49cc0}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:820px;margin:0 auto;padding:24px 16px 48px}} small{{color:var(--muted)}}
section{{background:var(--card);border-radius:12px;padding:12px 16px;margin-bottom:10px}} p{{margin:0 0 8px}}
.ab{{display:flex;gap:16px;flex-wrap:wrap}} label{{display:flex;flex-direction:column;gap:4px;color:var(--muted);font-size:13px}}
audio{{width:300px;max-width:100%;height:36px}}
</style></head><body><main><h1>Маша: сейчас и с ударениями</h1>
<p><small>Под фразой — как её разметил RUAccent («+» перед ударной гласной). Слушай ударения и интонацию.</small></p>
{rows}</main></body></html>"""
open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(page)
print("страница:", os.path.join(OUT, "index.html"))
