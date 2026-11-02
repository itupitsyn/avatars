"""Записи Маши из корпуса Dialogs (langswap/dialogs-ru-emotional-conversations, OpenRAIL) -> датасет для train.py.
Берём разговорные реплики (без стихов, скороговорок и шёпота), приводим к 24 кГц моно, режем тишину по краям,
выравниваем громкость. Отсеиваем то, где whisper слышит не тот текст (актёры местами импровизировали),
и то, где голос не её (по отпечатку). Образец для сервера — самая типичная нейтральная реплика 6–12 с.
Пишет data/masha/{wavs, train_raw.jsonl, rejected.jsonl} и voices/masha.{wav,txt}."""
import csv, json, os, re
import torch
os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]
import numpy as np, soundfile as sf, librosa
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from faster_qwen3_tts import FasterQwen3TTS

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "data", "dialogs")
OUT = os.path.join(HERE, "..", "data", "masha")
VOICES = os.path.join(HERE, "..", "voices")
SKIP = {"poem", "sgorogovrki", "whisper"}  # стихи, скороговорки и шёпот — не разговорная манера
SR, LEVEL, MIN_S, MAX_S = 24000, 0.1, 0.6, 18.0
MIN_HIT, MAX_EXTRA, MIN_SIM = 0.9, 0.2, 0.8

words = lambda s: re.findall(r"[а-яё]+", s.lower().replace("ё", "е"))
NUMERAL = re.compile(r"^(нол|нул|од[ин]|дв[аеуо]|тр[иеё]|четыр|пят|шест|сем|восем|девят|десят|сорок|девяност|сот|ст[аоуе]|тысяч|миллион|полтор|процент)")


def match(text, heard):
    """Доля слов текста, которые whisper расслышал, и доля лишних слов в услышанном."""
    t, h = words(text), words(heard)
    digits = bool(re.search(r"\d", heard))
    hs, ts, solid = set(h), set(t), "".join(h)
    hit = [bool(w in hs or w in solid or (digits and NUMERAL.match(w))) for w in t if len(w) >= 3]
    extra = [w for w in h if len(w) >= 3 and w not in ts and w not in "".join(t)]
    return (np.mean(hit) if hit else 1.0), len(extra) / max(1, len(h))


rows = [r for r in csv.DictReader(open(os.path.join(SRC, "metadata.csv"), encoding="utf-8"), delimiter="|")
        if r["speaker_id"] == "M" and r["emotion"] not in SKIP]
stt = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
enc = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base").model.model
os.makedirs(os.path.join(OUT, "wavs"), exist_ok=True)

kept, rejected = [], []
for i, r in enumerate(rows):
    path = os.path.join(SRC, r["audio_path"])
    if not os.path.exists(path):
        rejected.append(dict(r, why="нет файла")); continue
    a, sr = sf.read(path, dtype="float32")
    a = a.mean(1) if a.ndim > 1 else a
    a = resample_poly(a, SR, sr).astype(np.float32)
    _, (s, e) = librosa.effects.trim(a, top_db=40, frame_length=512, hop_length=128)
    a = a[max(0, s - int(0.15 * SR)): e + int(0.15 * SR)]
    dur = len(a) / SR
    if not MIN_S <= dur <= MAX_S:
        rejected.append(dict(r, why=f"длина {dur:.1f} с")); continue
    text = re.sub(r"\s+-\s+", " — ", r["text"]).strip()
    a16 = resample_poly(a, 16000, SR).astype(np.float32)
    heard = " ".join(x.text for x in stt.transcribe(a16, language="ru", beam_size=5, condition_on_previous_text=False)[0]).strip()
    hit, extra = match(text, heard)
    emb = torch.nn.functional.normalize(enc.extract_speaker_embedding(a, SR).float(), dim=-1).cpu()
    item = dict(id=f"m{i:04d}", text=text, emotion=r["emotion"], dur=round(dur, 2), hit=round(float(hit), 2),
                extra=round(extra, 2), heard=heard, src=r["audio_path"])
    if hit < MIN_HIT or extra > MAX_EXTRA:
        rejected.append(dict(item, why="текст не совпал")); continue
    a = np.clip(a * (LEVEL / (np.sqrt(np.mean(a ** 2)) + 1e-9)), -0.99, 0.99)
    item["audio"] = os.path.abspath(os.path.join(OUT, "wavs", item["id"] + ".wav"))
    sf.write(item["audio"], a, SR, subtype="PCM_16")
    kept.append((item, emb))
    if i % 250 == 0:
        print(f"{i}/{len(rows)}: оставлено {len(kept)}, отсеяно {len(rejected)}", flush=True)

# голос не её: далеко от среднего отпечатка
centre = torch.nn.functional.normalize(torch.stack([e for _, e in kept]).mean(0), dim=-1)
final = []
for item, e in kept:
    item["sim"] = round(float(e @ centre), 3)
    (final if item["sim"] >= MIN_SIM else rejected).append(item if item["sim"] >= MIN_SIM else dict(item, why="чужой голос"))
with open(os.path.join(OUT, "train_raw.jsonl"), "w", encoding="utf-8") as f:
    for item in final:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
with open(os.path.join(OUT, "rejected.jsonl"), "w", encoding="utf-8") as f:
    for item in rejected:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

ref = max((it for it in final if it["emotion"] == "neutral" and 6 <= it["dur"] <= 12 and it["hit"] == 1.0),
          key=lambda it: it["sim"])
a, _ = sf.read(ref["audio"], dtype="float32")
sf.write(os.path.join(VOICES, "masha.wav"), a, SR, subtype="PCM_16")
open(os.path.join(VOICES, "masha.txt"), "w", encoding="utf-8").write(ref["text"])
sims = [it["sim"] for it in final]
print(f"готово: {len(final)} записей, {sum(it['dur'] for it in final) / 3600:.2f} ч; отсеяно {len(rejected)}; "
      f"сходство с центром: 5% {np.percentile(sims, 5):.3f}, медиана {np.median(sims):.3f}; образец {ref['id']}: {ref['text']}")
