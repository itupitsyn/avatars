"""Датасет Маши для дообучения Vosk-TTS: звук 22050 Гц моно, тексты через vosk_front (ударения омографов
и незнакомых слов — от RUAccent). Пишет db-masha/ рядом с кодом обучения (vosk-train/training/vits2).
Запуск из папки vits2 (код обучения читает db/dictionary относительно неё):
PYTHONUTF8=1 python ../../../dataset/vosk_prep.py"""
import json, os, re, sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.getcwd())
from dataset.accent import load as load_accent  # noqa: E402
from dataset.vosk_front import Front  # noqa: E402
from text import text_to_sequence_g2p  # noqa: E402  код обучения: тем же путём он будет читать тексты

SRC = os.path.join(ROOT, "data", "masha", "train_raw.jsonl")
OUT = "db-masha"
SPEAKER = 0  # место первого голоса стартовой модели
DEV_EVERY = 100

os.makedirs(os.path.join(OUT, "wavs"), exist_ok=True)
front = Front(load_accent(), "db/dictionary")
rows, dropped = [], []
for r in map(json.loads, open(SRC, encoding="utf-8")):
    if re.search(r"[a-zA-Z]", r["text"]):  # латиницу vosk не читает
        dropped.append((r["id"], "латиница"))
        continue
    text = front(r["text"])
    try:
        text_to_sequence_g2p(text)
    except KeyError as e:
        dropped.append((r["id"], f"нет фонемы {e}"))
        continue
    wav = os.path.join(OUT, "wavs", r["id"] + ".wav")
    if not os.path.exists(wav):
        audio, sr = sf.read(r["audio"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        sf.write(wav, np.clip(resample_poly(audio, 22050, sr), -1, 1), 22050, subtype="PCM_16")
    rows.append(f"{wav}|{SPEAKER}|{text}|_")
    if len(rows) % 500 == 1:
        print(len(rows), text, flush=True)

with open(os.path.join(OUT, "metadata.csv.train"), "w", encoding="utf-8") as tr, \
        open(os.path.join(OUT, "metadata.csv.dev"), "w", encoding="utf-8") as dev:
    for i, line in enumerate(rows):
        (dev if i % DEV_EVERY == 1 else tr).write(line + "\n")
print("готово:", len(rows), "записей, отброшено", len(dropped), dropped[:10])
