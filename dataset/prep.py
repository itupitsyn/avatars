"""Подготовка к дообучению: каждую запись -> токены текста и коды звука (12 кадров в секунду × 16 слоёв).
Отпечаток голоса — с образца REF (по умолчанию voices/mika.wav): тем же образцом сервер потом и синтезирует.
Данные — из папки DATA (по умолчанию dataset/). Пишет DATA/prep.pt."""
import json, os
import torch
os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from qwen_tts import Qwen3TTSModel

ROOT = os.environ.get("DATA", os.path.dirname(os.path.abspath(__file__)))  # папка с train_raw.jsonl
BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

tts = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
model = tts.model

REF = os.environ.get("REF", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "voices", "mika.wav"))
ref, sr = sf.read(REF, dtype="float32")
ref = resample_poly(ref, 24000, sr).astype(np.float32) if sr != 24000 else ref
with torch.inference_mode():
    spk = model.extract_speaker_embedding(ref, 24000).float().cpu()

items = [json.loads(l) for l in open(os.path.join(ROOT, "train_raw.jsonl"), encoding="utf-8")]
data = []
with torch.inference_mode():
    for i, it in enumerate(items):
        wav, sr = sf.read(it["audio"], dtype="float32")
        codes = model.speech_tokenizer.encode(wav, sr=sr).audio_codes[0].cpu()  # (кадры, 16)
        ids = tts._tokenize_texts([tts._build_assistant_text(it["text"])])[0][0].cpu()
        data.append(dict(id=it["id"], ids=ids, codes=codes))
        if i % 100 == 0:
            print(i, tuple(codes.shape), len(ids), it["text"][:40], flush=True)
torch.save(dict(spk=spk, data=data), os.path.join(ROOT, "prep.pt"))
print("готово:", len(data), "записей,", sum(len(d["codes"]) for d in data) / 12 / 60, "мин")
