"""Тот же звук, новый текст: берёт готовый prep.pt (коды звука и отпечаток голоса) и заново
токенизирует тексты из train_raw.jsonl другой папки — например, с ударениями (accent.py).
Звук второй раз не кодируем. python retok.py <папка с prep.pt> <папка с новым train_raw.jsonl>"""
import json, os, sys
import torch
os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]
from qwen_tts import Qwen3TTSModel

src, dst = sys.argv[1], sys.argv[2]
tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base", device_map="cpu", dtype=torch.bfloat16)
prep = torch.load(os.path.join(src, "prep.pt"))
texts = {r["id"]: r["text"] for r in map(json.loads, open(os.path.join(dst, "train_raw.jsonl"), encoding="utf-8"))}
for d in prep["data"]:
    d["ids"] = tts._tokenize_texts([tts._build_assistant_text(texts[d["id"]])])[0][0].cpu()
torch.save(prep, os.path.join(dst, "prep.pt"))
print("готово:", len(prep["data"]), "записей; пример:", texts[prep["data"][0]["id"]])
