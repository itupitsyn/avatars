"""Проверка дообученной модели против базовой в том режиме, в котором её будет звать сервер:
потоковый синтез, только отпечаток голоса mika.wav. Меряем задержку первого звука, тембр, тон и слова.
Запуск: .venv/Scripts/python.exe dataset/evaluate.py <модель> <метка>; звук -> static/samples/ft_<метка>.wav"""
import json, os, re, sys, time
import torch
os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]
import numpy as np, soundfile as sf, librosa
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from faster_qwen3_tts import FasterQwen3TTS

ROOT = os.path.dirname(os.path.abspath(__file__))
REF = os.path.abspath(os.environ.get("REF", os.path.join(ROOT, "..", "voices", "mika.wav")))
model_id, label = sys.argv[1], sys.argv[2]
# фразы не из датасета: короткие реакции (там клон срывался чаще всего), обычные и длинные
LINES = [
    "Угу.", "Ой!", "Вот.", "Хм, дай подумать.", "Ну да.", "Правда?", "Ага, поняла.", "Ладно, пока-пока!",
    "Я сижу дома и пью чай.", "Это знакомая игра, я её обожаю!", "Ой, правда? Вот это да!",
    "Мне очень жаль, что так вышло, давай я тебя подбодрю.", "Сегодня я научилась новому танцу.",
    "А что ты любишь делать по вечерам?",
    "Знаешь, есть такие места в космосе, где время течёт медленнее, чем у нас на Земле.",
    "Глубоко под землёй плавит горные породы магма, и иногда она прорывается наружу через вулканы.",
    "Представь крыло самолёта: сверху воздух бежит быстрее, давление там ниже, и крыло тянет вверх.",
    "Если честно, я немного волнуюсь, но мне ужасно интересно, что из этого получится!",
]
if len(sys.argv) > 3 and sys.argv[3] == "short":  # срывы тона чаще всего на коротких — гоняем их сериями
    LINES = ["Угу.", "Ой!", "Вот.", "Ну да.", "Ага.", "Хм.", "Правда?", "Ого!", "Ну-у...", "Да.", "Нет.", "Ясно.",
             "Конечно!", "Ой, точно.", "Не знаю.", "Ура!", "Ладно.", "Спасибо!", "Привет!", "Пока!"]
TAKES = int(os.environ.get("TAKES", 2))
words = lambda s: re.findall(r"[а-яё]+", s.lower().replace("ё", "е"))

tts = FasterQwen3TTS.from_pretrained(model_id)
enc = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base").model.model
stt = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
emb = lambda a: torch.nn.functional.normalize(enc.extract_speaker_embedding(a.astype(np.float32), 24000).float(), dim=-1)


def pitch(a16):
    f = librosa.yin(a16, fmin=70, fmax=500, sr=16000, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(y=a16, frame_length=1024, hop_length=256)[0][: len(f)]
    f = f[rms > 0.03 * rms.max()] if len(rms) and rms.max() > 0 else []
    return float(np.median(f)) if len(f) else 0.0


ref, rsr = sf.read(REF, dtype="float32")
ref = resample_poly(ref, 24000, rsr).astype(np.float32)
ref_e, ref_hz = emb(ref), pitch(resample_poly(ref, 16000, 24000).astype(np.float32))
kw = dict(language="Russian", ref_audio=REF, xvec_only=True, chunk_size=4, temperature=0.7)
for _ in tts.generate_voice_clone_streaming(text="Проверка связи.", **kw):
    pass

rows, clips = [], []
for take in range(TAKES):
    for line in LINES:
        t = time.time(); first = None; chunks = []
        for chunk, sr, _ in tts.generate_voice_clone_streaming(text=line, max_new_tokens=int(24 + len(line) * 1.5), **kw):
            first = first or time.time() - t
            chunks.append(chunk)
        a = np.clip(np.concatenate(chunks).astype(np.float32), -1, 1)
        a16 = resample_poly(a, 16000, 24000).astype(np.float32)
        heard = " ".join(s.text for s in stt.transcribe(a16, language="ru", beam_size=5, condition_on_previous_text=False)[0])
        got, solid = set(words(heard)), "".join(words(heard))
        missed = [w for w in words(line) if len(w) >= 4 and w not in got and w not in solid]
        hz = pitch(a16)
        rows.append(dict(line=line, take=take, first_ms=round(first * 1000), sim=round(float(emb(a) @ ref_e), 3),
                         hz_ratio=round(hz / ref_hz, 2), missed=missed, heard=heard.strip()))
        r = rows[-1]
        print(f"{r['first_ms']:4d} мс  sim {r['sim']}  тон {r['hz_ratio']}  {missed or ''}  {line[:50]}", flush=True)
        if take == 0:
            clips += [a, np.zeros(int(24000 * 0.4), np.float32)]
sf.write(os.path.join(ROOT, "..", "static", "samples", f"ft_{label}.wav"), np.concatenate(clips), 24000)
summary = dict(label=label, first_ms_median=float(np.median([r["first_ms"] for r in rows])),
               sim_mean=round(float(np.mean([r["sim"] for r in rows])), 3), sim_min=min(r["sim"] for r in rows),
               drops=sum(r["hz_ratio"] < 0.72 for r in rows), misspoken=sum(bool(r["missed"]) for r in rows), n=len(rows))
print(json.dumps(summary, ensure_ascii=False))
json.dump(dict(summary=summary, rows=rows), open(os.path.join(ROOT, f"eval_{label}.json"), "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
