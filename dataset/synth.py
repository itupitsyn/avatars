"""Озвучка датасета эталонным голосом Мики: Qwen3-TTS VoiceDesign с тем же описанием, что дал voices/mika.wav.
Голос по описанию каждый раз чуть другой, поэтому каждую реплику пробуем до TRIES раз и берём дубль,
только если он прошёл все проверки: тембр близок к образцу, тон не просел, whisper расслышал все слова.
Пишет dataset/wavs/NNNN.wav и dataset/train_raw.jsonl; повторный запуск продолжает с места остановки.
Запуск: ttslab/.venv/Scripts/python.exe dataset/synth.py [сколько реплик]"""
import json, os, re, sys, time
import torch
os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]
import numpy as np, soundfile as sf, librosa
from scipy.signal import resample_poly
from faster_whisper import WhisperModel
from faster_qwen3_tts import FasterQwen3TTS

ROOT = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(ROOT, "..", "voices", "mika.wav")
INSTRUCT = ("A young Russian woman with a clear, soft, slightly low and calm voice, friendly and caring, "
            "like a smart voice assistant; natural conversational intonation, not robotic.")
TRIES = 4
MIN_SIM = float(os.environ.get("MIN_SIM", 0.95))   # косинус отпечатка голоса с образцом; у удачных дублей 0.96–0.99, у «чужих» 0.88–0.93
PITCH_LO, PITCH_HI = 0.78, 1.35  # тон в этих долях от образца: спокойные фразы ниже, восклицания выше; срывы — ~0.5–0.6
LEVEL = 0.1      # громкость всех записей к одному уровню (RMS)
SR = 24000

texts = [l.strip() for l in open(os.path.join(ROOT, "texts.txt"), encoding="utf-8") if l.strip()]
limit = int(sys.argv[1]) if len(sys.argv) > 1 else len(texts)
os.makedirs(os.path.join(ROOT, "wavs"), exist_ok=True)
out_path, log_path = os.path.join(ROOT, "train_raw.jsonl"), os.path.join(ROOT, "rejected.jsonl")
done = set()
for p in (out_path, log_path):
    if os.path.exists(p):
        done |= {json.loads(l)["id"] for l in open(p, encoding="utf-8")}

design = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
enc = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base").model.model  # только отпечатки голоса
stt = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")

words = lambda s: re.findall(r"[а-яё]+", s.lower().replace("ё", "е"))
# whisper пишет числа цифрами («8», «20%»), поэтому числительные сверяем только по наличию цифр
NUMERAL = re.compile(r"^(нол|нул|од[ин]|дв[аеуо]|тр[иеё]|четыр|пят|шест|сем|восем|девят|десят|сорок|девяност|сот|ст[аоуе]|тысяч|миллион|полтор|процент)")


def embed(a):
    return torch.nn.functional.normalize(enc.extract_speaker_embedding(a.astype(np.float32), SR).float(), dim=-1)


def pitch(a16):
    f = librosa.yin(a16, fmin=70, fmax=500, sr=16000, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(y=a16, frame_length=1024, hop_length=256)[0][: len(f)]
    f = f[rms > 0.03 * rms.max()] if len(rms) and rms.max() > 0 else []
    return float(np.median(f)) if len(f) else 0.0


def trim(a, pad=0.15):
    """Тишину по краям — до pad секунд: модель учится и на паузах, длинные хвосты ей ни к чему."""
    _, (s, e) = librosa.effects.trim(a, top_db=40, frame_length=512, hop_length=128)
    return a[max(0, s - int(pad * SR)): min(len(a), e + int(pad * SR))]


ref, rsr = sf.read(REF, dtype="float32")
ref = resample_poly(ref, SR, rsr).astype(np.float32)
ref_e, ref_hz = embed(ref), pitch(resample_poly(ref, 16000, SR).astype(np.float32))
print(f"образец: тон {ref_hz:.0f} Гц", flush=True)
design.generate_voice_design(text="Проверка связи.", instruct=INSTRUCT, language="Russian")

t0, total = time.time(), 0.0
with open(out_path, "a", encoding="utf-8") as out, open(log_path, "a", encoding="utf-8") as rej:
    for i, text in enumerate(texts[:limit]):
        uid = f"{i:04d}"
        if uid in done:
            continue
        best, reasons = None, []
        for take in range(TRIES):
            torch.manual_seed(1234 + 7919 * take)
            wavs, sr = design.generate_voice_design(text=text, instruct=INSTRUCT, language="Russian",
                                                    temperature=0.7, max_new_tokens=int(24 + len(text) * 1.5))
            a = trim(np.clip(np.asarray(wavs[0], np.float32), -1, 1))
            a16 = resample_poly(a, 16000, SR).astype(np.float32)
            segs, _ = stt.transcribe(a16, language="ru", beam_size=5, condition_on_previous_text=False)
            heard = " ".join(s.text for s in segs).strip()
            got, solid = set(words(heard)), "".join(words(heard))
            digits = bool(re.search(r"\d", heard))
            missed = [w for w in words(text) if len(w) >= 3 and w not in got and w not in solid
                      and not (digits and NUMERAL.match(w))]
            sim, hz = float(embed(a) @ ref_e), pitch(a16)
            ok_hz = PITCH_LO * ref_hz < hz < PITCH_HI * ref_hz
            sec_per_char = len(a) / SR / max(1, len(text))
            ok_rate = 0.035 * len(text) < len(a) / SR < 0.12 * len(text) + 0.8  # зациклилась или проглотила полфразы; короткому «угу» — запас
            reasons.append(dict(take=take, sim=round(sim, 3), hz=round(hz), missed=missed, heard=heard,
                                rate=round(sec_per_char, 3)))
            if sim >= MIN_SIM and ok_hz and ok_rate and not missed:
                best = (a, sim, hz, take)
                break
        if best is None:
            rej.write(json.dumps(dict(id=uid, text=text, takes=reasons), ensure_ascii=False) + "\n"); rej.flush()
            print(f"{uid} ✗ {text[:60]}  {[(r['sim'], r['hz'], r['missed'][:2]) for r in reasons]}", flush=True)
            continue
        a, sim, hz, take = best
        a = np.clip(a * (LEVEL / (np.sqrt(np.mean(a ** 2)) + 1e-9)), -0.99, 0.99)
        path = os.path.join(ROOT, "wavs", f"{uid}.wav")
        sf.write(path, a, SR, subtype="PCM_16")
        total += len(a) / SR
        out.write(json.dumps(dict(id=uid, audio=path, text=text, ref_audio=os.path.abspath(REF),
                                  sim=round(sim, 3), hz=round(hz), take=take), ensure_ascii=False) + "\n"); out.flush()
        el = time.time() - t0
        print(f"{uid} ✓ дубль {take} sim {sim:.3f} {hz:.0f} Гц | собрано {total / 60:.1f} мин за {el / 60:.1f} мин | {text[:50]}",
              flush=True)
