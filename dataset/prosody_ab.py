"""Ровность интонации между фразами одного ответа: одни и те же ответы, разбитые на фразы, как в разговоре.
Варианты: дообученная по отпечатку (как сейчас) при разной температуре; дообученная с образцом mika.wav в контексте;
дообученная, где образцом служит её же прошлая фраза (цепочка); старый клон базовой модели.
Пишет static/samples/prosody_<вариант>.wav и печатает разброс тона и темпа между фразами."""
import json, os, sys, tempfile
import torch
os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]
import numpy as np, soundfile as sf, librosa
from faster_qwen3_tts import FasterQwen3TTS

ROOT = os.path.dirname(os.path.abspath(__file__))
REF = os.path.abspath(os.environ.get("REF", os.path.join(ROOT, "..", "voices", "mika.wav")))
REF_TEXT = open(REF[:-4] + ".txt", encoding="utf-8").read().strip()
OUT = os.path.join(ROOT, "..", "static", "samples")
REPLIES = [
    ["Ой, тема моя!", "Знаешь, есть такие места в космосе, где время течёт медленнее, чем у нас на Земле.",
     "Представь, если бы ты полетел к чёрной дыре и вернулся,", "твои друзья уже бы состарились.", "Жутко и красиво одновременно, правда?"],
    ["Всё дело в форме крыла.", "Сверху воздух бежит быстрее, чем снизу,", "и давление над крылом становится ниже.",
     "Эта разница и толкает самолёт вверх.", "Здорово придумано, да?"],
    ["Привет-привет!", "Я так рада тебя слышать.", "Сегодня я весь день думала, чем бы нам заняться вечером.",
     "Может, расскажешь, как прошёл твой день?"],
]
VOWELS = set("аеёиоуыэюя")


def stats(a, text):
    f = librosa.yin(a, fmin=70, fmax=500, sr=24000, frame_length=2048, hop_length=480)
    rms = librosa.feature.rms(y=a, frame_length=2048, hop_length=480)[0][: len(f)]
    f = f[rms > 0.05 * rms.max()]
    voiced = librosa.effects.trim(a, top_db=35)[0]
    return np.log(np.median(f)), sum(c in VOWELS for c in text.lower()) / (len(voiced) / 24000)


def run(name, model, mode, temperature):
    clips, spread_hz, spread_rate = [], [], []
    for reply in REPLIES:
        prev, hz, rate = None, [], []
        for line in reply:
            kw = dict(text=line, language="Russian", temperature=temperature, max_new_tokens=int(24 + len(line) * 1.5))
            if mode == "xvec":
                kw.update(ref_audio=REF, xvec_only=True)
            elif mode == "icl" or prev is None:
                kw.update(ref_audio=REF, ref_text=REF_TEXT)
            else:  # цепочка: образец — прошлая фраза этого же ответа
                kw.update(ref_audio=prev[0], ref_text=prev[1])
            torch.manual_seed(len(clips))
            wavs, sr = model.generate_voice_clone(**kw)
            a = np.clip(np.asarray(wavs[0], np.float32), -1, 1)
            if mode == "chain":
                path = os.path.join(tempfile.gettempdir(), f"mika_prev_{len(clips)}.wav")
                sf.write(path, a, sr)
                prev = (path, line)
            h, r = stats(a, line)
            hz.append(h); rate.append(r)
            clips += [a, np.zeros(int(sr * 0.25), np.float32)]
        clips.append(np.zeros(int(sr * 1.0), np.float32))
        spread_hz.append(np.std(hz)); spread_rate.append(np.std(rate) / np.mean(rate))
    sf.write(os.path.join(OUT, f"prosody_{name}.wav"), np.concatenate(clips), 24000)
    res = dict(variant=name, pitch_spread_semitones=round(float(np.mean(spread_hz)) * 12 / np.log(2), 2),
               rate_spread_pct=round(float(np.mean(spread_rate)) * 100, 1))
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


# MODEL и PREFIX — для другого голоса: MODEL=models/masha-tts REF=voices/masha.wav PREFIX=masha_
MODEL, PREFIX = os.environ.get("MODEL"), os.environ.get("PREFIX", "")
results = []
if MODEL:
    ft = FasterQwen3TTS.from_pretrained(os.path.join(ROOT, "..", MODEL))
    results += [run(PREFIX + "A_xvec_t07", ft, "xvec", 0.7), run(PREFIX + "B_xvec_t05", ft, "xvec", 0.5),
                run(PREFIX + "E_chain_t05", ft, "chain", 0.5)]
else:
    ft = FasterQwen3TTS.from_pretrained(os.path.join(ROOT, "..", "models", "mika-tts"))
    results += [run("A_now_t07", ft, "xvec", 0.7), run("B_xvec_t04", ft, "xvec", 0.4),
                run("C_icl_t07", ft, "icl", 0.7), run("D_icl_t04", ft, "icl", 0.4), run("E_chain_t05", ft, "chain", 0.5)]
    del ft
    torch.cuda.empty_cache()
    base = FasterQwen3TTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    results.append(run("F_old_clone", base, "icl", 0.7))
json.dump(results, open(os.path.join(ROOT, PREFIX + "prosody_ab.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
