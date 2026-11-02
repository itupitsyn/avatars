"""Голос Мики без сервера: текст -> звук с проверками и расписание рта для губ аватара.

Общий код для server.py (живой разговор) и video/note.py (кружочки). Ударения для губ — RUAccent (MIT),
а не разметка Silero, как в разговоре: у Silero некоммерческая лицензия, а ролики — товар.

    voice = Voice("models/masha-calm-tts", "voices/masha.wav")
    clip = voice.script("[greet] Привет! Это Мика. [wink] До завтра!")
    clip["lines"]  # [{start, end, text, mouth, acts}] — расписание для страницы (?render=1, mikaLoad)
    clip["audio"]  # float32, clip["rate"] Гц — звуковая дорожка того же ролика
"""
import re
import time

import librosa
import numpy as np
from num2words import num2words
from scipy.signal import resample_poly

STT_RATE = 16000
OUT_RATE = 48000

# Формы рта VRM под русские буквы: (форма, раскрытие). Ударная гласная открывается
# полностью, безударная — слабее и с редукцией: «о» звучит как «а», «е» как «и».
_STRESSED = {"а": ("aa", 1.0), "я": ("aa", 1.0), "о": ("oh", 1.0), "ё": ("oh", 1.0), "у": ("ou", 1.0),
             "ю": ("ou", 1.0), "и": ("ih", 0.9), "ы": ("ih", 0.8), "э": ("ee", 1.0), "е": ("ee", 0.9)}
_UNSTRESSED = {"а": ("aa", 0.6), "я": ("aa", 0.5), "о": ("aa", 0.55), "ё": ("oh", 0.8), "у": ("ou", 0.7),
               "ю": ("ou", 0.7), "и": ("ih", 0.55), "ы": ("ih", 0.5), "э": ("ee", 0.6), "е": ("ih", 0.5)}
_LIPS_SHUT = set("мбп")  # губы сомкнуты — без этого липсинка не видно
_LIPS_ROUND = set("шжчщ")
_LETTERS = set("абвгдежзийклмнопрстуфхцчшщъыьэюяё")
_VOWELS = set("аеёиоуыэюя")
_SHORT_GAP = 0.12  # пробел между словами короче этого — рот не закрываем, человек тоже не закрывает


def mouth_cues(symbols, frames, seconds):
    """Буквы с длительностями -> [[начало, конец, форма, раскрытие], ...] в секундах.

    Рот ведут гласные. Согласная не захлопывает его, а держит приоткрытым в форме соседней
    гласной — иначе рот дёргается на каждой букве, десять раз в секунду.
    """
    per_frame = seconds / max(sum(frames), 1)
    spans, t, stressed, carried = [], 0.0, False, None  # [начало, конец, буква, форма, раскрытие]
    for ch, n in zip(symbols, frames):
        start = t if carried is None else carried
        t += n * per_frame
        if ch == "+":  # знак ударения стоит перед гласной, его время отдаём ей
            stressed, carried = True, start
            continue
        carried = None
        shape = (_STRESSED if stressed else _UNSTRESSED).get(ch)
        stressed = False
        spans.append([start, t, ch, shape])

    def vowel_near(i):
        """Ближайшая гласная: сначала вперёд по слову (рот готовится заранее), потом назад."""
        for step in (1, -1):
            j = i + step
            while 0 <= j < len(spans) and (spans[j][2] in _LETTERS or spans[j][1] - spans[j][0] < _SHORT_GAP and spans[j][2] == " "):
                if spans[j][3]:
                    return spans[j][3]
                j += step
        return None

    cues = []
    for i, (start, end, ch, shape) in enumerate(spans):
        if shape is None:
            near = vowel_near(i)
            if ch in _LIPS_SHUT or near is None or (ch not in _LETTERS and not (ch == " " and end - start < _SHORT_GAP)):
                shape = ("aa", 0.0)
            elif ch in _LIPS_ROUND:
                shape = ("ou", 0.35)
            else:
                shape = (near[0], round(near[1] * (0.3 if ch == " " else 0.42), 2))
        if cues and cues[-1][2:] == list(shape):
            cues[-1][1] = round(end, 3)
        else:
            cues.append([round(start, 3), round(end, 3), *shape])
    return cues


def cues_from_words(words, seconds, stress):
    """Слова со временем -> буквы с длительностями -> mouth_cues.
    Внутри слова время делится между буквами: гласные тянутся дольше согласных.
    stress: слово без ударения (ё -> е) -> оно же с «+» перед ударной гласной."""
    symbols, durs, t = [], [], 0.0
    for start, end, word in words:
        letters = "".join(c for c in word.lower() if c in _LETTERS)
        start = max(start, t)
        if not letters or end <= start:
            continue
        if start > t:
            symbols.append(" ")
            durs.append(start - t)
        marked = stress.get(letters.replace("ё", "е"))
        if not marked:  # слова нет в тексте ответа: ударение ставим, только если гласная одна
            vowels = [i for i, c in enumerate(letters) if c in _VOWELS]
            marked = letters[: vowels[0]] + "+" + letters[vowels[0] :] if len(vowels) == 1 else letters
        weights = [0 if c == "+" else 1.7 if c in _VOWELS else 0.3 if c in "ьъ" else 1.0 for c in marked]
        for c, w in zip(marked, weights):
            symbols.append(c)
            durs.append((end - start) * w / sum(weights))
        t = end
    if not symbols:
        return []
    if seconds > t:
        symbols.append(" ")
        durs.append(seconds - t)
    return mouth_cues(symbols, durs, sum(durs))


def readable(text: str) -> str:
    """Текст для Qwen: ударения и интонацию он ставит сам, но цифры и значки не читает."""
    text = re.sub(r"\d+", lambda m: " " + num2words(int(m.group()), lang="ru") + " ", text)
    text = re.sub(r"[^\w\s.,!?…:;\-—«»\"'()]|_", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def pitch(audio16) -> float:
    """Средняя высота голоса по звонким кадрам, Гц; 0 — звонких кадров нет."""
    f = librosa.yin(audio16, fmin=70, fmax=500, sr=STT_RATE, frame_length=1024, hop_length=256)
    rms = librosa.feature.rms(y=audio16, frame_length=1024, hop_length=256)[0][: len(f)]
    f = f[rms > 0.03 * rms.max()] if len(rms) and rms.max() > 0 else []
    return float(np.median(f)) if len(f) else 0.0


_WORD = re.compile(r"[а-яёa-z]+", re.IGNORECASE)
# числительные: whisper пишет их цифрами («31 октября»), словами в его тексте их не найти
_NUMERAL = re.compile(r"^(нол|нул|од[ин]|перв|втор|трет|дв[аеуо]|тр[иеё]|четыр|четв|пят|шест|сем|восем|девят|десят|"
                      r"сорок|девяност|сот|ст[аоуе]|тысяч|миллион|полтор)")


def misspoken(text, heard):
    """Слова от четырёх букв, которых whisper не расслышал в готовом звуке: «знакомая» -> «знаковая».
    Слитное и раздельное написание не в счёт: «супер-клей» он пишет «суперклей», и наоборот."""
    norm = lambda s: [w.lower().replace("ё", "е") for w in _WORD.findall(s)]
    got, solid = set(norm(heard)), "".join(norm(heard))
    digits = bool(re.search(r"\d", heard))
    return [w for w in norm(text) if len(w) >= 4 and w not in got and w not in solid
            and not (digits and _NUMERAL.match(w))]


def word_timings(stt, audio16):
    """Когда звучит каждое слово готовой фразы — по распознаванию её же whisper'ом."""
    segments, _ = stt.transcribe(
        audio16, language="ru", beam_size=1, word_timestamps=True, vad_filter=False, condition_on_previous_text=False
    )
    words = []
    for w in (w for s in segments for w in s.words):
        piece = audio16[int(w.start * STT_RATE) : int(w.end * STT_RATE)]
        # на тишине в конце whisper иногда дописывает слова, которых нет, — рот там должен молчать
        if len(piece) and float(np.sqrt(np.mean(piece**2))) > 0.01:
            words.append((w.start, w.end, w.word))
    return words


ACT = re.compile(r"\[([a-z_]+)\]")
_SENTENCE = re.compile(r"(?<=[.!?…])[»\")]*\s+")
SHORT_LINE = 15  # символов: короче — фразу не озвучиваем отдельно


def split_script(script):
    """Сценарий с метками жестов -> [(фраза, [метки перед ней и в ней])]. Режем по предложениям: в ролике
    торопиться некуда, а на каждом стыке кусков интонация начинается заново. Метки без слов — к следующей фразе,
    в самом конце — отдельной строкой без звука."""
    parts, acts, held = [], [], ""
    for piece in _SENTENCE.split(script.strip()):
        acts += ACT.findall(piece)
        words = (held + " " + ACT.sub("", piece)).strip()
        if len(words) < SHORT_LINE:  # «Привет!» отдельным заходом звучит рвано — вместе со следующей фразой
            held = words
            continue
        held = ""
        if re.search(r"[а-яёa-z0-9]", words, re.IGNORECASE):
            parts.append((re.sub(r"\s+", " ", words), acts))
            acts = []
    if held or acts:
        parts.append((held, acts))
    return parts


class Voice:
    """Дообученная Qwen-Маша + whisper для проверок и губ + RUAccent для ударений во рту."""

    VOICE_DROP = 0.72  # ниже этой доли от тона образца — уже не она (срывы ~0.5–0.6, норма ~0.9)
    TRIES = 2  # сорвалась или исказила слово — переозвучиваем раз; whisper и сам ошибается

    def __init__(self, model_dir, ref_wav, whisper="large-v3-turbo", device="cuda", temperature=0.7, accent=None):
        import soundfile as sf
        from faster_qwen3_tts import FasterQwen3TTS
        from faster_whisper import WhisperModel

        self.tts = FasterQwen3TTS.from_pretrained(str(model_dir))
        self.stt = WhisperModel(whisper, device=device, compute_type="float16")
        self.ref, self.temperature = str(ref_wav), temperature
        ref, sr = sf.read(self.ref, dtype="float32")
        self.ref_f0 = pitch(resample_poly(ref.mean(1) if ref.ndim > 1 else ref, STT_RATE, sr).astype(np.float32))
        if accent is None:
            from dataset.accent import load as load_accent
            accent = load_accent()
        self.accent = accent

    def stress(self, text):
        """Слово (ё -> е) -> оно же с «+»: «прив+ет»."""
        marked = self.accent.process_all(readable(text)).lower()
        return {w.replace("+", "").replace("ё", "е"): w for w in re.findall(r"[а-яё+]+", marked)}

    def say(self, text):
        """Одна фраза -> (звук float32 OUT_RATE, расписание рта). Сорвался тон или не то слово — второй дубль,
        берём лучший."""
        say = readable(text)[:600]
        if not re.search(r"[а-яёa-z]", say, re.IGNORECASE):
            return np.zeros(0, np.float32), []
        t, best = time.perf_counter(), None
        for attempt in range(1, self.TRIES + 1):
            wavs, sr = self.tts.generate_voice_clone(text=say, language="Russian", ref_audio=self.ref, xvec_only=True,
                                                     temperature=self.temperature, max_new_tokens=int(24 + len(say) * 1.5))
            audio = np.clip(np.asarray(wavs[0], dtype=np.float32), -1, 1)
            audio16 = resample_poly(audio, STT_RATE, sr).astype(np.float32)
            hz, words = pitch(audio16), word_timings(self.stt, audio16)
            wrong = misspoken(say, " ".join(w[2] for w in words))
            dropped = hz < self.VOICE_DROP * self.ref_f0
            score = (dropped, len(wrong), abs(np.log(hz / self.ref_f0)) if hz else 9.0)
            if best is None or score < best[0]:
                best = (score, audio, sr, words, hz, wrong)
            if not dropped and not wrong:
                break
        _, audio, sr, words, hz, wrong = best
        print(f"[voice] {len(audio) / sr:.1f} с, {hz:.0f} Гц, {attempt} дубл., {time.perf_counter() - t:.1f} с: {say!r}"
              + (f" | не выговорила {wrong}" if wrong else ""), flush=True)
        cues = cues_from_words(words, len(audio) / sr, self.stress(text))
        return np.clip(resample_poly(audio, OUT_RATE, sr), -1, 1).astype(np.float32), cues

    def script(self, script, lead=0.5, tail=0.5):
        """Сценарий -> ролик: фразы встык, как в разговоре, с паузой lead в начале и tail в конце."""
        lines, chunks, at = [], [np.zeros(int(lead * OUT_RATE), np.float32)], lead
        for text, acts in split_script(script):
            audio, cues = self.say(text) if text else (np.zeros(0, np.float32), [])
            dur = len(audio) / OUT_RATE
            lines.append(dict(start=round(at, 4), end=round(at + max(dur, 2.0 if not text else 0), 4),
                              text=text, mouth=cues, acts=acts))
            chunks.append(audio)
            at += dur
        audio = np.concatenate(chunks + [np.zeros(int(tail * OUT_RATE), np.float32)])
        return dict(lines=lines, audio=audio, rate=OUT_RATE, duration=len(audio) / OUT_RATE)
