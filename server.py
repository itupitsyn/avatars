"""Голосовой собеседник с рисованным аватаром.

Цепочка: браузер режет речь на фразы (VAD) -> whisper -> LLM на соседнем хосте ->
Qwen3-TTS (клон голоса с образца) -> звук обратно в браузер, где под него шевелится рот аватара.
Silero — запасной голос (TTS=silero) и словарь ударений для губ.

Протокол WebSocket (/ws):
  клиент -> сервер  бинарный кадр: фраза целиком, float32 моно 16 кГц
                    {"type": "text", "text": ...}   то же, но набрано руками
                    {"type": "interrupt"}           замолчи
                    {"type": "reset"}               забудь разговор
                    {"type": "idle"}                пауза затянулась — заговори сама
                    {"type": "look", "text"}        как выглядит текущая модель аватара
  сервер -> клиент  {"type": "user", "turn", "text"}        что расслышали
                    {"type": "turn", "turn"}                начат ответ без реплики человека (на паузу)
                    {"type": "say", "turn", "text", "mouth", "acts"} + бинарный кадр int16 моно 48 кГц;
                        mouth — расписание форм рта [[начало, конец, форма, раскрытие], ...],
                        acts — жесты и эмоции, которые надо начать вместе с этой фразой;
                        первая фраза ответа приходит кусками: много «say» подряд с одним текстом,
                        пустым mouth (рот по громкости), rate — частотой звука куска, seg — номером
                        фразы и offset — началом куска в ней, с;
                    {"type": "mouth", "turn", "seg", "mouth"}  расписание рта потоковой фразы целиком, вдогонку
                    {"type": "act", "turn", "acts"}         жесты без слов
                    {"type": "done", "turn", "timing"}      ответ договорён
                    {"type": "skip", "turn"}                в записи не было речи
"""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch

# faster-whisper (ctranslate2) ищет cuDNN и cuBLAS через PATH, а лежат они в колесе torch.
_torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
os.environ["PATH"] = _torch_lib + os.pathsep + os.environ["PATH"]
os.add_dll_directory(_torch_lib)

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from num2words import num2words
from scipy.signal import resample_poly

from mika_voice import (cues_from_words as _cues_from_words, misspoken as _misspoken,
                        mouth_cues as _mouth_cues, pitch, readable, word_timings)
from silero import silero_tts

ROOT = Path(__file__).parent
LLM_URL = os.environ["LLM_URL"]
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
TTS = os.environ.get("TTS", "qwen")  # qwen — живой клон голоса; silero — запасной, мгновенный, но дикторский
# vosk — Маша, дообученная на фонемах (dataset/vosk_prep.py): ударения из словаря, у омографов — от RUAccent
VOSK_MODEL = Path(os.environ.get("VOSK_MODEL", ROOT / "models" / "vosk" / "masha"))
VOSK_SPEAKER = int(os.environ.get("VOSK_SPEAKER", 0))
# Дообученные Qwen3-TTS (dataset/train.py) и образцы к ним, по предпочтению: живая актриса Маша из корпуса Dialogs
# (OpenRAIL, data/dialogs/LICENSE.md), затем синтетический голос. Нет ни одной — клон базовой модели по mika.wav.
_VOICES = [("masha-tts", "masha.wav"), ("mika-tts", "mika.wav")]
_found = next(((ROOT / "models" / m, f) for m, f in _VOICES if (ROOT / "models" / m).exists()), (None, "mika.wav"))
QWEN_MODEL = os.environ.get("QWEN_TTS", str(_found[0]) if _found[0] else "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
FINETUNED = Path(QWEN_MODEL).exists()  # голос в самой модели: ей нужен только отпечаток образца
# Первую фразу ответа дообученная модель отдаёт кусками по мере синтеза — звук через ~0.15 с.
# Губы у неё идут по громкости: тайминги слов whisper даёт только по готовой фразе.
STREAM_FIRST = FINETUNED and os.environ.get("STREAM", "1") == "1"
# Модель, дообученная на текстах с ударениями (dataset/accent.py), говорит по меткам «+» перед ударной
# гласной: их перед синтезом ставит тот же RUAccent, по смыслу фразы («з+амок» / «зам+ок»).
_TUNE = Path(QWEN_MODEL) / "mika_finetune.json"
STRESS = FINETUNED and _TUNE.exists() and json.loads(_TUNE.read_text(encoding="utf-8")).get("stress", False)
VOICE_REF = ROOT / "voices" / os.environ.get("VOICE_REF", _found[1])  # образец; рядом .txt с его текстом
VOICE = os.environ.get("VOICE", "xenia")  # голос Silero: aidar, baya, kseniya, eugene, xenia
HISTORY_LIMIT = 30  # сообщений, не считая системного
TTS_RATE = 48000
STT_RATE = 16000

# Что whisper дописывает на тишине и шуме: это обрывки субтитров из обучающих данных.
HALLUCINATIONS = re.compile(
    r"(продолжение следует|субтитры (сделал|создавал|делал)|редактор субтитров|"
    r"корректор [а-я]\.|спасибо за просмотр|подписывайтесь на канал)[^.!?]*[.!?…]*",
    re.IGNORECASE,
)

models = {}
gpu_lock = threading.Lock()  # распознавание и синтез по очереди: карта одна, собеседник один
# Вся работа с картой — в одном потоке: CUDA-графы Qwen захватываются и проигрываются там же.
GPU = ThreadPoolExecutor(1, thread_name_prefix="gpu")


async def on_gpu(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(GPU, fn, *args)


@asynccontextmanager
async def lifespan(app):
    t = time.perf_counter()
    models["stt"] = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
    tts = silero_tts(language="ru", speaker="v5_ru")
    tts = tts[0] if isinstance(tts, tuple) else tts
    tts.to(torch.device("cuda"))
    models["tts"] = tts
    if TTS == "vosk":
        from vosk_tts import Model as VoskModel, Synth as VoskSynth
        from dataset.accent import load as load_accent
        from dataset.vosk_front import Front

        models["vosk"] = VoskSynth(VoskModel(model_path=str(VOSK_MODEL)))
        models["front"] = Front(await asyncio.to_thread(load_accent), VOSK_MODEL / "dictionary")
    if TTS == "qwen":
        from faster_qwen3_tts import FasterQwen3TTS

        models["qwen"] = await on_gpu(FasterQwen3TTS.from_pretrained, QWEN_MODEL)
        if STRESS:
            from dataset.accent import load as load_accent

            models["accent"] = await asyncio.to_thread(load_accent)
        models["ref_text"] = VOICE_REF.with_suffix(".txt").read_text(encoding="utf-8").strip()
        ref, ref_sr = sf.read(VOICE_REF, dtype="float32")
        models["ref_f0"] = pitch(resample_poly(ref, STT_RATE, ref_sr).astype(np.float32))  # заодно прогрев numba
    await on_gpu(synth, "Проверка связи.")  # первый прогон на карте стоит секунду, платим её здесь
    # trust_env=False: в окружении прописан внешний прокси, а LLM живёт в локалке
    models["http"] = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(60, connect=5))
    voice = (f"{QWEN_MODEL} по образцу {VOICE_REF.name}" if "qwen" in models
             else f"Vosk {VOSK_MODEL.name}, голос {VOSK_SPEAKER}" if "vosk" in models else f"Silero {VOICE}")
    voice += ", ударения RUAccent" if "accent" in models else ""
    print(f"[init] модели подняты за {time.perf_counter() - t:.1f} с, голос {voice}, LLM {LLM_URL}")
    yield
    await models["http"].aclose()


app = FastAPI(lifespan=lifespan)


def transcribe(audio: np.ndarray) -> str:
    with gpu_lock:
        segments, _ = models["stt"].transcribe(
            audio, language="ru", beam_size=5, vad_filter=False, condition_on_previous_text=False
        )
        text = " ".join(s.text.strip() for s in segments if s.no_speech_prob < 0.8)
    return HALLUCINATIONS.sub("", text).strip(" «»\"")


_LATIN = str.maketrans(
    {
        "a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х",
        "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п",
        "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс",
        "y": "и", "z": "з",
    }
)


def speakable(text: str) -> str:
    """Приводит текст к тому, что Silero умеет произнести: он не читает цифры и латиницу."""
    text = re.sub(r"\d+", lambda m: " " + num2words(int(m.group()), lang="ru") + " ", text)
    text = text.lower().translate(_LATIN)
    text = re.sub(r"[^а-яё\s.,!?…:;\-—]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tts_with_cues(text):
    """То же, что apply_tts, но разобранное по шагам: изнутри модель отдаёт длительность
    каждой буквы, а наружу — нет. Завязано на устройство пакета silero 0.5.x."""
    p = models["tts"].packages[models["tts"].speaker_to_package[VOICE]]
    speaker_ids, model_id = p.get_speakers(VOICE, None)
    sentences, _, breaks, rates, pitches, sp_ids = p.prepare_tts_model_input(text, ssml=False, speaker_ids=speaker_ids)
    if not p.q_model_unpacked:
        torch._C._jit_set_profiling_mode(False)
        p.unpack_q_model()
        p.q_model_unpacked = True
    with torch.no_grad():
        seq, symb_durs, durs_rate, pitch_coefs = p.merge_batch_model(sentences, breaks, rates, pitches)
        out, durs = p.models[model_id](
            sequence=seq.to(p.device), speaker_ids=sp_ids.to(p.device), sr=TTS_RATE, symb_durs=symb_durs,
            durs_rate=durs_rate.to(p.device), pitch_coefs=pitch_coefs.to(p.device), device=str(p.device),
        )
        if type(out) == tuple:
            out = p.wrapped_jit_v(out, TTS_RATE)
    audio = out[0]
    symbols = [p.symbols[i] for i in seq[0].tolist()]
    return audio, _mouth_cues(symbols, durs[0].tolist(), audio.shape[-1] / TTS_RATE)


# LLM без размышлений иногда говорит о себе в мужском роде («Рад, что…», «я понял»). Правим только
# очевидное: формы сразу после «я» (через пару наречий) и благодарность/радость в начале фразы.
_SHORT_ADJ = ("рад|готов|уверен|согласен|должен|счастлив|благодарен|свободен|занят|удивлён|удивлен|обязан|"
              "способен|знаком|влюблён|влюблен|расстроен|огорчён|огорчен|восхищён|восхищен|заинтересован|сам|один|сыт")
_FILLER = r"(?:(?:не|тоже|уже|так|очень|просто|ещё|еще|всегда|никогда|только|сначала|сегодня|вчера|правда|бы|и|же|даже|точно|сейчас|вот)\s+){0,3}"
_I_FORM = re.compile(rf"\b([Яя]\s+{_FILLER})([а-яё]+?)(л|лся|ёл|ел|мог)\b(?!-)|\b([Яя]\s+{_FILLER})({_SHORT_ADJ})\b(?![а-яё])")
_START_ADJ = re.compile(r"(^|[.!?…]\s+)(Рад|Благодарен|Счастлив|Согласен)\b(?![а-яё])(?![^.!?…]*\?)")
_FEM = {"сам": "сама", "один": "одна", "сыт": "сыта", "знаком": "знакома", "занят": "занята", "должен": "должна",
        "согласен": "согласна", "благодарен": "благодарна", "свободен": "свободна", "способен": "способна"}


def feminine(text: str) -> str:
    """«я понял» -> «я поняла», «я пришёл» -> «я пришла», «я бы хотел» -> «я бы хотела», «Рад, что…» -> «Рада, что…»."""
    def verb(stem, end):
        if end == "лся":
            return stem + "лась"
        if end in ("ёл", "ел") and stem.endswith("ш"):  # шёл, пришёл, нашёл
            return stem + "ла"
        if end == "мог":
            return stem + "могла"
        return stem + end + "а"

    def fix(m):
        if m.group(2) is not None:
            return m.group(1) + verb(m.group(2), m.group(3))
        word = m.group(5)
        return m.group(4) + _FEM.get(word.lower(), word + "а")

    text = _I_FORM.sub(fix, text)
    return _START_ADJ.sub(lambda m: m.group(1) + (_FEM.get(m.group(2).lower(), m.group(2) + "а")).capitalize(), text)


def _stress_map(text):
    """Слово -> оно же с ударением («прив+ет»): берём из разметки Silero, она знает ударения."""
    try:
        p = models["tts"].packages[models["tts"].speaker_to_package[VOICE]]
        speaker_ids, _ = p.get_speakers(VOICE, None)
        sentences, _, breaks, rates, pitches, _ = p.prepare_tts_model_input(speakable(text), ssml=False, speaker_ids=speaker_ids)
        seq = p.merge_batch_model(sentences, breaks, rates, pitches)[0]
        marked = "".join(p.symbols[i] for i in seq[0].tolist())
    except Exception:
        return {}
    return {w.replace("+", "").replace("ё", "е"): w for w in re.findall(r"[а-яё+]+", marked)}


# Клон иногда срывается в чужой низкий голос — чаще на коротких репликах вроде «угу» и «вот»:
# у него почти нет текста, чтобы держаться образца. Такую фразу переозвучиваем.
VOICE_DROP = 0.72  # ниже этой доли от тона образца — уже не она (срывы ~0.5–0.6, норма ~0.9)
VOICE_TRIES = 2 if FINETUNED else 3  # дообученная срывается в ~3% коротких фраз против ~8% у клона
WORD_TRIES = 1 if FINETUNED else 2  # из-за слова переозвучиваем раз, не больше: whisper и сам ошибается, а ждать человеку
# 0.9 по умолчанию срывается вдвое чаще. 0.4 ровнее по тембру, но на слух хуже интонации,
# а однословные ответы чаще срываются и ждут переозвучки.
# QWEN_XVEC=1 — клонировать только тембр, без манеры образца.
QWEN_SAMPLING = dict(temperature=float(os.environ.get("QWEN_TEMP", 0.7)), xvec_only=FINETUNED or os.environ.get("QWEN_XVEC") == "1")


def stressed(say):
    """Что отдать модели: тот же текст, с метками ударений, если она на них учена. Проверяем потом по чистому."""
    return models["accent"].process_all(say) if "accent" in models else say


def _synth_qwen(text):
    say = readable(text)[:600]
    if not re.search(r"[а-яёa-z]", say, re.IGNORECASE):
        return b"", []
    t = time.perf_counter()
    spoken = stressed(say)
    with gpu_lock:
        best = None
        for attempt in range(1, VOICE_TRIES + 1):
            wavs, sr = models["qwen"].generate_voice_clone(
                text=spoken, language="Russian", ref_audio=str(VOICE_REF), ref_text=models["ref_text"],
                **QWEN_SAMPLING,
                max_new_tokens=int(24 + len(say) * 1.5),  # 12 кадров в секунду; потолок — от зацикливания
            )
            audio = np.clip(np.asarray(wavs[0], dtype=np.float32), -1, 1)
            audio16 = resample_poly(audio, STT_RATE, sr).astype(np.float32)
            hz = pitch(audio16)
            words = word_timings(models["stt"], audio16)  # нужны губам, а заодно проверяют, то ли сказано
            wrong = _misspoken(say, " ".join(w[2] for w in words))
            dropped = hz < VOICE_DROP * models["ref_f0"]
            score = (dropped, len(wrong), abs(np.log(hz / models["ref_f0"])) if hz else 9.0)
            if best is None or score < best[0]:
                best = (score, audio, words, hz, wrong)
            if not dropped and (not wrong or attempt >= WORD_TRIES):
                break
            why = f"голос сорвался ({hz:.0f} Гц при образце {models['ref_f0']:.0f})" if dropped else f"исказила {wrong}"
            print(f"[tts] попытка {attempt}: {why} — переозвучиваю {say!r}")
        _, audio, words, hz, wrong = best
        stress = _stress_map(text)
    cues = _cues_from_words(words, len(audio) / sr, stress)
    print(f"[tts] {len(audio) / sr:.1f} с речи, {hz:.0f} Гц, за {time.perf_counter() - t:.2f} с: {say!r}"
          + (f" | так и не выговорила {wrong}" if wrong else ""))
    pcm = np.clip(resample_poly(audio, TTS_RATE, sr), -1, 1)
    return (pcm * 32767).astype(np.int16).tobytes(), cues


VOSK_RATE = 22050


def _synth_vosk(text):
    say = models["front"](text)[:900]
    if not re.search(r"[а-яё]", say, re.IGNORECASE):
        return b"", []
    t = time.perf_counter()
    audio = models["vosk"].synth_audio(say, speaker_id=VOSK_SPEAKER).astype(np.float32) / 32768
    with gpu_lock:  # губам — слова с таймингами от whisper, как у Qwen
        words = word_timings(models["stt"], resample_poly(audio, STT_RATE, VOSK_RATE).astype(np.float32))
        cues = _cues_from_words(words, len(audio) / VOSK_RATE, _stress_map(text))
    print(f"[tts] vosk: {len(audio) / VOSK_RATE:.1f} с речи за {time.perf_counter() - t:.2f} с: {say!r}")
    pcm = np.clip(resample_poly(audio, TTS_RATE, VOSK_RATE), -1, 1)
    return (pcm * 32767).astype(np.int16).tobytes(), cues


def synth(text: str) -> tuple[bytes, list]:
    """Возвращает звук и расписание форм рта; без расписания клиент шевелит ртом по громкости."""
    if "vosk" in models:
        try:
            return _synth_vosk(text)
        except Exception as e:
            print(f"[tts] Vosk не смог, фраза пойдёт голосом Silero: {e!r}")
    if "qwen" in models:
        try:
            return _synth_qwen(text)
        except Exception as e:
            print(f"[tts] Qwen не смог, фраза пойдёт голосом Silero: {e!r}")
    return _synth_silero(text)


STREAM_RATE = 24000  # куски потока шлём как есть: пересэмплинг по кускам щёлкает на стыках


def stream_qwen(text, emit, stop: threading.Event):
    """Синтез первой фразы кусками по ~0.33 с: emit(pcm int16) на каждый кусок, пока не попросили замолчать.
    Проверок нет: переозвучить уже звучащее нельзя, а дообученная модель срывается редко.
    Возвращает расписание рта для всей фразы: whisper размечает её, когда она досинтезирована, —
    обычно она ещё звучит, и дальше губы идут по словам, а не по громкости. Перебили — None."""
    say = readable(text)[:600]
    if not re.search(r"[а-яёa-z]", say, re.IGNORECASE):
        return None
    t, first, chunks = time.perf_counter(), None, []
    spoken = stressed(say)
    with gpu_lock:
        for chunk, sr, _ in models["qwen"].generate_voice_clone_streaming(
            text=spoken, language="Russian", ref_audio=str(VOICE_REF), chunk_size=4,
            max_new_tokens=int(24 + len(say) * 1.5), **QWEN_SAMPLING,
        ):
            if stop.is_set():
                return None
            first = first or time.perf_counter() - t
            pcm = np.clip(resample_poly(chunk, STREAM_RATE, sr) if sr != STREAM_RATE else chunk, -1, 1)
            chunks.append(pcm)
            emit((pcm * 32767).astype(np.int16).tobytes())
        if not chunks:
            return None
        audio = np.concatenate(chunks).astype(np.float32)
        took = time.perf_counter() - t
        words = word_timings(models["stt"], resample_poly(audio, STT_RATE, STREAM_RATE).astype(np.float32))
        cues = _cues_from_words(words, len(audio) / STREAM_RATE, _stress_map(text))
    print(f"[tts] поток: первый звук через {first:.2f} с, {len(audio) / STREAM_RATE:.1f} с речи за {took:.2f} с, "
          f"губы через {time.perf_counter() - t:.2f} с: {say!r}")
    return cues


_cues_broken = False


def _synth_silero(text: str) -> tuple[bytes, list]:
    global _cues_broken
    text = speakable(text)[:900]
    if not re.search(r"[а-яё]", text):
        return b"", []
    with gpu_lock:
        audio, cues = None, []
        if not _cues_broken:
            try:
                audio, cues = _tts_with_cues(text)
            except Exception as e:
                _cues_broken = True
                print(f"[tts] длительности букв недоступны, рот пойдёт по громкости: {e!r}")
        if audio is None:
            audio = models["tts"].apply_tts(text=text, speaker=VOICE, sample_rate=TTS_RATE)
    return (audio.clamp(-1, 1) * 32767).short().cpu().numpy().tobytes(), cues


async def llm_stream(messages):
    body = {
        "messages": messages,
        "stream": True,
        "max_tokens": 400,
        "temperature": 0.8,
        "reasoning_effort": "none",  # без этого Qwen сначала думает, и до первого слова секунды
        # Сервер на .116 поднят с presence_penalty 1.5 — это приехало блоком
        # «рекомендованных» параметров Qwen3 в самый первый запуск и нигде не
        # задокументировано, а мы наследовали его молча. Замер 25.10.2026: штраф
        # не даёт ничего (склонения, иероглифы и повторы при 1.5 и 0 одинаковы,
        # залипания без него не случилось ни разу — даже на тексте с
        # принудительным повтором имени), зато сами Qwen предупреждают, что
        # высокий penalty подмешивает языки, и порча слов у Мики («магнема»
        # вместо «магма») вылезала только при 1.5. Поэтому здесь ноль, явно.
        # Бота это не касается: флаг сервера не тронут.
        "presence_penalty": 0,
    }
    async with models["http"].stream("POST", LLM_URL, json=body) as r:
        if r.status_code != 200:
            raise RuntimeError(f"LLM ответила {r.status_code}: {(await r.aread()).decode(errors='replace')[:300]}")
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content")
            if delta:
                yield delta


# Метки тела в ответе LLM: [wave], [happy] и т.п. Список и смысл — в persona.txt,
# исполнение — в static/index.html; сервер их только вынимает из текста.
_ACT = re.compile(r"\[([a-z_]+)\]")
_SENTENCE_END = re.compile(r"[.!?…]+[»\")]*\s+|\n+")
_CLAUSE_END = re.compile(r"[,;:]\s+|\s+[—–]\s+")
FIRST_CHUNK, LONG_CHUNK = 45, 110  # длиннее этого (символов) фраза режется по запятой или тире
# Каждый следующий кусок — не длиннее GROWTH × уже сказанного: он синтезируется, пока звучат прошлые.
# Поток первой фразы съедает ~40% её длительности, поэтому после него запас меньше.
GROWTH = 1.5 if STREAM_FIRST else 2.2


async def sentences(deltas):
    """Режет поток токенов на фразы, чтобы озвучивать первую, не дожидаясь всего ответа.

    Звук фразы уходит, только когда она озвучена целиком, поэтому длинная первая фраза — это
    секунды тишины. Её режем по запятой или тире: первый кусок звучит сразу, а пока он играет,
    синтезатор успевает остальное. Синтез идёт втрое быстрее речи, так что следующий кусок может
    быть примерно вдвое длиннее всего, что уже сказано, — иначе между ними повиснет пауза. Дальше режем только
    очень длинные фразы: на стыках кусков интонация начинается заново.
    """
    buf, limit, said = "", FIRST_CHUNK, 0
    seen = lambda text: len(_ACT.sub("", text).strip())  # метки жестов не звучат — длину считаем без них
    async for delta in deltas:
        buf += delta
        while True:
            # слишком короткий кусок не отдаём: «Да.» отдельным заходом звучит рвано
            m = next((m for m in _SENTENCE_END.finditer(buf) if seen(buf[: m.end()]) >= 15), None)
            if seen(buf[: m.end()] if m else buf) > limit:
                c = next((c for c in _CLAUSE_END.finditer(buf, 0, m.end() if m else len(buf)) if seen(buf[: c.end()]) >= 15), None)
                if c:
                    m = c
            if not m:
                break
            piece = buf[: m.end()].strip()
            yield piece
            buf = buf[m.end() :]
            said += seen(piece)
            limit = min(LONG_CHUNK, int(said * GROWTH))
    if buf.strip():
        yield buf.strip()


DEFAULT_LOOK = "милая аниме-девушка"

IDLE_HINT = (
    "Сейчас в разговоре повисла пауза: собеседник несколько секунд молчит. Не спрашивай, тут ли он, "
    "и не говори про тишину. Сама подхвати беседу: продолжи прошлую тему с новой стороны, поделись "
    "своей мыслью или маленькой историей, или предложи новую тему, связанную с тем, что он рассказывал. "
    "Если разговор только начался — поздоровайся и спроси что-нибудь лёгкое. Одна-две фразы."
)


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.persona = (ROOT / "persona.txt").read_text(encoding="utf-8").strip()
        self.look = DEFAULT_LOOK  # внешность текущей модели аватара, присылает страница
        self.history = []
        self.turn = 0
        self.seg = 0  # номер потоковой фразы: к нему клиент привязывает расписание рта
        self.task = None

    @property
    def system(self):
        return self.persona.replace("{appearance}", self.look)

    async def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def start(self, audio=None, text=None, idle=False, stt=None):
        await self.stop()
        self.turn += 1
        self.task = asyncio.create_task(self.respond(self.turn, audio, text, idle, stt))

    async def hear(self, audio):
        """Сначала слушаем, потом перебиваем: детектор речи срабатывает и на шум, и на голос Мики из колонок —
        если слов нет, текущий ответ не трогаем, а клиент доигрывает его с места паузы."""
        t0 = time.perf_counter()
        text = await on_gpu(transcribe, audio)
        if text:
            await self.start(text=text, stt=round(time.perf_counter() - t0, 3))
        else:
            await self.ws.send_json({"type": "skip"})

    def remember(self, role, text):
        # перебитый до первого слова ответ оставляет две реплики человека подряд — склеиваем
        if self.history and self.history[-1]["role"] == role:
            self.history[-1]["content"] += "\n" + text
        else:
            self.history.append({"role": role, "content": text})
        del self.history[:-HISTORY_LIMIT]

    async def stream(self, turn, words, acts, timing, t0):
        """Первая фраза потоком: каждый кусок — свой «say» с тем же текстом, номером фразы seg и смещением
        от её начала; жесты — с первым куском. Когда фраза готова, вдогонку идёт «mouth» с расписанием рта."""
        loop, q, stop = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()
        self.seg += 1
        seg = self.seg

        def work():
            cues = None
            try:
                cues = stream_qwen(words, lambda pcm: loop.call_soon_threadsafe(q.put_nowait, pcm), stop)
            finally:
                loop.call_soon_threadsafe(q.put_nowait, cues or [])
                loop.call_soon_threadsafe(q.put_nowait, None)

        job = loop.run_in_executor(GPU, work)
        sent, offset = False, 0.0
        try:
            while (item := await q.get()) is not None:
                if isinstance(item, list):  # расписание рта всей фразы
                    if item and sent:
                        await self.ws.send_json({"type": "mouth", "turn": turn, "seg": seg, "mouth": item})
                    continue
                await self.ws.send_json({"type": "say", "turn": turn, "text": words, "mouth": [], "seg": seg,
                                         "offset": round(offset, 4), "acts": [] if sent else acts, "rate": STREAM_RATE})
                await self.ws.send_bytes(item)
                offset += len(item) / 2 / STREAM_RATE
                if not sent:
                    timing.setdefault("first_audio", round(time.perf_counter() - t0, 3))
                sent = True
            await job  # ошибки синтеза — наверх
        finally:
            stop.set()  # перебили — карта бросает фразу на следующем куске
        return sent

    async def respond(self, turn, audio, text, idle=False, stt=None):
        t0 = time.perf_counter() - (stt or 0)
        timing = {"stt": stt} if stt else {}
        try:
            if idle:
                # пауза затянулась — Мика заговаривает сама; в память эта подсказка не идёт
                await self.ws.send_json({"type": "turn", "turn": turn})
                messages = ([{"role": "system", "content": self.system + "\n\n" + IDLE_HINT}] + self.history
                            + [{"role": "user", "content": "(молчит)"}])
                text = "(пауза)"
            else:
                if audio is not None:
                    text = await on_gpu(transcribe, audio)
                    timing["stt"] = round(time.perf_counter() - t0, 3)
                if not text:
                    await self.ws.send_json({"type": "skip", "turn": turn})
                    return
                await self.ws.send_json({"type": "user", "turn": turn, "text": text})
                self.remember("user", text)
                messages = [{"role": "system", "content": self.system}] + self.history

            spoken = []
            acts = []  # метки без слов рядом ждут следующей фразы
            try:
                async for sentence in sentences(llm_stream(messages)):
                    timing.setdefault("llm_first_sentence", round(time.perf_counter() - t0, 3))
                    sentence = feminine(sentence)
                    acts += _ACT.findall(sentence)
                    words = _ACT.sub("", sentence).strip()
                    if STREAM_FIRST and not spoken and "qwen" in models:
                        if await self.stream(turn, words, acts, timing, t0):
                            spoken.append(sentence)
                            acts = []
                        continue
                    pcm, cues = await on_gpu(synth, words)
                    if not pcm:
                        continue
                    await self.ws.send_json({"type": "say", "turn": turn, "text": words, "mouth": cues, "acts": acts})
                    await self.ws.send_bytes(pcm)
                    timing.setdefault("first_audio", round(time.perf_counter() - t0, 3))
                    # в память кладём вместе с метками, чтобы модель видела свой формат
                    spoken.append(sentence)
                    acts = []
                if acts:  # ответ из одних жестов или жест в самом конце
                    await self.ws.send_json({"type": "act", "turn": turn, "acts": acts})
                    spoken.append("".join(f"[{a}]" for a in acts))
            finally:
                # в память идёт только то, что успело уйти в звук: остальное человек не слышал
                if spoken:
                    self.remember("assistant", " ".join(spoken))
            timing["total"] = round(time.perf_counter() - t0, 3)
            print(f"[turn {turn}] {timing} | {text!r}")
            await self.ws.send_json({"type": "done", "turn": turn, "timing": timing})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[turn {turn}] ошибка: {e!r}")
            await self.ws.send_json({"type": "error", "turn": turn, "text": str(e)})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session(ws)
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            if msg.get("bytes") is not None:
                audio = np.frombuffer(msg["bytes"], dtype=np.float32)
                if len(audio) >= STT_RATE * 0.3:
                    session.listening = asyncio.create_task(session.hear(audio))  # ссылку держим, чтобы задачу не собрал GC
                else:
                    await ws.send_json({"type": "skip"})
                continue
            data = json.loads(msg["text"])
            if data["type"] == "text":
                await session.start(text=data["text"].strip())
            elif data["type"] == "look":
                session.look = data.get("text", "").strip()[:600] or DEFAULT_LOOK
            elif data["type"] == "idle":
                await session.start(idle=True)
            elif data["type"] == "interrupt":
                await session.stop()
            elif data["type"] == "reset":
                await session.stop()
                session.history.clear()
    except WebSocketDisconnect:
        pass
    finally:
        await session.stop()


@app.get("/models")
async def avatar_models():
    """Модели аватара для селектора: файл, название из <имя>.json рядом и размер."""
    out = []
    for f in sorted((ROOT / "static" / "models").glob("*.vrm")):
        try:
            cfg = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}
        out.append({"url": f"/static/models/{f.name}", "title": cfg.get("title", f.stem),
                    "mb": round(f.stat().st_size / 2**20)})
    # варианты: <имя>.json с "model" — та же модель в другом наряде
    for f in sorted((ROOT / "static" / "models").glob("*.json")):
        try:
            cfg = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        vrm = f.parent / str(cfg.get("model", ""))
        if cfg.get("model") and vrm.is_file():
            out.append({"url": f"/static/models/{f.name}", "title": cfg.get("title", f.stem),
                        "mb": round(vrm.stat().st_size / 2**20)})
    out.sort(key=lambda m: m["title"])
    return out


@app.get("/")
async def index():
    # страницу правим на ходу, а браузер норовит показать старую из кэша
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", "8765")))
