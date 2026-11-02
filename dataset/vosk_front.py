"""Разбор текста для Vosk-TTS — один и тот же при подготовке датасета и при синтезе.
Vosk сам переводит слова в фонемы: слово из словаря произношений берёт оттуда (там учтены «что»,
«его», «сегодня»), незнакомое читает по правилам и ставит ударение по метке «+». Поэтому слова,
которые словарь знает однозначно, оставляем как есть, а омографы («замок», «все») и незнакомые
отдаём с ударением от RUAccent — он выбирает его по смыслу фразы."""
import re
from num2words import num2words

SPLIT = re.compile(r"([,.?!;:\"() ])")  # как делит текст сам vosk
_YO = str.maketrans("ё", "е")
_VOWELS = "аеёиоуыэюя"


def normalize(text):
    """Только то, что vosk умеет читать: числа — словами, тире и многоточие — в паузы, кавычки и прочее — прочь."""
    text = re.sub(r"\d+", lambda m: " " + num2words(int(m.group()), lang="ru") + " ", text)
    text = re.sub(r"\s*[—–]\s*", ", ", text)
    text = text.replace("…", ".").replace("«", "").replace("»", "").replace("“", "").replace("”", "").replace("„", "").replace('"', "")
    text = re.sub(r"[^а-яёА-ЯЁ0-9,.?!;:() +-]", " ", text)  # латиницу vosk не читает
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([,.?!;:])", r"\1", text)


def load_dictionary(path):
    """Слова, ударение в которых словарь знает однозначно, -> номер ударной гласной. Вариантов
    произношения бывает несколько («конечно» / «конешно») — это не страшно, пока ударная гласная
    у всех одна и та же."""
    stress = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            word, _, phones = line.rstrip("\n").split(" ", 2)
            vowels = [p for p in phones.split() if p[-1] in "01"]
            at = next((i for i, p in enumerate(vowels) if p.endswith("1")), -1)
            stress.setdefault(word, set()).add(at)
    return {w: next(iter(s)) for w, s in stress.items() if len(s) == 1}


class Front:
    def __init__(self, accent, dictionary_path):
        self.accent = accent
        self.certain = load_dictionary(dictionary_path)

    def __call__(self, text):
        text = normalize(text)
        plain = SPLIT.split(text)
        marked = SPLIT.split(self.accent.process_all(text))
        same = len(plain) == len(marked) and all(
            p.lower().translate(_YO) == m.replace("+", "").lower().translate(_YO) for p, m in zip(plain, marked))
        out = []
        for i, word in enumerate(plain):
            w = word.lower()
            yo = same and "ё" in marked[i].lower() and "ё" not in w  # «все» -> «всё»: это решает только RUAccent
            # словарь местами ошибается («+асцендент», «переж+иваю»): если RUAccent ставит ударение иначе — верим ему
            agree = same and "+" in marked[i] and self.certain.get(w) == sum(c in _VOWELS for c in marked[i][:marked[i].index("+")].lower())
            if not w or SPLIT.fullmatch(w) or not same or "+" not in marked[i] or (w in self.certain and agree and not yo):
                out.append(word)
            else:
                out.append(marked[i])
        return "".join(out)
