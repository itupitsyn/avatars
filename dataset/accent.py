"""Ударения в текстах датасета: RUAccent ставит «+» перед ударной гласной и возвращает «ё»
(«з+амок» / «зам+ок» — по смыслу фразы). Модель учится говорить по меткам, сервер ставит
их тем же RUAccent перед синтезом.
python accent.py <вход train_raw.jsonl> <выход train_raw.jsonl>  — или без аргументов: проба на фразах."""
import json, os, sys, time
import numpy as np
from ruaccent import RUAccent

WORKDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "ruaccent")


def load():
    os.makedirs(WORKDIR, exist_ok=True)
    acc = RUAccent()
    acc.load(omograph_model_size="turbo3.1", use_dictionary=True, workdir=WORKDIR)
    # новые transformers не отдают token_type_ids, а модели RUAccent их ждут — подставляем нули
    for m in vars(acc).values():
        sess = getattr(m, "session", None)
        if sess and "token_type_ids" in {i.name for i in sess.get_inputs()}:
            sess.run = (lambda run: lambda out, feed: run(out, feed if "token_type_ids" in feed else
                        {**feed, "token_type_ids": np.zeros_like(feed["input_ids"])}))(sess.run)
    return acc


if __name__ == "__main__":
    acc = load()
    if len(sys.argv) < 3:
        for s in ["На двери висел старый замок, а за рекой стоял замок.", "Мука для пирога и муки творчества.",
                  "Все пришли, и всё было хорошо.", "Я позвоню тебе завтра в три часа, договорились?",
                  "Какие у тебя планы на выходные?", "Звёзды сегодня особенно яркие!"]:
            t = time.time()
            print(acc.process_all(s), f"({(time.time() - t) * 1000:.0f} мс)")
        sys.exit()
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    t = time.time()
    with open(sys.argv[2], "w", encoding="utf-8") as out:
        for i, r in enumerate(rows):
            r["plain"] = r["text"]
            r["text"] = acc.process_all(r["text"])
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i % 500 == 0:
                print(i, r["text"][:80], flush=True)
    print("готово:", len(rows), f"за {time.time() - t:.0f} с")
