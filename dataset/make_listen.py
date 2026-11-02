"""Страничка прослушивания датасета: случайные принятые записи разной длины и все отбракованные с причинами."""
import json, os, random, html
import soundfile as sf
ROOT = os.path.dirname(os.path.abspath(__file__))
ok = [json.loads(l) for l in open(os.path.join(ROOT, "train_raw.jsonl"), encoding="utf-8")]
bad = [json.loads(l) for l in open(os.path.join(ROOT, "rejected.jsonl"), encoding="utf-8")] if os.path.exists(os.path.join(ROOT, "rejected.jsonl")) else []
for r in ok:
    r["dur"] = sf.info(r["audio"]).duration
total = sum(r["dur"] for r in ok)
rng = random.Random(7)
short = [r for r in ok if r["dur"] < 2.5]; mid = [r for r in ok if 2.5 <= r["dur"] < 8]; long = [r for r in ok if r["dur"] >= 8]
pick = lambda xs, n: sorted(rng.sample(xs, min(n, len(xs))), key=lambda r: r["id"])
row = lambda r: (f'<div class="row"><audio controls preload="none" src="wavs/{r["id"]}.wav"></audio>'
                 f'<div><b>{r["id"]}</b> · {r["dur"]:.1f} с · тембр {r["sim"]} · {r["hz"]} Гц<br>{html.escape(r["text"])}</div></div>')
sec = lambda title, rows: f"<section><h2>{title}</h2>{''.join(rows)}</section>"
rej = [f'<div class="row bad"><div><b>{b["id"]}</b> {html.escape(b["text"])}<br><small>не расслышал: '
       f'{html.escape(", ".join(sorted({w for t in b["takes"] for w in t["missed"]})) or "—")}; '
       f'Whisper: «{html.escape(b["takes"][0]["heard"])}»</small></div></div>' for b in bad]
page = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Датасет Мики</title><style>
:root{{--bg:#1b1730;--card:#262040;--text:#ece8f5;--muted:#a49cc0}}
body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:820px;margin:0 auto;padding:24px 16px 48px}} p{{color:var(--muted)}}
section{{background:var(--card);border-radius:12px;padding:14px 16px;margin-bottom:12px}} h2{{font-size:16px;margin:0 0 8px}}
.row{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:10px 0}} .row div{{flex:1;min-width:240px}}
audio{{width:300px;max-width:100%;height:36px}} small,b{{color:var(--muted)}}
</style></head><body><main><h1>Датасет голоса Мики</h1>
<p>Принято {len(ok)} записей, {total / 60:.1f} мин речи; отсеяно {len(bad)}. Ниже случайная выборка — проверь, что голос везде один и тот же и звучит живо.</p>
{sec("Короткие реакции", map(row, pick(short, 8)))}{sec("Обычные реплики", map(row, pick(mid, 10)))}{sec("Длинные фразы", map(row, pick(long, 8)))}
{sec("Отсеянные", rej)}</main></body></html>"""
open(os.path.join(ROOT, "listen.html"), "w", encoding="utf-8").write(page)
print(len(ok), round(total / 60, 1), len(short), len(mid), len(long))
