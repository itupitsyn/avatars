"""Дообучение Qwen3-TTS 1.7B Base на голос Мики (LoRA на внимание и MLP talker'а и предсказателя кодов).
Последовательность собирается ровно как в Qwen3TTSForConditionalGeneration.generate для режима
«только отпечаток голоса» (x_vector_only, потоковая подача текста), чтобы сервер потом синтезировал так же.
На выходе models/mika-tts: копия базовой модели со слитыми в веса поправками LoRA.
Запуск: .venv/Scripts/python.exe dataset/train.py [эпох]"""
import json, math, os, random, shutil, sys, time
import torch
os.environ["PATH"] = os.path.join(os.path.dirname(torch.__file__), "lib") + os.pathsep + os.environ["PATH"]
import torch.nn as nn, torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from qwen_tts import Qwen3TTSModel

ROOT = os.environ.get("DATA", os.path.dirname(os.path.abspath(__file__)))  # папка с train_raw.jsonl
BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", os.environ.get("OUT_NAME", "mika-tts"))
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
RANK, ALPHA, LR, ACCUM, SUB_WEIGHT = 16, 32, 2e-4, 8, 0.3
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRA(nn.Module):
    def __init__(self, base: nn.Linear):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(RANK, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, RANK, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (F.linear(F.linear(x.float(), self.a), self.b) * (ALPHA / RANK)).to(x.dtype)

    def merged(self):
        return (self.base.weight.float() + self.b @ self.a * (ALPHA / RANK)).to(self.base.weight.dtype)


tts = Qwen3TTSModel.from_pretrained(BASE, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
model, cfg = tts.model, tts.model.config
talker, tc = model.talker, cfg.talker_config
model.requires_grad_(False)
model.eval()

loras = {}
for prefix, layers in (("talker.model.layers", talker.model.layers), ("talker.code_predictor.model.layers", talker.code_predictor.model.layers)):
    for i, layer in enumerate(layers):
        for owner_name, owner in list(layer.named_modules()):
            for t in TARGETS:
                lin = getattr(owner, t, None)
                if isinstance(lin, nn.Linear):
                    lora = LoRA(lin)
                    setattr(owner, t, lora)
                    loras[f"{prefix}.{i}.{owner_name}.{t}.weight".replace("..", ".")] = lora
params = [p for l in loras.values() for p in (l.a, l.b)]
print(f"LoRA: {len(loras)} слоёв, {sum(p.numel() for p in params) / 1e6:.1f} млн параметров", flush=True)

prep = torch.load(os.path.join(ROOT, "prep.pt"))
dev = talker.device
spk = prep["spk"].to(dev, torch.bfloat16)
data = prep["data"]
random.Random(0).shuffle(data)
val, train = data[:24], data[24:]

tp = lambda ids: talker.text_projection(talker.get_text_embeddings()(ids))
cemb = talker.get_input_embeddings()
lang = tc.codec_language_id["russian"]
long_ = lambda xs: torch.tensor([xs], device=dev)


def sequence(ids, codes):
    """Вход talker'а, метки первого слоя кодов и позиции, чьи скрытые состояния кормят предсказатель остальных слоёв."""
    ids, codes = ids.to(dev)[None], codes.to(dev)
    bos, eos, pad = tp(long_([cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id])).chunk(3, dim=1)
    codec = torch.cat([cemb(long_([tc.codec_think_id, tc.codec_think_bos_id, lang, tc.codec_think_eos_id])),
                       spk.view(1, 1, -1), cemb(long_([tc.codec_pad_id, tc.codec_bos_id]))], dim=1)
    prefix = torch.cat([tp(ids[:, :3]),
                        torch.cat([pad.expand(-1, codec.shape[1] - 2, -1), bos], dim=1) + codec[:, :-1],
                        tp(ids[:, 3:4]) + codec[:, -1:]], dim=1)
    text = ids[:, 3:-5]
    n, L = text.shape[1], codes.shape[0]
    trailing = torch.cat([tp(text[:, 1:]), eos], dim=1)
    trailing = torch.cat([trailing, pad.expand(-1, max(0, L - n), -1)], dim=1)[:, :L]
    frames = cemb(codes[None, :, 0]) + sum(talker.code_predictor.get_input_embeddings()[i](codes[None, :, i + 1])
                                           for i in range(tc.num_code_groups - 1))
    x = torch.cat([prefix, frames + trailing], dim=1)
    p0 = prefix.shape[1] - 1
    labels = torch.full((x.shape[1],), -100, device=dev)
    labels[p0: p0 + L] = codes[:, 0]
    labels[p0 + L] = tc.codec_eos_token_id
    return x, labels, p0, L


def loss_of(item):
    x, labels, p0, L = sequence(item["ids"], item["codes"])
    out = talker(inputs_embeds=x, attention_mask=torch.ones(x.shape[:2], device=dev, dtype=torch.long),
                 output_hidden_states=True, use_cache=False)
    hidden = out.hidden_states[0][-1][0]
    main = F.cross_entropy(out.logits[0].float(), labels, ignore_index=-100)
    codes = item["codes"].to(dev)
    sub_logits, _ = talker.forward_sub_talker_finetune(codes, hidden[p0: p0 + L])
    sub = F.cross_entropy(sub_logits.float().reshape(-1, sub_logits.shape[-1]), codes[:, 1:].reshape(-1))
    return main, sub


@torch.no_grad()
def validate():
    m = s = 0.0
    for it in val:
        a, b = loss_of(it)
        m, s = m + a.item(), s + b.item()
    return m / len(val), s / len(val)


opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
steps = EPOCHS * len(train) // ACCUM
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 20) * max(0.05, 1 - s / steps))
print("до обучения: val talker %.3f, коды %.3f" % validate(), flush=True)
t0, step = time.time(), 0
for epoch in range(EPOCHS):
    random.Random(epoch + 1).shuffle(train)
    run_m = run_s = 0.0
    for i, it in enumerate(train):
        main, sub = loss_of(it)
        ((main + SUB_WEIGHT * sub) / ACCUM).backward()
        run_m, run_s = run_m + main.item(), run_s + sub.item()
        if (i + 1) % ACCUM == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 20 == 0:
                print(f"эпоха {epoch} шаг {step}/{steps}: talker {run_m / (20 * ACCUM):.3f}, коды {run_s / (20 * ACCUM):.3f}"
                      f" | {(time.time() - t0) / 60:.1f} мин, {torch.cuda.max_memory_allocated() / 1e9:.1f} ГБ", flush=True)
                run_m = run_s = 0.0
    print(f"после эпохи {epoch}: val talker %.3f, коды %.3f" % validate(), flush=True)

# сохраняем: копия базовой модели, в которой веса со LoRA заменены слитыми
src = snapshot_download(BASE)
if os.path.exists(OUT):
    shutil.rmtree(OUT)
shutil.copytree(src, OUT)
weights = load_file(os.path.join(src, "model.safetensors"))
missing = [k for k in loras if k not in weights]
assert not missing, missing[:5]
with torch.no_grad():
    for k, lora in loras.items():
        weights[k] = lora.merged().to(weights[k].dtype).cpu().contiguous()
os.remove(os.path.join(OUT, "model.safetensors"))
save_file(weights, os.path.join(OUT, "model.safetensors"), metadata={"format": "pt"})
# тексты с метками ударений (accent.py) — серверу ставить их и перед синтезом
raw = os.path.join(ROOT, "train_raw.jsonl")
stress = os.path.exists(raw) and "+" in open(raw, encoding="utf-8").read(20000)
json.dump(dict(base=BASE, epochs=EPOCHS, rank=RANK, alpha=ALPHA, lr=LR, items=len(train), stress=stress),
          open(os.path.join(OUT, "mika_finetune.json"), "w"))
print("сохранено в", os.path.abspath(OUT))
