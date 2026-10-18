"""QLoRA (nf4) de Qwen3-4B-Instruct-2507 para predecir repartos de respuesta (método de Cao et al. 2025). Se ejecuta en .venv_ft.

La salida del modelo es la distribución de probabilidad de su PRIMER token restringida a las letras válidas A, B, C… de cada ejemplo; la pérdida
es KL(reparto real || softmax de esos logits) ponderada por min(n, 1000)/1000 (y ×0,5 en Eurobarómetro y ESS). En inferencia no se genera texto.

    set PYTHONPATH=src
    .venv_ft\\Scripts\\python -m mavericks.ft.train --cut A --name a1 [--max-train 500] [--resume]
"""

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from . import core

BASE = "Qwen/Qwen3-4B-Instruct-2507"
EXAMPLES = Path("outputs/ft/data/examples.pkl")
CUTS = {"A": ("in_a", "2025-07-01"), "B": ("in_b", "2026-04-01")}
LORA = dict(r=8, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
LR, WARMUP, EFFECTIVE_BATCH = 1e-4, 0.03, 16


def load_frame(cut):
    flag, limit = CUTS[cut]
    frame = pd.read_pickle(EXAMPLES)
    frame = frame[frame[flag]].reset_index(drop=True)
    assert (frame.date < limit).all(), "ejemplo con trabajo de campo posterior al corte"
    assert (frame.date < "2026-04-01").all()  # nunca estudios reservados (publicados desde 2026-04-01)
    if cut == "A":  # el corte A' no puede contener nada de dev_grande
        dev = pd.read_pickle("outputs/prior_v2/bigdev_preds.pkl")
        dev_ids, dev_studies = {r["evidence_id"] for r in dev}, {str(r["study"]) for r in dev}
        assert not set(frame.evidence_id) & dev_ids, "dev_grande dentro del entrenamiento"
        assert not set(frame.study.astype(str)) & dev_studies, "estudio de dev_grande dentro del entrenamiento"
    return frame


def split_validation(frame, share=0.02, seed=0):
    """2 % de los ejemplos como validación, separado POR ESTUDIO (estudios enteros)."""
    studies = sorted(frame.study.unique())
    np.random.RandomState(seed).shuffle(studies)
    chosen, count = [], 0
    for s in studies:
        size = int((frame.study == s).sum())
        if count >= share * len(frame):
            break
        if chosen and count + size > 1.5 * share * len(frame):
            continue  # no pasarse mucho del 2 %: se prueba con otro estudio más pequeño
        chosen.append(s)
        count += size
    val = frame.study.isin(chosen)
    return frame[~val].reset_index(drop=True), frame[val].reset_index(drop=True)


def fit_budget(train, max_examples, seed=0):
    """Submuestrea SUBGRUPOS (nunca los nacionales), estratificado por fuente, dimensión y nº de opciones, hasta ``max_examples``."""
    if len(train) <= max_examples:
        return train
    national = train[train.scope == "nacional"]
    groups = train[train.scope != "nacional"]
    fraction = max(0.0, (max_examples - len(national)) / len(groups))
    key = groups.source + "|" + groups.dim.astype(str) + "|" + groups.options.map(len).astype(str)
    parts = [g.sample(frac=fraction, random_state=seed) for _, g in groups.groupby(key)]
    return pd.concat([national, *parts]).sort_values("id").reset_index(drop=True)


def load_model(adapter=None, trainable=True):
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16)
    if adapter is not None:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=trainable)
    elif trainable:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = get_peft_model(model, LoraConfig(**LORA))
    model.config.use_cache = False
    return tok, model


def letter_ids(tok):
    ids = []
    for c in core.LETTERS:
        t = tok.encode(c, add_special_tokens=False)
        assert len(t) == 1, f"la letra {c} no es un solo token"
        ids.append(t[0])
    return ids


def make_batch(rows, epoch, tok, seed=0, shuffle=True):
    prompts, props, weights = [], [], []
    for _, r in rows.iterrows():
        prompt, p = core.build_item(r, epoch, seed, shuffle)
        prompts.append(tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True))
        props.append(p)
        weights.append(min(r["n"], 1000) / 1000 * r["weight_source"])
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    return enc, props, torch.tensor(weights, dtype=torch.float32)


def letter_log_probs(model, enc, ids, n_options):
    """log q (sobre las letras válidas de cada ejemplo) a partir de los logits del primer token de la respuesta."""
    device = next(model.parameters()).device
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=enc["input_ids"].to(device), attention_mask=enc["attention_mask"].to(device), logits_to_keep=1)
    logits = out.logits[:, -1, :][:, ids].float()
    valid = torch.arange(len(ids), device=device)[None, :] < torch.tensor(n_options, device=device)[:, None]
    return torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1), valid


@torch.no_grad()
def validation_loss(model, tok, ids, val, limit=1000, micro=8):
    model.eval()
    rows = val.sample(min(limit, len(val)), random_state=0) if len(val) else val
    total = count = 0.0
    for start in range(0, len(rows), micro):
        enc, props, weights = make_batch(rows.iloc[start : start + micro], 0, tok, shuffle=False)
        log_q, valid = letter_log_probs(model, enc, ids, [len(p) for p in props])
        loss, _ = core.kl_loss(log_q, valid, props, weights)
        total += float(loss) * len(props)
        count += len(props)
    model.train()
    return total / max(count, 1)


def gpu_temperature():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "?"


def log(path, text):
    line = f"{time.strftime('%H:%M:%S')} {text}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut", choices=list(CUTS), default="A")
    parser.add_argument("--name", default="run")
    parser.add_argument("--max-train", type=int)
    parser.add_argument("--micro", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-hours", type=float, default=5.8, help="Si una época no cabe, submuestrea subgrupos hasta que quepa")
    parser.add_argument("--seconds-per-step", type=float, default=6.3, help="Medido en la prueba corta (micro-lote 8, 4060 portátil)")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    run = Path("outputs/ft/runs") / args.name
    run.mkdir(parents=True, exist_ok=True)
    status = Path("outputs/ft/status.log")
    frame = load_frame(args.cut)
    train, val = split_validation(frame, seed=args.seed)
    if args.max_train:
        train = train.sample(min(args.max_train, len(train)), random_state=args.seed).reset_index(drop=True)
    budget = int(args.target_hours * 3600 / args.seconds_per_step * EFFECTIVE_BATCH)
    before = len(train)
    train = fit_budget(train, budget, args.seed)
    total_steps = math.ceil(len(train) / EFFECTIVE_BATCH)
    ckpt_every = max(1, round(0.2 * total_steps))
    log(status, f"INICIO {args.name} corte {args.cut}: {len(train)} ejemplos de entrenamiento (de {before} disponibles; presupuesto {budget}), {len(val)} de validación (por estudio), {total_steps} pasos, micro-lote {args.micro}")
    assert args.cut != "A" or train.date.max() < "2025-07-01"

    state_file = run / "state.json"
    start_step = 0
    adapter = None
    if args.resume and state_file.exists():
        start_step = json.loads(state_file.read_text())["step"]
        adapter = run / "adapter"
    tok, model = load_model(adapter)
    ids = letter_ids(tok)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    warm = max(1, round(WARMUP * total_steps))
    schedule = lambda s: (s + 1) / warm if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm))  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if start_step and (run / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(run / "optimizer.pt"))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule, last_epoch=start_step - 1)
    model.train()
    order = np.random.RandomState(args.seed).permutation(len(train))
    accumulate = EFFECTIVE_BATCH // args.micro
    torch.cuda.reset_peak_memory_stats()
    started, window, window_n = time.time(), 0.0, 0
    for step in range(start_step, total_steps):
        batch = train.iloc[order[step * EFFECTIVE_BATCH : (step + 1) * EFFECTIVE_BATCH]]
        step_loss = 0.0
        for start in range(0, len(batch), args.micro):
            enc, props, weights = make_batch(batch.iloc[start : start + args.micro], 0, tok, seed=args.seed + step)
            log_q, valid = letter_log_probs(model, enc, ids, [len(p) for p in props])
            loss, _ = core.kl_loss(log_q, valid, props, weights)
            (loss / accumulate).backward()
            step_loss += float(loss) / accumulate
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        window, window_n = window + step_loss, window_n + 1
        done = step + 1
        if done % 10 == 0 or done == total_steps:
            per_step = (time.time() - started) / (done - start_step)
            log(status, f"{args.name} paso {done}/{total_steps} pérdida {window / window_n:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                        f"VRAM máx {torch.cuda.max_memory_allocated() / 2**30:.2f} GB temp {gpu_temperature()}°C {per_step:.2f} s/paso "
                        f"(época completa ≈ {per_step * total_steps / 3600:.1f} h)")
            window, window_n = 0.0, 0
        if not args.no_checkpoints and (done % ckpt_every == 0 or done == total_steps):
            model.save_pretrained(run / "adapter")
            torch.save(optimizer.state_dict(), run / "optimizer.pt")
            state_file.write_text(json.dumps({"step": done, "total": total_steps}))
            log(status, f"{args.name} guardado en el paso {done}; pérdida de validación {validation_loss(model, tok, ids, val, micro=args.micro):.4f}")
    if args.no_checkpoints:
        log(status, f"{args.name} prueba corta terminada; pérdida de validación {validation_loss(model, tok, ids, val, limit=200, micro=args.micro):.4f}")
    log(status, f"FIN {args.name}: {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
