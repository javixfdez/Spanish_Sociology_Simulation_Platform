"""Entrenamiento del modelo B en Kaggle (2×T4 con reparto manual del lote, o 1×T4). Misma configuración que a1 salvo el cómputo: fp16 (la T4 no tiene bf16).

    torchrun --nproc_per_node=2 train_kaggle.py --cut B --name b1 --data /kaggle/input/<dataset>

Reutiliza de train.py lo que no depende del dtype (validación por estudio, submuestreo de subgrupos, letras, lotes). El lote efectivo es 16 como en a1:
con 2 GPUs cada una procesa 8 y se promedian los gradientes; con 1 GPU, dos micro-lotes de 8 acumulados.
"""

import argparse
import json
import math
import os
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import train  # el train.py del dataset (junto a core.py)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from train import BASE, EFFECTIVE_BATCH, LORA, LR, WARMUP, core
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"  # misma revisión que a1 (Qwen/Qwen3-4B-Instruct-2507)
DTYPE = torch.float16  # se puede cambiar a float32 con --compute fp32


def log(path, text):
    line = f"{time.strftime('%H:%M:%S')} {text}"
    print(line, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_model(adapter, rank):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE, revision=REVISION)
    tok.padding_side = "left"
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(BASE, revision=REVISION, quantization_config=quant, device_map={"": rank}, dtype=DTYPE)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
    model = PeftModel.from_pretrained(model, adapter, is_trainable=True) if adapter else get_peft_model(model, LoraConfig(**LORA))
    model.config.use_cache = False
    return tok, model


def letter_log_probs(model, enc, ids, n_options):
    device = next(model.parameters()).device
    with torch.autocast("cuda", dtype=DTYPE):
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
        enc, props, weights = train.make_batch(rows.iloc[start : start + micro], 0, tok, shuffle=False)
        log_q, valid = letter_log_probs(model, enc, ids, [len(p) for p in props])
        loss, _ = core.kl_loss(log_q, valid, props, weights)
        total += float(loss) * len(props)
        count += len(props)
    model.train()
    return total / max(count, 1)


def micro_step(model, tok, ids, rows, step, seed, accumulate, scaler):
    enc, props, weights = train.make_batch(rows, 0, tok, seed=seed + step)
    log_q, valid = letter_log_probs(model, enc, ids, [len(p) for p in props])
    loss, _ = core.kl_loss(log_q, valid, props, weights)
    scaler.scale(loss / accumulate).backward()
    return float(loss) / accumulate


def sync_grads(params, world):
    if world == 1:
        return
    flat = torch.cat([p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten() for p in params])
    dist.all_reduce(flat)
    flat /= world
    offset = 0
    for p in params:
        n = p.numel()
        p.grad = flat[offset : offset + n].view_as(p).clone()
        offset += n


def save_state(model, optimizer, scaler, run, step, total):
    model.save_pretrained(run / "adapter")
    torch.save({"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, run / "optimizer.pt")
    (run / "state.json").write_text(json.dumps({"step": step, "total": total}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut", default="B")
    parser.add_argument("--name", default="b1")
    parser.add_argument("--data", required=True, help="carpeta del dataset de Kaggle (examples_b.pkl)")
    parser.add_argument("--out", default="/kaggle/working")
    parser.add_argument("--micro", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-hours", type=float, default=9.5, help="horas de entrenamiento; el resto del límite de 11 h queda para carga, validaciones y guardados")
    parser.add_argument("--calibration-steps", type=int, default=6)
    parser.add_argument("--max-train", type=int, help="solo para pruebas cortas")
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--compute", choices=["fp16", "fp32"], default="fp16")
    args = parser.parse_args()
    global DTYPE
    DTYPE = torch.float32 if args.compute == "fp32" else torch.float16

    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(rank)
    if world > 1:
        dist.init_process_group("nccl", timeout=timedelta(minutes=90))
    torch.manual_seed(args.seed)
    run = Path(args.out) / "runs" / args.name
    run.mkdir(parents=True, exist_ok=True)
    status = Path(args.out) / "status.log"
    say = (lambda t: log(status, t)) if rank == 0 else (lambda t: None)

    assert args.cut == "B"
    frame = pd.read_parquet(Path(args.data) / "examples_b.parquet")  # parquet y no pickle: independiente de la versión de pandas de Kaggle
    frame["options"] = frame.options.map(list)
    frame["props"] = frame.props.map(lambda p: [float(x) for x in p])
    frame = frame[frame.in_b].reset_index(drop=True)
    assert (frame.date < "2026-04-01").all(), "ejemplo posterior al corte B"
    full_train, val = train.split_validation(frame, seed=args.seed)
    if args.max_train:
        full_train = full_train.sample(min(args.max_train, len(full_train)), random_state=args.seed).reset_index(drop=True)
    per_rank = EFFECTIVE_BATCH // world
    accumulate = per_rank // args.micro
    assert accumulate >= 1 and per_rank % args.micro == 0

    state_file, budget_file = run / "state.json", run / "budget.json"
    start_step, adapter = 0, None
    if state_file.exists():
        start_step, adapter = json.loads(state_file.read_text())["step"], run / "adapter"
    tok, model = load_model(adapter, rank)
    ids = train.letter_ids(tok)
    say(f"GPUs: {world} × {torch.cuda.get_device_name(rank)}; cómputo {args.compute}; {len(full_train)} ejemplos de entrenamiento disponibles, {len(val)} de validación")

    # Comprobación de estabilidad numérica con el modelo base, en 64 ejemplos reales.
    # OJO: las letras de opciones que no existen se enmascaran a -inf a propósito; solo se comprueban las VÁLIDAS.
    bad = 0
    with torch.no_grad():
        for s0 in range(0, 64, args.micro):
            enc, props, weights = train.make_batch(full_train.iloc[s0 : s0 + args.micro], 0, tok, seed=args.seed)
            log_q, valid = letter_log_probs(model, enc, ids, [len(p) for p in props])
            bad += int((~torch.isfinite(log_q[valid])).sum())
            loss, _ = core.kl_loss(log_q, valid, props, weights)
            bad += int(not torch.isfinite(loss))
    say(f"COMPROBACIÓN NUMÉRICA ({args.compute}): {bad} valores no finitos en 64 ejemplos")
    assert bad == 0, f"logits no finitos en {args.compute}: el entrenamiento no es seguro (prueba --compute fp32)"

    # Presupuesto de tiempo: se mide la velocidad real y se decide cuántos ejemplos caben (solo se recortan SUBGRUPOS); se guarda para poder reanudar
    scaler = torch.amp.GradScaler("cuda")
    params = [p for p in model.parameters() if p.requires_grad]
    if budget_file.exists():
        cal = json.loads(budget_file.read_text())
    else:
        model.train()
        times = []
        for i in range(args.calibration_steps + 2):
            batch = full_train.iloc[i * EFFECTIVE_BATCH + rank * per_rank : i * EFFECTIVE_BATCH + (rank + 1) * per_rank]
            t = time.time()
            for s in range(0, per_rank, args.micro):
                micro_step(model, tok, ids, batch.iloc[s : s + args.micro], i, args.seed, accumulate, scaler)
            sync_grads(params, world)
            torch.cuda.synchronize()
            times.append(time.time() - t)
            for p in params:
                p.grad = None
        seconds = float(np.mean(times[2:]))
        if world > 1:
            t = torch.tensor([seconds], device=f"cuda:{rank}")
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            seconds = float(t)
        cal = {"seconds_per_step": seconds, "max_examples": int(args.target_hours * 3600 / seconds * EFFECTIVE_BATCH)}
        if rank == 0:
            budget_file.write_text(json.dumps(cal))
        say(f"CALIBRACIÓN: {seconds:.2f} s/paso → caben {cal['max_examples']} ejemplos en {args.target_hours} h (disponibles {len(full_train)})")
    train_set = train.fit_budget(full_train, cal["max_examples"], args.seed)
    total_steps = math.ceil(len(train_set) / EFFECTIVE_BATCH)
    ckpt_every = max(1, round(0.2 * total_steps))
    sub = "COMPLETO (sin submuestreo)" if len(train_set) == len(full_train) else f"SUBMUESTREO solo de subgrupos: {len(train_set)} de {len(full_train)}"
    say(f"INICIO {args.name} corte {args.cut}: {sub}; {total_steps} pasos, guardado cada {ckpt_every}; empieza en el paso {start_step}")

    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    warm = max(1, round(WARMUP * total_steps))
    schedule = lambda s: (s + 1) / warm if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm))  # noqa: E731
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    if start_step and (run / "optimizer.pt").exists():
        blob = torch.load(run / "optimizer.pt")
        optimizer.load_state_dict(blob["optimizer"])
        scaler.load_state_dict(blob["scaler"])
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule, last_epoch=start_step - 1)
    model.train()
    order = np.random.RandomState(args.seed).permutation(len(train_set))
    started, window, window_n, history, skipped = time.time(), 0.0, 0, [], 0
    for step in range(start_step, total_steps):
        batch = train_set.iloc[order[step * EFFECTIVE_BATCH : (step + 1) * EFFECTIVE_BATCH]]
        mine = batch.iloc[rank * (len(batch) // world) : (rank + 1) * (len(batch) // world)] if len(batch) == EFFECTIVE_BATCH else batch
        step_loss = 0.0
        for s in range(0, len(mine), args.micro):
            step_loss += micro_step(model, tok, ids, mine.iloc[s : s + args.micro], step, args.seed, max(1, len(mine) // args.micro), scaler)
        sync_grads(params, world)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        skipped += int(scaler.get_scale() < before)
        ran = step + 1 - start_step
        if ran >= 100 and skipped > 0.2 * ran:
            say(f"ABORTADO: {skipped} de {ran} pasos omitidos por desbordamiento en {args.compute}; relanzar con --compute fp32")
            raise SystemExit(3)
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if world > 1:
            t = torch.tensor([step_loss], device=f"cuda:{rank}")
            dist.all_reduce(t)
            step_loss = float(t) / world
        window, window_n = window + step_loss, window_n + 1
        history.append(step_loss)
        done = step + 1
        if done % 10 == 0 or done == total_steps:
            per_step = (time.time() - started) / (done - start_step)
            say(f"{args.name} paso {done}/{total_steps} pérdida {window / window_n:.4f} lr {scheduler.get_last_lr()[0]:.2e} VRAM máx {torch.cuda.max_memory_allocated() / 2**30:.2f} GB "
                f"{per_step:.2f} s/paso (quedan ≈ {per_step * (total_steps - done) / 3600:.1f} h; pasos omitidos por fp16: {skipped})")
            window, window_n = 0.0, 0
        if not args.no_checkpoints and (done % ckpt_every == 0 or done == total_steps):
            if rank == 0:
                save_state(model, optimizer, scaler, run, done, total_steps)
                say(f"{args.name} guardado en el paso {done}; pérdida de validación {validation_loss(model, tok, ids, val, micro=args.micro):.4f}")
            if world > 1:
                dist.barrier()
    if rank == 0:
        final_val = validation_loss(model, tok, ids, val, limit=len(val) or 1, micro=args.micro)
        model.save_pretrained(run / "adapter_final")
        say(f"FIN {args.name} {time.strftime('%Y-%m-%d %H:%M')} pérdida final de entrenamiento {float(np.mean(history[-100:])) if history else float('nan'):.4f} "
            f"(media de los últimos 100 pasos) pérdida de validación {final_val:.4f}; adaptador final en {run / 'adapter_final'}; {(time.time() - started) / 3600:.2f} h en esta ejecución")
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
