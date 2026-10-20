"""Fase 3, paso 3: métricas internas de a1 en su validación (estudios que a1 no vio). Se ejecuta en .venv_ft; usa la GPU.

    set PYTHONPATH=src
    .venv_ft\\Scripts\\python -m mavericks.ft.internal_val --adapter outputs/ft/runs/a1/adapter_final

Compara a1, el modelo base sin adaptador (mismo método de lectura del primer token) y el reparto uniforme, con el orden original de las opciones (sin barajar).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from . import train

OUT = Path("outputs/ft/phase3")


@torch.no_grad()
def read_probs(model, tok, ids, frame, micro=8):
    model.eval()
    probs = []
    for start in range(0, len(frame), micro):
        enc, props, _ = train.make_batch(frame.iloc[start : start + micro], 0, tok, shuffle=False)
        log_q, _ = train.letter_log_probs(model, enc, ids, [len(p) for p in props])
        probs += [log_q[i].exp()[: len(p)].float().cpu().numpy() for i, p in enumerate(props)]
    return probs


def metrics(frame, probs):
    """Una fila por ejemplo con KL, MAE (puntos) y acierto top-1."""
    rows = []
    for (_, r), q in zip(frame.iterrows(), probs, strict=True):
        p = np.asarray(r["props"], dtype=float)
        q = np.clip(np.asarray(q, dtype=float), 1e-12, None)
        q = q / q.sum()
        kl = float((p * (np.log(np.clip(p, 1e-12, None)) - np.log(q))).sum())
        rows.append({"kl": kl, "mae": 100 * float(np.abs(p - q).mean()), "top1": float(np.argmax(q) == np.argmax(p)),
                     "w": min(r["n"], 1000) / 1000 * r["weight_source"]})
    return pd.DataFrame(rows, index=frame.index)


def summarize(frame, per, system):
    out = []
    n_opt = frame.options.map(len)
    subsets = {
        "global": np.ones(len(frame), bool),
        "nacional": (frame.scope == "nacional").to_numpy(),
    }
    for name, mask in subsets.items():
        for label, sel in (("todas", np.ones(len(frame), bool)), ("2 opciones", (n_opt == 2).to_numpy()), ("3+ opciones", (n_opt >= 3).to_numpy())):
            m = mask & sel
            if not m.any():
                continue
            d = per[m]
            out.append({"sistema": system, "ámbito": name, "opciones": label, "ejemplos": int(m.sum()), "KL sin ponderar": d.kl.mean(),
                        "KL ponderada": float((d.kl * d.w).sum() / d.w.sum()), "MAE (pp)": d.mae.mean(), "top-1": d.top1.mean()})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/ft/runs/a1/adapter_final")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    OUT.mkdir(parents=True, exist_ok=True)
    frame = train.load_frame("A")
    _, val = train.split_validation(frame, seed=args.seed)
    assert (val.date < "2025-07-01").all()
    print(f"validación: {len(val)} ejemplos, {val.study.nunique()} estudios, {(val.scope == 'nacional').sum()} nacionales")
    tok, model = train.load_model(adapter=Path(args.adapter), trainable=False)
    ids = train.letter_ids(tok)
    a1 = read_probs(model, tok, ids, val)
    with model.disable_adapter():
        base = read_probs(model, tok, ids, val)
    uniform = [np.full(len(o), 1 / len(o)) for o in val.options]
    rows = []
    for system, probs in (("a1", a1), ("Qwen base sin ajustar", base), ("uniforme", uniform)):
        rows += summarize(val, metrics(val, probs), system)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "internal_val.csv", index=False)
    show = table.copy()
    for c in ("KL sin ponderar", "KL ponderada", "MAE (pp)", "top-1"):
        show[c] = show[c].map(lambda v: f"{v:.3f}")
    md = ["# Validación interna de a1 (estudios no vistos por a1; orden de opciones original, semilla 0)", "",
          f"{len(val)} ejemplos de {val.study.nunique()} estudios, {int((val.scope == 'nacional').sum())} nacionales.", "",
          "| " + " | ".join(show.columns) + " |", "|" + "---|" * len(show.columns),
          *["| " + " | ".join(str(v) for v in row) + " |" for row in show.to_numpy()]]
    (OUT / "internal_val.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
