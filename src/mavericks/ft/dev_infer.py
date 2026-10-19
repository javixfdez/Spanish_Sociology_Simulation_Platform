"""Fase 3, paso 4 (GPU): probabilidades de a1 y del modelo base sin ajustar para las preguntas nuevas de dev_grande. Se ejecuta en .venv_ft.

    set PYTHONPATH=src
    .venv_ft\\Scripts\\python -m mavericks.ft.dev_infer

Prompts idénticos a los del entrenamiento (examples.pkl, orden original de las opciones, sin barajar). Nacionales: una pregunta cada vez (para medir segundos por
pregunta). Subgrupos del CIS de esas mismas preguntas (los ya muestreados, ≤ 6 por pregunta y dimensión): por lotes. No se vuelve a ejecutar si existe el marcador.
"""

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..prior_v2 import data
from . import train

OUT = Path("outputs/ft/phase3")
ADAPTER = Path("outputs/ft/runs/a1/adapter_final")


def read(model, tok, ids, rows, micro):
    enc, props, _ = train.make_batch(rows, 0, tok, shuffle=False)
    log_q, _ = train.letter_log_probs(model, enc, ids, [len(p) for p in props])
    return [log_q[i].exp()[: len(p)].float().cpu().numpy() for i, p in enumerate(props)]


@torch.no_grad()
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = data.load("outputs/prior_v2/bigdev_preds.pkl")
    new = {r["evidence_id"] for r in rows if not r["repeated"]}
    frame = pd.read_pickle(train.EXAMPLES)
    frame = frame[(frame.source == "CIS") & frame.evidence_id.isin(new)].reset_index(drop=True)
    assert len(frame) and (frame.date >= "2025-07-01").all() and (frame.date < "2026-04-01").all() and not frame.in_a.any()
    national = frame[frame.scope == "nacional"].reset_index(drop=True)
    groups = frame[frame.scope != "nacional"].reset_index(drop=True)
    assert len(national) == len(new) == 703, (len(national), len(new))
    print(f"{len(national)} nacionales, {len(groups)} subgrupos")
    tok, model = train.load_model(adapter=ADAPTER, trainable=False)
    model.eval()
    ids = train.letter_ids(tok)
    out = {"base": {}, "a1": {}}
    seconds = {"base": [], "a1": []}
    for name in ("a1", "base"):
        for _, r in national.iterrows():
            t = time.time()
            if name == "a1":
                p = read(model, tok, ids, national.iloc[[_]], 1)[0]
            else:
                with model.disable_adapter():
                    p = read(model, tok, ids, national.iloc[[_]], 1)[0]
            torch.cuda.synchronize()
            seconds[name].append(time.time() - t)
            out[name][r["id"]] = p
        print(name, "nacionales listos;", f"{np.mean(seconds[name][1:]):.3f} s/pregunta", flush=True)
    order = np.argsort(groups.question.str.len().to_numpy() + groups.options.map(len).to_numpy() * 50)
    for name in ("a1", "base"):
        for start in range(0, len(groups), 16):
            batch = groups.iloc[order[start : start + 16]]
            if name == "a1":
                probs = read(model, tok, ids, batch, 16)
            else:
                with model.disable_adapter():
                    probs = read(model, tok, ids, batch, 16)
            for (_, r), p in zip(batch.iterrows(), probs, strict=True):
                out[name][r["id"]] = p
            if start % 1600 == 0:
                print(name, "subgrupos", start, "/", len(groups), flush=True)
    result = {"probs": out, "seconds": {k: float(np.mean(v[1:])) for k, v in seconds.items()}}
    (OUT / "dev_probs.pkl").write_bytes(pickle.dumps(result))
    print("listo", result["seconds"])


if __name__ == "__main__":
    main()
