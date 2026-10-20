"""Comprobación previa (Fase 1.5): con el modelo base SIN ajustar, 20 preguntas del CIS: letras A-G de un solo token y lectura de probabilidades.

    set PYTHONPATH=src
    .venv_ft\\Scripts\\python -m mavericks.ft.probe
"""

import time

import numpy as np
import pandas as pd
import torch

from . import train


def main(n=20, seed=0):
    frame = train.load_frame("A")
    sample = frame[(frame.source == "CIS") & (frame.scope == "nacional")].sample(n, random_state=seed).reset_index(drop=True)
    tok, model = train.load_model(trainable=False)
    model.eval()
    ids = train.letter_ids(tok)
    print("ids de las letras A-G (un token cada una):", dict(zip("ABCDEFG", ids, strict=True)))
    kls, maes, masses, seconds = [], [], [], []
    for i in range(n):
        enc, props, weights = train.make_batch(sample.iloc[i : i + 1], 0, tok, shuffle=False)
        t = time.time()
        with torch.no_grad():
            log_q, valid = train.letter_log_probs(model, enc, ids, [len(props[0])])
            full = model(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda(), logits_to_keep=1).logits[0, -1].float()
        seconds.append(time.time() - t)
        q = log_q.exp()[0, : len(props[0])].cpu().numpy()
        p = np.array(props[0])
        kls.append(float((p * (np.log(np.clip(p, 1e-12, None)) - np.log(np.clip(q, 1e-12, None)))).sum()))
        maes.append(float(np.abs(p - q).mean()))
        masses.append(float(torch.softmax(full, -1)[ids[: len(p)]].sum()))
        if i < 3:
            print(f"- {sample.question[i][:70]!r} real {np.round(p, 2).tolist()} leído {np.round(q, 2).tolist()}")
    print(f"modelo base sin ajustar: KL media {np.mean(kls):.3f} | MAE medio {100 * np.mean(maes):.1f} puntos (el reparto uniforme daría {100 * np.mean([np.abs(np.array(r) - 1 / len(r)).mean() for r in sample.props]):.1f})")
    print(f"masa de probabilidad del primer token en las letras válidas: media {np.mean(masses):.3f}, mínima {np.min(masses):.3f}")
    print(f"segundos por pregunta (una a una, incluida la primera): media {np.mean(seconds):.2f}")
    print(f"VRAM máxima: {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    pd.DataFrame({"kl": kls, "mass": masses}).to_csv("outputs/ft/probe.csv", index=False)


if __name__ == "__main__":
    main()
