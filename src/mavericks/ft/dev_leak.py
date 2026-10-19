"""Búsqueda de fuga tras el resultado de dev_grande (mejora de a1 > 4 pp): sensibilidades que quitan las preguntas con texto idéntico a una de
entrenamiento y las variables de perfil sociodemográfico. Diagnóstico descriptivo; no cambia el criterio prerregistrado.

    python -m mavericks.ft.dev_leak
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..prior_v2.bigdev_eval import is_profile, study_bootstrap

OUT = Path("outputs/ft/phase3")
REF = "Prior de producción (50/50 KNN+MLP)"


def main():
    err = pd.read_csv(OUT / "dev_grande_errors.csv")
    ex = pd.read_pickle("outputs/ft/data/examples.pkl")
    norm = lambda s: re.sub(r"\W+", " ", str(s).lower()).strip()  # noqa: E731
    train_text = {norm(q) for q in ex[ex.in_a].question}
    dev = ex[(ex.scope == "nacional") & ex.evidence_id.isin(err.evidence_id)].set_index("evidence_id")
    err["same_text"] = err.evidence_id.map(lambda e: norm(dev.loc[e, "question"]) in train_text)
    err["profile"] = err.evidence_id.map(is_profile)
    lines = ["# Búsqueda de fuga (mejora de a1 > 4 pp)", "", "| Subconjunto | preguntas | estudios | MAE prior | MAE a1 | Δ a1 − prior [IC 95 % por estudio] | MAE 50/50 | Δ 50/50 − prior [IC] |", "|---|---|---|---|---|---|---|---|"]
    sets = {"todas": np.ones(len(err), bool), "sin texto idéntico a uno de entrenamiento": ~err.same_text, "solo texto idéntico": err.same_text,
            "sin variables de perfil": ~err.profile, "solo variables de perfil": err.profile, "sin ambos": ~err.same_text & ~err.profile}
    for name, mask in sets.items():
        d = err[mask]
        cells = []
        for system in ("a1", "Media 50/50 a1 + prior"):
            delta = (d[system] - d[REF]).to_numpy()
            lo, hi = study_bootstrap(delta, d.study.astype(str).to_numpy(), draws=10_000, seed=0)
            cells.append((d[system].mean(), delta.mean(), lo, hi))
        lines.append(f"| {name} | {len(d)} | {d.study.nunique()} | {d[REF].mean():.2f} | {cells[0][0]:.2f} | {cells[0][1]:+.2f} [{cells[0][2]:+.2f}, {cells[0][3]:+.2f}] | "
                     f"{cells[1][0]:.2f} | {cells[1][1]:+.2f} [{cells[1][2]:+.2f}, {cells[1][3]:+.2f}] |")
    top = err.assign(gain=err[REF] - err["a1"]).sort_values("gain", ascending=False).head(10)
    lines += ["", "Diez preguntas con mayor mejora de a1 (id, MAE prior, MAE a1):", *[f"- {r.evidence_id}: {r[REF]:.1f} → {r.a1:.1f}" for _, r in top.iterrows()]]
    (OUT / "dev_leak.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
