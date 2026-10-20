"""Calibración de los rangos del modo ajustado: conformal Mondrian (2 opciones / 3 o más) con los residuos NACIONALES de a1 en las preguntas nuevas de dev_grande.

a1 no vio dev_grande (B sí, por eso no se calibra con B). Misma escala sqrt(p(1-p)+0,02) que el resto de rangos. Escribe ``ft_intervals.json`` y
``ft_validation.json`` en outputs/models (fuera de git) con el mismo esquema que ``prior_intervals.json`` / ``prior_validation.json``.

    python -m mavericks.ft.artifacts
"""

import json
import pickle
from pathlib import Path

import numpy as np

from ..fastcal import _quantile, scale
from ..prior_v2 import data
from ..prior_v2.bigdev_eval import mae

OUT = Path("outputs/models")
GROUPS = ("2 opciones", "3 o más")


def group_of(n_options):
    return "2 opciones" if n_options == 2 else "3 o más"


def build():
    rows = [r for r in data.load("outputs/prior_v2/bigdev_preds.pkl") if not r["repeated"]]
    probs = pickle.loads(Path("outputs/ft/phase3/dev_probs.pkl").read_bytes())["probs"]["a1"]
    scores = {g: [] for g in GROUPS}
    maes = {g: [] for g in GROUPS}
    for r in rows:
        p = np.asarray(probs[f"{r['evidence_id']}|nacional|"], dtype=float)
        p = p / p.sum()
        g = group_of(r["n_options"])
        scores[g].append(np.abs(p - r["observed"]) / scale(p))
        maes[g].append(mae(p, r["observed"]))
    groups = {g: {"q90": _quantile(np.concatenate(scores[g]), 0.9), "q50": _quantile(np.concatenate(scores[g]), 0.5), "preguntas": len(scores[g])} for g in GROUPS}
    intervals = {"groups": groups, "scale_floor": 0.02, "coverage": {}, "calibrada_con": "residuos nacionales de a1 en las preguntas nuevas de dev_grande (a1 no las vio)"}
    typical = {g: {"mae": float(np.mean(maes[g])), "preguntas": len(maes[g]), "basis": "preguntas nuevas de dev_grande (a1 no las vio)"} for g in GROUPS}
    typical["todas"] = {"mae": float(np.mean(sum(maes.values(), []))), "preguntas": len(rows), "basis": typical["2 opciones"]["basis"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ft_intervals.json").write_text(json.dumps(intervals, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "ft_validation.json").write_text(json.dumps({"error_tipico": typical}, ensure_ascii=False, indent=1), encoding="utf-8")
    return intervals, typical


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=1))
