"""Artefactos de producción derivados de las validaciones: cuantiles de los rangos y tabla de «Validación con datos no vistos».

Se generan con ``mavericks calibrate-prior`` a partir de las predicciones guardadas de dev_grande y de las dos confirmaciones
prerregistradas. Cuantiles de los rangos: solo con las preguntas nuevas de dev_grande (calibración); las confirmaciones solo validan.
"""

import json
from pathlib import Path

import numpy as np

from ..fastcal import scale  # noqa: F401  (la misma escala sqrt(p(1-p)+0,02) de los rangos)
from . import bigdev_eval, data, task3
from .task1 import OUT

SETS = {
    "dev_grande": ("bigdev_preds.pkl", "Desarrollo grande (2025-07 a 2026-03)", "elegida aquí"),
    "confirmacion_2025h1": ("confirm_preds.pkl", "Confirmación prerregistrada (2025-01 a 2025-06)", "datos no vistos"),
    "confirmacion_2024h2": ("confirm2_preds.pkl", "Segunda confirmación (2024-07 a 2024-12)", "datos no vistos"),
}
GROUPS = ("2 opciones", "3 o más")


def average(row):
    return 0.5 * row["knn_now"] + 0.5 * row["mlp"]


def group_of(row):
    return "2 opciones" if row["n_options"] == 2 else "3 o más"


def new_questions(name):
    return [r for r in data.load(OUT / SETS[name][0]) if not r["repeated"]]


def config_errors(rows):
    fns = {"uniforme": lambda r: r["uniform"], "knn_actual": lambda r: r["knn_now"], "media_knn_mlp": average}
    return {n: np.array([bigdev_eval.mae(f(r), r["observed"]) for r in rows]) for n, f in fns.items()}


def summarize_set(rows):
    """MAE, IC por estudio y Wilcoxon de cada configuración (frente al KNN actual) para un conjunto y sus dos grupos."""
    out = {"preguntas": len(rows), "estudios": len({r["study"] for r in rows}), "grupos": {}}
    for label, subset in (("todas", rows), *((g, [r for r in rows if group_of(r) == g]) for g in GROUPS)):
        if not subset:
            continue
        errors = config_errors(subset)
        studies = [r["study"] for r in subset]
        cell = {"preguntas": len(subset), "estudios": len(set(studies)), "configuraciones": {}}
        for name, err in errors.items():
            entry = {"mae": float(err.mean())}
            if name != "knn_actual":
                c = bigdev_eval.compare(errors["knn_actual"], err, studies)
                entry.update(delta_vs_knn=c["delta"], ci=list(c["ci"]), mejoran=c["better"], empeoran=c["worse"], p=c["p"])
            if name == "media_knn_mlp":
                u = bigdev_eval.compare(errors["uniforme"], err, studies)
                entry.update(delta_vs_uniforme=u["delta"], ci_vs_uniforme=list(u["ci"]), p_vs_uniforme=u["p"])
            cell["configuraciones"][name] = entry
        out["grupos"][label] = cell
    return out


def interval_artifact():
    """Cuantiles por grupo (nivel 90 % y 50 %) con las nuevas de dev_grande y su cobertura en cada conjunto."""
    calibration = new_questions("dev_grande")
    groups = {}
    for g in GROUPS:
        rows = [r for r in calibration if group_of(r) == g]
        scores = np.concatenate([task3.case_scores(r, average)[1] for r in rows])
        groups[g] = {
            "q90": task3._quantile(scores, 0.9), "q50": task3._quantile(scores, 0.5), "preguntas": len(rows),
        }
    coverage = {}
    for level in (0.9, 0.5):
        loso = task3.leave_one_study_out_by(calibration, average, group_of, level)
        coverage[f"dev_grande_{int(level * 100)}"] = {k: {"cobertura": v["coverage"], "anchura_pp": v["width_pp"]} for k, v in loso.items()}
        pooled_rows = new_questions("confirmacion_2025h1") + new_questions("confirmacion_2024h2")
        conf, _ = task3.calibrate_and_test(calibration, pooled_rows, average, group_of, level)
        coverage[f"confirmaciones_{int(level * 100)}"] = {k: {"cobertura": v["coverage"], "anchura_pp": v["width_pp"]} for k, v in conf.items()}
    return {"groups": groups, "scale_floor": 0.02, "calibrada_con": "preguntas nuevas de dev_grande", "coverage": coverage}


def validation_artifact():
    sets = {}
    for name, (_, title, status) in SETS.items():
        sets[name] = {"titulo": title, "estado": status, **summarize_set(new_questions(name))}
    # error típico de la media (para «Confianza»): conjunto de los tres, ponderado por preguntas
    typical = {}
    for label in ("todas", *GROUPS):
        n = sum(s["grupos"][label]["preguntas"] for s in sets.values() if label in s["grupos"])
        mae = sum(s["grupos"][label]["preguntas"] * s["grupos"][label]["configuraciones"]["media_knn_mlp"]["mae"] for s in sets.values() if label in s["grupos"]) / n
        typical[label] = {"mae": mae, "preguntas": n}
    return {"sets": sets, "error_tipico": typical}


def build(out_dir="outputs/models"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    intervals, validation = interval_artifact(), validation_artifact()
    (out / "prior_intervals.json").write_text(json.dumps(intervals, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "prior_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"intervals": intervals["groups"], "error_tipico": validation["error_tipico"]}
