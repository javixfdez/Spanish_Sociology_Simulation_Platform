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
    """Cuantiles por grupo (nivel 90 % y 50 %) con las preguntas nuevas de los tres conjuntos juntos.

    La cobertura se mide dejando fuera un conjunto (ventana temporal) cada vez: se calibra con los otros dos y se prueba en él.
    """
    by_set = {name: new_questions(name) for name in SETS}
    everything = [r for rows in by_set.values() for r in rows]
    groups = {}
    for g in GROUPS:
        rows = [r for r in everything if group_of(r) == g]
        scores = np.concatenate([task3.case_scores(r, average)[1] for r in rows])
        groups[g] = {"q90": task3._quantile(scores, 0.9), "q50": task3._quantile(scores, 0.5), "preguntas": len(rows)}
    coverage, folds = {}, {}
    for level in (0.9, 0.5):
        pooled = {k: {"hit": 0.0, "n": 0, "width": 0.0} for k in (*GROUPS, "global")}
        for held, test in by_set.items():
            train = [r for name, rows in by_set.items() if name != held for r in rows]
            summary, _ = task3.calibrate_and_test(train, test, average, group_of, level)
            folds[f"{held}_{int(level * 100)}"] = {k: {"cobertura": v["coverage"], "anchura_pp": v["width_pp"], "preguntas": v["cases"]} for k, v in summary.items()}
            for k, v in summary.items():
                pooled[k]["hit"] += v["coverage"] * v["options"]
                pooled[k]["n"] += v["options"]
                pooled[k]["width"] += v["width_pp"] * v["options"]
        coverage[f"fuera_un_conjunto_{int(level * 100)}"] = {
            k: {"cobertura": v["hit"] / v["n"], "anchura_pp": v["width"] / v["n"]} for k, v in pooled.items() if v["n"]
        }
    return {
        "groups": groups, "scale_floor": 0.02, "calibrada_con": "preguntas nuevas de los tres conjuntos (dev_grande y dos confirmaciones)",
        "coverage": coverage, "coverage_por_conjunto": folds,
    }


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
