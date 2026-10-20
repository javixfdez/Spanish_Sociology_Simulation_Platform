"""Suelo de ruido de los subgrupos: error esperado solo por muestreo, dado el n real del CIS en cada subgrupo.

Un predictor perfecto de la distribución poblacional de un subgrupo tendría igualmente un MAE > 0 frente a lo observado
porque lo observado es una muestra de n entrevistados. Se estima por bootstrap de los propios microdatos (con pesos).
Sirve para decidir si repartir el total por igual (plano), usar desviaciones empíricas o llamar al LLM aporta algo
que el ruido no tape.
"""

import numpy as np

from .anchor import constraints_of  # noqa: F401  (reexport útil para scripts)
from .backtest_cases import SUBGROUP_DIMENSIONS, clean_option

MIN_SUBGROUP = 100


def _distribution(answers, weights, options):
    sums = {o: 0.0 for o in options}
    for a, w in zip(answers, weights):
        if a in sums:
            sums[a] += w
    total = sum(sums.values())
    return {o: v / total for o, v in sums.items()} if total > 0 else None


def bootstrap_floor(answers, weights, options, rng, draws=200):
    """MAE medio entre la distribución de una remuestra y la de la muestra completa (error por muestreo)."""
    answers, weights = np.asarray(answers), np.asarray(weights, dtype=float)
    full = _distribution(answers, weights, options)
    if full is None:
        return None
    n = len(answers)
    errors = []
    for _ in range(draws):
        idx = rng.integers(0, n, n)
        boot = _distribution(answers[idx], weights[idx], options)
        if boot is not None:
            errors.append(np.mean([abs(boot[o] - full[o]) for o in options]))
    return float(np.mean(errors)) if errors else None


def case_frame(anchors, study_id, question_uid, options):
    """Respuestas del estudio con demografía, limpias y restringidas a las opciones sustantivas del caso."""
    frame = anchors._frame(study_id, question_uid).copy()
    frame["answer"] = frame.answer.map(clean_option)
    return frame[frame.answer.isin(options)]


def subgroup_rows(case, anchors, predictors, rng, draws=200):
    """Una fila por (caso, dimensión, categoría) con el suelo y el MAE de cada predictor.

    ``predictors``: {nombre: función(dimensión, categoría) -> distribución o None}.
    """
    frame = case_frame(anchors, case["estudio_id"], case["evidence_id"], case["options"])
    rows = []
    for dimension in SUBGROUP_DIMENSIONS:
        for category, target in case.get("observed_subgroups", {}).get(dimension, {}).items():
            part = frame[frame[dimension] == category]
            if len(part) < MIN_SUBGROUP:
                continue
            floor = bootstrap_floor(part.answer.to_numpy(), part.peso.to_numpy(), case["options"], rng, draws)
            row = {"evidence_id": case["evidence_id"], "topic": case["topic"], "dimension": dimension,
                   "category": category, "n": int(target["n"]), "floor": floor}
            for name, predict in predictors.items():
                dist = predict(dimension, category)
                row[name] = None if dist is None else float(np.mean([abs(dist[o] - target["distribution"][o]) for o in case["options"]]))
            rows.append(row)
    return rows


def case_floors(case, anchors, rng, draws=100):
    """Suelo de ruido de un caso: error por muestreo del total nacional y medio de sus subgrupos."""
    frame = case_frame(anchors, case["estudio_id"], case["evidence_id"], case["options"])
    national = bootstrap_floor(frame.answer.to_numpy(), frame.peso.to_numpy(), case["options"], rng, draws)
    rows = subgroup_rows(case, anchors, {}, rng, draws)
    floors = [r["floor"] for r in rows if r["floor"] is not None]
    return {
        "national_floor": national,
        "subgroup_floor": float(np.mean(floors)) if floors else None,
        "n": int(frame.respondent_uid.nunique()),
    }


def summarize(rows, names):
    out = {"subgroups": len(rows), "cases": len({r["evidence_id"] for r in rows})}
    for key in ("floor", *names):
        values = [r[key] for r in rows if r.get(key) is not None]
        out[key] = float(np.mean(values)) if values else None
    return out
