"""Segmentos coherentes sin LLM: el total sale del prior por analogía y las diferencias entre grupos de los microdatos
del CIS de las propias análogas (misma ponderación por similitud), con raking para reproducir el total."""

import numpy as np

from .anchor import constraints_of
from .population import build_segments
from .schemas import Prediction
from .statistics import aggregate, subgroup_distributions


def segment_priors(result, options, segments, anchors):
    """{segment_id: distribución} = mezcla de las distribuciones del segmento en cada análoga, contraída a la uniforme.

    Si una análoga no tiene microdatos (EB/ESS) aporta su distribución nacional a todos los segmentos.
    """
    uniform = np.full(len(options), 1 / len(options))
    out = {}
    for segment in segments:
        constraints = constraints_of(segment)
        acc, total = np.zeros(len(options)), 0.0
        for weight, c in result["weighted"]:
            dist = c["dist"]
            if anchors is not None and c.get("microdata") and c.get("study_id"):
                spec = {"study_id": c["study_id"], "evidence_id": c["evidence_id"], "mapping": c["mapping"], "question_type": "single"}
                try:
                    dist = anchors.segment_distribution(spec, constraints)["probabilities"]
                except Exception:  # sin microdatos utilizables para este perfil: nivel nacional
                    dist = c["dist"]
            vector = np.array([dist[o] for o in options], dtype=float)
            acc += weight * vector / max(vector.sum(), 1e-12)
            total += weight
        mix = acc / total if total > 0 else uniform
        trust = result["trust"] if total > 0 else 0.0
        out[segment["segment_id"]] = dict(zip(options, (trust * mix + (1 - trust) * uniform).tolist()))
    return out


def rake(distributions, weights, target, iterations=100, tol=1e-9):
    """Reescala las distribuciones por segmento para que su media ponderada reproduzca ``target``."""
    ids = list(distributions)
    options = list(target)
    matrix = np.array([[distributions[i][o] for o in options] for i in ids], dtype=float)
    w = np.array([weights[i] for i in ids], dtype=float)
    w = w / w.sum()
    goal = np.array([target[o] for o in options], dtype=float)
    goal = goal / goal.sum()
    for _ in range(iterations):
        mean = w @ matrix
        factor = goal / np.maximum(mean, 1e-12)
        matrix = matrix * factor
        matrix = matrix / matrix.sum(axis=1, keepdims=True)
        if np.abs(w @ matrix - goal).max() < tol:
            break
    return {i: dict(zip(options, row.tolist())) for i, row in zip(ids, matrix)}


def predict(result, options, population, anchors, n_segments=10, dimensions=None, deviations=True, do_rake=True):
    """Total + segmentos + subgrupos desde un prior por analogía (``combine``). Devuelve un dict con todo."""
    segments = build_segments(population, n_segments, dimensions=dimensions)
    weights = {s["segment_id"]: s["population_weight"] for s in segments}
    target = result["prior"]
    if deviations:
        priors = segment_priors(result, list(target), segments, anchors)
        if do_rake:
            priors = rake(priors, weights, target)
    else:
        priors = {sid: dict(target) for sid in weights}
    predictions = {
        sid: Prediction(segment_id=sid, probabilities=p, explanation="prior por analogía") for sid, p in priors.items()
    }
    options_list = list(target)
    return {
        "segments": segments,
        "segment_predictions": priors,
        "national": aggregate(segments, predictions, options_list, "single", evidence_ids=[]),
        "subgroups": subgroup_distributions(segments, predictions, options_list, "single", evidence_ids=[]),
    }
