import numpy as np
from scipy.spatial.distance import jensenshannon


def aggregate(segments, predictions, options, question_type="single"):
    expected = {s["segment_id"] for s in segments}
    if len(expected) != len(segments) or set(predictions) != expected:
        raise ValueError("Segmentos incompletos o duplicados: se aborta la estimación nacional")
    total = sum(s["population_weight"] for s in segments)
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Peso total inválido")
    result = dict.fromkeys(options, 0.0)
    for segment in segments:
        weight = segment["population_weight"]
        if not np.isfinite(weight) or weight <= 0:
            raise ValueError("Peso de segmento inválido")
        prediction = predictions[segment["segment_id"]]
        prediction.validate_context(segment["segment_id"], options, prediction.evidence_ids, question_type)
        for option in options:
            result[option] += weight * prediction.probabilities[option] / total
    if question_type == "single" and not np.isclose(sum(result.values()), 1, atol=1e-6):
        raise ValueError("Distribución final inválida")
    return result


def _homogeneous(segment):
    return set(segment.get("homogeneous_dimensions", segment["demographic_profile"]))


def subgroup_resolution(segments):
    """Fracción del peso en que cada dimensión distingue a los segmentos (no está mezclada)."""
    total = sum(s["population_weight"] for s in segments)
    dimensions = set().union(*(_homogeneous(s) for s in segments)) if segments else set()
    return {
        d: sum(s["population_weight"] for s in segments if d in _homogeneous(s)) / total
        for d in sorted(dimensions)
    }


def _share(segment, dimension, category):
    if dimension in _homogeneous(segment):
        return 1.0 if segment["demographic_profile"][dimension] == category else 0.0
    return segment.get("socioeconomic_profile", {}).get(dimension, {}).get(category, 0.0)


def subgroup_distributions(segments, predictions, options, question_type="single", min_resolution=0.6):
    """Estimación por subgrupo ponderando cada segmento por su composición en la dimensión.

    Solo se publican dimensiones que la segmentación distingue en al menos ``min_resolution`` del peso:
    en el resto se asumiría que todos los miembros del segmento responden igual y se inventarían diferencias.
    """
    result = {}
    for dimension, resolution in subgroup_resolution(segments).items():
        if resolution < min_resolution:
            continue
        categories = set()
        for s in segments:
            if dimension in _homogeneous(s):
                categories.add(s["demographic_profile"][dimension])
            else:
                categories.update(s.get("socioeconomic_profile", {}).get(dimension, {}))
        result[dimension] = {}
        for category in sorted(categories):
            weighted = [
                {**s, "population_weight": s["population_weight"] * _share(s, dimension, category)}
                for s in segments
                if _share(s, dimension, category) > 0
            ]
            if not weighted:
                continue
            ids = {s["segment_id"] for s in weighted}
            result[dimension][category] = aggregate(
                weighted, {k: v for k, v in predictions.items() if k in ids}, options, question_type
            )
    return result


def mae(predicted, observed):
    if set(predicted) != set(observed):
        raise ValueError("Opciones incompatibles en evaluación")
    return float(np.mean([abs(predicted[k] - observed[k]) for k in predicted]))


def js_divergence(first, second):
    if set(first) != set(second):
        raise ValueError("Opciones incompatibles")
    a, b = [first[k] for k in sorted(first)], [second[k] for k in sorted(first)]
    if (
        any(not np.isfinite(v) or v < 0 for v in a + b)
        or not np.isclose(sum(a), 1)
        or not np.isclose(sum(b), 1)
    ):
        raise ValueError("JSD requiere distribuciones de respuesta única válidas")
    return float(jensenshannon(a, b, base=2) ** 2)


def calibrated_intervals(prediction, historical_errors, alpha=0.1):
    if not 0 < alpha < 1:
        raise ValueError("alpha inválido")
    if len(historical_errors) < 20:
        return None
    errors = np.asarray(historical_errors, dtype=float)
    if not np.isfinite(errors).all() or (errors < 0).any() or (errors > 1).any():
        raise ValueError("Errores históricos inválidos")
    rank = min(len(errors), int(np.ceil((len(errors) + 1) * (1 - alpha))))
    radius = float(np.sort(errors)[rank - 1])
    return {k: [max(0, p - radius), min(1, p + radius)] for k, p in prediction.items()}


def audit(evidence, metadata, predictions):
    return {
        "evidence_quality": {
            "n_retrieved": len(evidence),
            "sources": [e["evidence"]["source"] for e in evidence],
            "fieldwork_dates": [e["evidence"].get("fieldwork_date") for e in evidence],
            "sample_sizes": [e["evidence"].get("sample_size") for e in evidence],
            "variable_provenance": metadata.variable_provenance,
        },
        "prediction_stability": {"measured": False, "js_divergence": None},
        "historical_accuracy": {"calibrated": False, "mae": None},
        "predictive_intervals": None,
        "reliability": "Fiabilidad predictiva todavía no calibrada",
        "limitations": [
            "Estabilidad no implica precisión",
            "Las imputaciones añaden incertidumbre",
            "El conocimiento preentrenado puede contener preguntas históricas",
        ],
    }
