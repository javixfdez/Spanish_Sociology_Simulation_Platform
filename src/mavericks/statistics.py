import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

from .schemas import Prediction


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


def pairwise_divergence(distributions, question_type="single"):
    """Divergencia media entre pares de repeticiones: JSD (respuesta única) o MAE (múltiple)."""
    if len(distributions) < 2:
        return None
    values = []
    for i in range(len(distributions)):
        for j in range(i + 1, len(distributions)):
            a, b = distributions[i], distributions[j]
            values.append(js_divergence(a, b) if question_type == "single" else mae(a, b))
    return float(np.mean(values))


def stability_summary(segments, predictions, question_type="single", options=None):
    """Estabilidad entre repeticiones: por segmento (ponderada por peso) y a nivel nacional."""
    weights, divergences, runs = [], [], []
    for s in segments:
        info = predictions[s["segment_id"]].get("stability")
        if info and info.get("mean_pairwise_divergence") is not None:
            weights.append(s["population_weight"])
            divergences.append(info["mean_pairwise_divergence"])
            runs.append(info["runs"])
    if not divergences:
        return {"measured": False, "runs": 1, "js_divergence": None}
    k = min(runs)
    national = []
    if options:
        for r in range(k):
            try:
                per_run = {
                    s["segment_id"]: Prediction(
                        segment_id=s["segment_id"],
                        probabilities=predictions[s["segment_id"]].get("repetitions", [predictions[s["segment_id"]]["probabilities"]] * k)[r],
                        explanation="",
                    )
                    for s in segments
                }
                national.append(aggregate(segments, per_run, options, question_type))
            except (ValueError, IndexError, KeyError):
                national = []
                break
    return {
        "measured": True,
        "runs": k,
        "metric": "jsd" if question_type == "single" else "mae",
        "js_divergence": float(np.average(divergences, weights=weights)),
        "max_segment_divergence": float(max(divergences)),
        "national_divergence": pairwise_divergence(national, question_type) if len(national) > 1 else None,
    }


def build_calibration(results, model=None, population_version=None, stability_runs=1, min_cases=20):
    """Errores históricos de un backtest, listos para calibrar la confianza."""
    done = [r for r in results if r["status"] == "completed"]

    def pack(items):
        return {
            "n_cases": len(items),
            "option_errors": [e for r in items for e in r["option_errors"]],
            "subgroup_option_errors": [e for r in items for e in r.get("subgroup_option_errors", [])],
            "mae": float(np.mean([r["mae"] for r in items])) if items else None,
            "mae_uniform_baseline": float(np.mean([r["mae_uniform_baseline"] for r in items])) if items else None,
        }

    return {
        "created": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "population_version": population_version,
        "stability_runs": stability_runs,
        "min_cases": min_cases,
        "overall": pack(done),
        "by_anchor_mode": {m: pack([r for r in done if r["anchor_mode"] == m]) for m in sorted({r["anchor_mode"] for r in done})},
        "by_topic": {t: pack([r for r in done if str(r.get("topic")) == t]) for t in sorted({str(r.get("topic")) for r in done})},
        "synthetic": any(r.get("synthetic") for r in done),
    }


def load_calibration(path):
    path = Path(path) if path else None
    if not path or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def select_stratum(calibration, anchor_mode, topic=None, min_cases=20):
    """Estrato de calibración más específico con suficientes casos (modo de ancla, si no el global)."""
    if not calibration:
        return None, None
    for name, block in [
        (f"modo de ancla '{anchor_mode}'", calibration.get("by_anchor_mode", {}).get(anchor_mode)),
        ("global", calibration.get("overall")),
    ]:
        if block and block["n_cases"] >= min_cases and len(block["option_errors"]) >= min_cases:
            return name, block
    return None, None


def audit(
    evidence,
    metadata,
    predictions,
    *,
    stability=None,
    calibration=None,
    national=None,
    anchor_mode="none",
    topic=None,
    model=None,
    alpha=0.1,
    min_cases=20,
):
    basis, block = select_stratum(calibration, anchor_mode, topic, min_cases)
    limitations = [
        "Estabilidad no implica precisión",
        "Las imputaciones añaden incertidumbre",
        "El conocimiento preentrenado puede contener preguntas históricas",
    ]
    intervals, accuracy, confidence = None, {"calibrated": False, "mae": None}, None
    reliability = "Fiabilidad predictiva todavía no calibrada"
    if block and national:
        intervals_map = calibrated_intervals(national, block["option_errors"], alpha)
        radius = float(np.sort(block["option_errors"])[min(len(block["option_errors"]), int(np.ceil((len(block["option_errors"]) + 1) * (1 - alpha)))) - 1])
        intervals = {"coverage": 1 - alpha, "radius": radius, "intervals": intervals_map, "basis": basis, "n_cases": block["n_cases"]}
        skill = None
        if block["mae"] is not None and block["mae_uniform_baseline"]:
            skill = float(np.clip(1 - block["mae"] / block["mae_uniform_baseline"], 0, 1))
        accuracy = {
            "calibrated": True,
            "mae": block["mae"],
            "mae_uniform_baseline": block["mae_uniform_baseline"],
            "skill_vs_uniform": skill,
            "n_cases": block["n_cases"],
            "basis": basis,
        }
        level = "alta" if (skill or 0) >= 0.6 else "media" if (skill or 0) >= 0.3 else "baja"
        confidence = {"level": level, "skill_score": skill, "expected_abs_error_per_option": radius}
        reliability = (
            f"Calibrada con {block['n_cases']} casos de backtest ({basis}): el error absoluto por opción "
            f"fue ≤ {radius * 100:.1f} puntos en el {int((1 - alpha) * 100)} % de los casos históricos."
        )
        if calibration.get("synthetic"):
            limitations.append("La calibración procede de un backtest con modelo o población de demostración")
        if model and calibration.get("model") and model != calibration["model"]:
            limitations.append(f"Calibración obtenida con otro modelo ({calibration['model']}), no con {model}")
        if calibration.get("population_version") and calibration["population_version"] != metadata.population_version:
            limitations.append("Calibración obtenida con otra versión de población")
        limitations.append("Los intervalos asumen que la pregunta es comparable a las del backtest (cambio de dominio no cubierto)")
    elif calibration:
        limitations.append(
            f"Calibración insuficiente para el modo '{anchor_mode}': menos de {min_cases} casos de backtest"
        )
    return {
        "evidence_quality": {
            "n_retrieved": len(evidence),
            "sources": [e["evidence"]["source"] for e in evidence],
            "fieldwork_dates": [e["evidence"].get("fieldwork_date") for e in evidence],
            "sample_sizes": [e["evidence"].get("sample_size") for e in evidence],
            "variable_provenance": metadata.variable_provenance,
            "anchor_mode": anchor_mode,
        },
        "prediction_stability": stability or {"measured": False, "js_divergence": None},
        "historical_accuracy": accuracy,
        "predictive_intervals": intervals,
        "confidence": confidence,
        "reliability": reliability,
        "limitations": limitations,
    }
