import json
from pathlib import Path

import numpy as np

from .anchor import option_key, option_mapping, question_similarity
from .graph import simulate
from .schemas import Question
from .statistics import js_divergence, mae


async def backtest(
    cases,
    repository,
    evidence_repository,
    service_factory,
    max_segments=5,
    anchors=None,
    calibration=None,
    on_result=None,
):
    results = []
    for case in cases:
        # Human-reviewed equivalence families complement exact text and ID exclusion.
        excluded = list({case["evidence_id"], *case.get("equivalent_ids", [])})
        filters = {
            "excluded_ids": excluded,
            # Modo temporal: la misma pregunta de oleadas anteriores es evidencia legítima (el corte
            # estricto por fecha ya impide usar el futuro); el modo estricto excluye cualquier texto igual.
            "excluded_texts": [
                *([] if case.get("allow_previous_waves") else [case["question"]]),
                *case.get("equivalent_texts", []),
            ],
            "cutoff": case.get("fieldwork_date"),
        }
        question = Question(
            question=case["question"],
            options=case["options"],
            question_type=case.get("question_type", "single"),
        )
        result = await simulate(
            question,
            repository,
            evidence_repository,
            service_factory(),
            max_segments,
            retrieval_filters=filters,
            anchors=anchors,
            calibration=calibration,
        )
        retrieved_ids = {e["evidence"]["evidence_id"] for e in result["retrieved_evidence"]}
        if retrieved_ids & set(excluded):
            raise ValueError("Fuga de evidencia en backtesting")
        cutoff = case.get("fieldwork_date")
        if cutoff and any(
            not (e["evidence"].get("fieldwork_date") or "9999")[:10] < cutoff for e in result["retrieved_evidence"]
        ):
            raise ValueError("Fuga temporal en backtesting: evidencia posterior al corte")
        predicted = result["national_distribution"]
        # Observed answers are accessed only after simulation; never enter the model payload.
        observed = case["observed_distribution"]
        failed = result["execution_metadata"]["status"] != "completed"
        spec = result.get("anchor_spec")
        subgroup_scores = [] if failed else subgroup_errors(result["subgroup_distributions"], case.get("observed_subgroups", {}))
        results.append(
            {
                "evidence_id": case["evidence_id"],
                "excluded_ids": excluded,
                "retrieved_ids": sorted(retrieved_ids),
                "simulation_id": result["simulation_id"],
                "status": "failed" if failed else "completed",
                "synthetic": result["execution_metadata"]["synthetic"],
                "mae": None if failed else mae(predicted, observed),
                "js_divergence": None
                if failed or question.question_type == "multiple"
                else js_divergence(predicted, observed),
                "topic": case.get("topic"),
                "anchor_mode": spec["mode"] if spec else "none",
                "assessment_fallback": "interpret_evidence:fallback" in result["stages"],
                "had_evidence": bool(result["retrieved_evidence"]),
                "execution_time": result["execution_metadata"]["execution_time"],
                "llm_latency": _latencies(result["execution_metadata"]["llm_calls"]),
                "observed": observed,
                **baseline_fields(result, case, anchors, observed, failed),
                "option_errors": [] if failed else [abs(predicted[o] - observed[o]) for o in observed],
                "subgroup_option_errors": option_errors_of(result, case) if not failed else [],
                "stability": result["trust_metrics"]["prediction_stability"] if "trust_metrics" in result else None,
                "mae_uniform_baseline": mae(dict.fromkeys(observed, 1 / len(observed)), observed),
                "subgroup_mae": float(np.mean([e["mae"] for e in subgroup_scores])) if subgroup_scores else None,
                "subgroup_errors": subgroup_scores,
                "predicted": predicted,
                "predicted_subgroups": {} if failed else result["subgroup_distributions"],
                "limitations": [
                    "Equivalencias requieren revisión humana; conocimiento preentrenado no controlable"
                ],
            }
        )
        if on_result:
            on_result(results[-1])
    return results


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def subgroup_errors(predicted, observed):
    """MAE por subgrupo comparable (misma dimensión y categoría) entre lo predicho y lo observado."""
    scores = []
    for dimension, categories in predicted.items():
        for category, distribution in categories.items():
            target = observed.get(dimension, {}).get(category)
            if target and set(target["distribution"]) == set(distribution):
                scores.append(
                    {
                        "dimension": dimension,
                        "category": category,
                        "n": target["n"],
                        "mae": mae(distribution, target["distribution"]),
                    }
                )
    return scores


def national_baseline(result, case, anchors):
    """Distribución nacional (sin segmentar) de la pregunta recuperada más parecida, en las opciones del caso.

    Alinea por etiqueta si las opciones son equivalentes; si no, por orden cuando ambas tienen el mismo número
    de opciones sustantivas (escalas ordinales). Devuelve None si no hay ninguna evidencia comparable.
    """
    if anchors is None:
        return None
    options = case["options"]
    ranked = sorted(
        (h["evidence"] for h in result["retrieved_evidence"]),
        key=lambda e: -question_similarity(case["question"], e.get("question_text") or ""),
    )
    for e in ranked:
        original = e.get("metadata", {}).get("original_fields", {})
        if not e.get("study_id") or original.get("question_type", "single") != "single":
            continue
        try:
            dist, n = anchors.national(e["study_id"], e["evidence_id"], "single")
        except Exception:
            continue
        if not n:
            continue
        mapping = option_mapping(options, e.get("options") or [])
        source = "etiqueta"
        if mapping is None:
            substantive = [o for o in (e.get("options") or []) if option_key(o) not in {"ns", "nc", "np", "no sabe", "no contesta"}]
            if len(substantive) != len(options):
                continue
            mapping, source = dict(zip(options, substantive)), "ordinal"
        values = {o: dist.get(label, 0.0) for o, label in mapping.items()}
        mass = sum(values.values())
        if mass <= 0:
            continue
        return {
            "distribution": {o: v / mass for o, v in values.items()},
            "source": source,
            "evidence_id": e["evidence_id"],
            "similarity": round(question_similarity(case["question"], e.get("question_text") or ""), 3),
        }
    return None


def baseline_fields(result, case, anchors, observed, failed):
    baseline = national_baseline(result, case, anchors)
    return {
        "baseline_national": baseline,
        "mae_national_baseline": mae(baseline["distribution"], observed) if baseline else None,
    }


def summarize_backtest(results):
    """Errores medios globales, por modo de ancla, por tema y por modo x tema, con las dos líneas base."""

    def block(items):
        done = [r for r in items if r["status"] == "completed"]
        sub = [r["subgroup_mae"] for r in done if r.get("subgroup_mae") is not None]
        jsd = [r["js_divergence"] for r in done if r.get("js_divergence") is not None]
        with_base = [r for r in done if r.get("mae_national_baseline") is not None]
        return {
            "cases": len(items),
            "failed": len(items) - len(done),
            "mae": float(np.mean([r["mae"] for r in done])) if done else None,
            "js_divergence": float(np.mean(jsd)) if jsd else None,
            "subgroup_mae": float(np.mean(sub)) if sub else None,
            "mae_uniform_baseline": float(np.mean([r["mae_uniform_baseline"] for r in done])) if done else None,
            "national_baseline": {
                "cases": len(with_base),
                "mae_baseline": float(np.mean([r["mae_national_baseline"] for r in with_base])) if with_base else None,
                "mae_model_same_cases": float(np.mean([r["mae"] for r in with_base])) if with_base else None,
                "wins": sum(r["mae"] < r["mae_national_baseline"] for r in with_base),
            },
            "assessment_fallback_rate": (
                float(np.mean([r["assessment_fallback"] for r in items if r.get("had_evidence")]))
                if any(r.get("had_evidence") for r in items)
                else None
            ),
        }

    summary = {"overall": block(results), "by_topic": {}, "by_anchor_mode": {}, "by_mode_and_topic": {}}
    for key, name in [("topic", "by_topic"), ("anchor_mode", "by_anchor_mode")]:
        for value in sorted({str(r.get(key)) for r in results}):
            summary[name][value] = block([r for r in results if str(r.get(key)) == value])
    for mode in sorted({r["anchor_mode"] for r in results}):
        for topic in sorted({str(r.get("topic")) for r in results}):
            items = [r for r in results if r["anchor_mode"] == mode and str(r.get("topic")) == topic]
            if items:
                summary["by_mode_and_topic"][f"{mode} | {topic}"] = block(items)
    return summary


def option_errors_of(result, case):
    """Errores absolutos por opción de los subgrupos comparables (para calibrar subgrupos)."""
    errors = []
    for dimension, categories in result["subgroup_distributions"].items():
        for category, distribution in categories.items():
            target = case.get("observed_subgroups", {}).get(dimension, {}).get(category)
            if target and set(target["distribution"]) == set(distribution):
                errors.extend(abs(distribution[o] - target["distribution"][o]) for o in distribution)
    return errors


def _latencies(calls):
    """Latencia media (s) y número de llamadas reales al LLM por tipo de agente."""
    summary = {}
    for call in calls:
        if call.get("cache_hit"):
            continue
        entry = summary.setdefault(call["schema"], {"n": 0, "total": 0.0, "retries": 0})
        entry["n"] += 1
        entry["total"] += call.get("latency", 0.0)
        entry["retries"] += int(call.get("attempt", 0))
    return {k: {"n": v["n"], "mean_s": v["total"] / v["n"], "retries": v["retries"]} for k, v in summary.items()}
