import json
from pathlib import Path

import numpy as np

from .graph import simulate
from .schemas import Question
from .statistics import js_divergence, mae


async def backtest(cases, repository, evidence_repository, service_factory, max_segments=5, anchors=None):
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
                "observed": observed,
                "mae_uniform_baseline": mae(dict.fromkeys(observed, 1 / len(observed)), observed),
                "subgroup_mae": float(np.mean([e["mae"] for e in subgroup_scores])) if subgroup_scores else None,
                "subgroup_errors": subgroup_scores,
                "predicted": predicted,
                "limitations": [
                    "Equivalencias requieren revisión humana; conocimiento preentrenado no controlable"
                ],
            }
        )
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


def summarize_backtest(results):
    """Errores medios globales, por tema y por modo de ancla, junto a la línea base uniforme."""

    def block(items):
        done = [r for r in items if r["status"] == "completed"]
        sub = [r["subgroup_mae"] for r in done if r.get("subgroup_mae") is not None]
        jsd = [r["js_divergence"] for r in done if r.get("js_divergence") is not None]
        return {
            "cases": len(items),
            "failed": len(items) - len(done),
            "mae": float(np.mean([r["mae"] for r in done])) if done else None,
            "js_divergence": float(np.mean(jsd)) if jsd else None,
            "subgroup_mae": float(np.mean(sub)) if sub else None,
            "mae_uniform_baseline": float(np.mean([r["mae_uniform_baseline"] for r in done])) if done else None,
        }

    summary = {"overall": block(results), "by_topic": {}, "by_anchor_mode": {}}
    for key, name in [("topic", "by_topic"), ("anchor_mode", "by_anchor_mode")]:
        for value in sorted({str(r.get(key)) for r in results}):
            summary[name][value] = block([r for r in results if str(r.get(key)) == value])
    return summary
