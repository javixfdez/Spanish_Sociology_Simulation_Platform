import json
from pathlib import Path

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
            "excluded_texts": [case["question"], *case.get("equivalent_texts", [])],
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
        predicted = result["national_distribution"]
        # Observed answers are accessed only after simulation; never enter the model payload.
        observed = case["observed_distribution"]
        failed = result["execution_metadata"]["status"] != "completed"
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
