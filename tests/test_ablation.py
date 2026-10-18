import numpy as np
import pytest

from mavericks import ablation


def base(i, mode="none", topic="general"):
    observed = {"a": 0.6, "b": 0.3, "c": 0.1}
    return {
        "evidence_id": f"C{i}", "status": "completed", "anchor_mode": mode, "topic": topic, "observed": observed,
        "mae_uniform_baseline": 0.2, "mae_national_baseline": 0.15,
        "baseline_national": {"distribution": {"a": 0.5, "b": 0.3, "c": 0.2}},
        "predicted": {"a": 0.4, "b": 0.4, "c": 0.2}, "execution_time": 100.0,
    }


def test_rows_are_homogeneous_and_grouped_by_reference_mode():
    bases = {f"C{i}": base(i, "prior" if i % 2 else "none") for i in range(20)}
    ids = list(bases)
    llm = [{**b, "predicted": {"a": 0.55, "b": 0.3, "c": 0.15}, "anchor_mode": "direct"} for b in bases.values()]
    rows = ablation.from_results(llm, bases, ids)
    assert {r["anchor_mode"] for r in rows} == {"prior", "none"}  # manda el modo de referencia
    assert rows[0]["mae"] == pytest.approx(np.mean([0.05, 0, 0.05]))
    assert ablation.uniform_rows(bases, ids)[0]["predicted"] == {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    assert len(ablation.national_rows(bases, ids)) == 20


def test_compare_counts_and_table_have_all_columns():
    bases = {f"C{i}": base(i) for i in range(30)}
    ids = list(bases)
    reference = ablation.from_results(list(bases.values()), bases, ids)
    better = ablation.from_results(
        [{**b, "predicted": {"a": 0.6, "b": 0.3, "c": 0.1}} for b in bases.values()], bases, ids
    )
    cmp_ = ablation.compare_rows(reference, better)
    assert cmp_["better"] == 30 and cmp_["worse"] == 0 and cmp_["mean_delta"] < 0 and cmp_["ci"][1] < 0
    text = ablation.table({"baseline_v1": reference, "mejor": better})
    assert "mejoran / empeoran" in text and "30 / 0 / 0" in text
    assert "general" in ablation.topic_table({"baseline_v1": reference})


def test_wilcoxon_p_value_in_comparison_and_table():
    # 12 casos que mejoran todos en la misma dirección: p exacto bilateral = 2 / 2**12
    bases = {f"C{i}": base(i) for i in range(12)}
    ids = list(bases)
    reference = ablation.from_results(list(bases.values()), bases, ids)
    deltas = [0.001 * (i + 1) for i in range(12)]
    better = ablation.from_results(
        [{**b, "predicted": {"a": 0.4 + d, "b": 0.4 - d / 2, "c": 0.2 - d / 2}} for b, d in zip(bases.values(), deltas, strict=True)],
        bases, ids,
    )
    cmp_ = ablation.compare_rows(reference, better)
    assert cmp_["p_value"] == pytest.approx(2 / 2**12)
    text = ablation.table({"baseline_v1": reference, "mejor": better})
    assert "Wilcoxon p" in text and "<0,001" in text
    few = ablation.compare_rows(reference[:5], better[:5])
    assert few["p_value"] is None
    assert "n insuficiente" in ablation.table({"baseline_v1": reference[:5], "mejor": better[:5]})
