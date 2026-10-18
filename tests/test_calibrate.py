import numpy as np
import pytest

from mavericks.calibrate import (
    Blend,
    blend,
    compare_variants,
    fit,
    jsd,
    leave_one_topic_out,
    log_loss,
    metrics,
)


def _rows(n=48, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        truth = rng.dirichlet([0.7, 0.7, 0.7])
        observed = dict(zip("ABC", truth))
        # el prior es bueno cuando la similitud es alta; el LLM, ruido plano
        sim = float(rng.random())
        good = np.clip(truth + rng.normal(0, 0.03 if sim >= 0.5 else 0.25, 3), 0.01, None)
        rows.append({
            "evidence_id": f"c{i}", "topic": f"t{i % 4}", "mode": "none", "sim": sim, "observed": observed,
            "p_llm": {"A": 1 / 3, "B": 1 / 3, "C": 1 / 3}, "p_prior": dict(zip("ABC", good / good.sum())),
        })
    return rows


def test_blend_identity_and_extremes():
    llm, prior = {"A": 0.6, "B": 0.4}, {"A": 0.2, "B": 0.8}
    assert blend(llm, prior, 1.0, 1.0) == llm and blend(llm, prior, 0.0, 1.0) == prior
    mixed = blend(llm, prior, 0.5, 1.0)
    assert mixed["A"] == pytest.approx(0.4) and blend(llm, prior, 0.5, 2.0)["B"] > mixed["B"]


def test_losses_are_zero_for_exact_predictions_and_positive_otherwise():
    p = {"A": 0.7, "B": 0.3}
    assert jsd(p, p) == pytest.approx(0, abs=1e-12) and jsd({"A": 0.5, "B": 0.5}, p) > 0
    assert log_loss(p, p) < log_loss({"A": 0.5, "B": 0.5}, p)


def test_fit_learns_to_trust_the_prior_only_where_similarity_is_high():
    rows = _rows()
    model = fit(rows, edges=(0.0, 0.5, 1.01))
    (w_low, _), (w_high, _) = model.params
    assert w_high <= 0.2  # con similitud alta el prior manda (w pequeño)
    assert Blend((0.0, 0.5, 1.01), model.params).bin(0.7) == 1


def test_leave_one_topic_out_never_trains_on_the_held_out_topic():
    rows = _rows()
    held, fitted = leave_one_topic_out(rows)
    assert len(held) == len(rows) and all(h is not None for h in held) and set(fitted) == {"t0", "t1", "t2", "t3"}
    table = compare_variants(rows)
    assert table["w y α conjuntos por similitud"]["mae"] < table["solo LLM"]["mae"]
    assert {"solo LLM", "solo prior (analog)", "w constante, sin afilar"} <= set(table)
    assert metrics(rows, [r["observed"] for r in rows])["simbench"] == pytest.approx(100.0)
