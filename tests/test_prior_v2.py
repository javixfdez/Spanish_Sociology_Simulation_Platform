import numpy as np
import pytest

from mavericks.prior_v2 import data, knn


def cand(i, lex, qq, qp, dist, date="2024-01-01", kind="label", pools=(True, True, True)):
    return {
        "evidence_id": i, "lex": lex, "cos_qq": qq, "cos_qp": qp, "dist": dist, "date": date, "kind": kind,
        "in_lex": pools[0], "in_qq": pools[1], "in_qp": pools[2],
    }


def case(cands, observed=None):
    return {
        "evidence_id": "Q", "estudio_id": "9", "topic": "general", "options": ["a", "b"],
        "observed": observed or {"a": 0.8, "b": 0.2}, "candidates": cands,
    }


def test_prior_is_weighted_mean_of_top_k_and_respects_pools():
    near = cand("near", 0.9, 0.9, 0.9, {"a": 0.9, "b": 0.1})
    far = cand("far", 0.3, 0.3, 0.3, {"a": 0.1, "b": 0.9})
    arrays = knn.CaseArrays(case([near, far]))
    p = arrays.prior("lex", 2, 1)
    assert p == pytest.approx([(0.9 * 0.9 + 0.3 * 0.1) / 1.2, (0.9 * 0.1 + 0.3 * 0.9) / 1.2])
    assert arrays.prior("lex", 1, 1) == pytest.approx([0.9, 0.1])  # k=1: solo la más parecida
    only_far = knn.CaseArrays(case([near | {"in_lex": False}, far]))
    assert only_far.prior("lex", 2, 1) == pytest.approx([0.1, 0.9])  # fuera del conjunto de candidatas no cuenta
    empty = knn.CaseArrays(case([]))
    assert empty.prior("lex", 5, 4) == pytest.approx([0.5, 0.5])  # sin candidatas, uniforme


def test_ties_go_to_the_most_recent_wave():
    old = cand("old", 0.8, 0.8, 0.8, {"a": 1.0, "b": 0.0}, date="2024-01-01")
    new = cand("new", 0.8, 0.8, 0.8, {"a": 0.0, "b": 1.0}, date="2025-01-01")
    assert knn.CaseArrays(case([old, new])).prior("lex", 1, 1) == pytest.approx([0.0, 1.0])


def test_nested_grouped_chooses_with_the_other_groups_and_measures_on_the_left_out_one():
    # configuración 0 buena en los grupos A y B, mala en C; configuración 1 al revés
    matrix = np.array([[0.1, 0.1, 0.1, 0.1, 0.9, 0.9], [0.5, 0.5, 0.5, 0.5, 0.2, 0.2]])
    groups = ["A", "A", "B", "B", "C", "C"]
    held, chosen = knn.nested_grouped(matrix, groups)
    assert chosen[:4].tolist() == [0, 0, 0, 0] or chosen[:4].tolist() == [1, 1, 1, 1]
    assert held[:4].tolist() == matrix[chosen[:4], range(4)].tolist()
    # con C fuera, la mejor media de A y B es la 0: se mide en C con su error real (0,9)
    assert chosen[4] == 0 and held[4] == pytest.approx(0.9)


def test_paired_statistics():
    reference = np.linspace(0.10, 0.30, 12)
    better = reference - 0.02
    r = knn.paired(reference, better)
    assert r["better"] == 12 and r["worse"] == 0 and r["mean_delta"] == pytest.approx(-0.02)
    assert r["ci"][1] < 0 and r["wilcoxon_p"] < 0.01
    assert knn.paired(reference[:5], better[:5])["wilcoxon_p"] is None  # pocos casos


def test_reserved_ids_stop_the_experiment():
    reserved = {"CIS3557_P1"}
    data.assert_no_reserved(["CIS3475_P1"], "casos", reserved)
    with pytest.raises(AssertionError, match="reservados"):
        data.assert_no_reserved(["CIS3475_P1", "CIS3557_P1"], "casos", reserved)
