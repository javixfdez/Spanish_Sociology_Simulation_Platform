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


def test_learned_features_and_softmax():
    from mavericks.prior_v2 import learned

    rng = np.random.default_rng(0)
    q = rng.normal(size=8)
    o = rng.normal(size=(3, 8))
    qs, os_, ex = learned.raw_rows(q, o, [False, True, False])
    assert qs.shape == (3, 8) and os_.shape == (3, 8) and ex.shape == (3, 8)
    assert ex[:, 1].tolist() == [0.0, 0.5, 1.0]  # posición en la escala
    assert ex[:, 5].tolist() == [0.0, 1.0, 0.0]  # espontánea
    _, _, two = learned.raw_rows(q, o[:2], [False, False])
    assert two[:, 1].tolist() == [0.5, 0.5] and two[:, 7].tolist() == [1.0, 1.0]  # con 2 opciones la posición no informa
    p = learned.softmax([1.0, 2.0, 3.0])
    assert p.sum() == pytest.approx(1) and p.argmax() == 2
    assert learned.logit([0.5])[0] == pytest.approx(0.0)


def test_featurizer_is_fitted_only_with_training_data():
    from mavericks.prior_v2 import learned

    rng = np.random.default_rng(1)
    q, o = rng.normal(size=(40, 16)), rng.normal(size=(40, 16))
    extras = rng.normal(size=(40, 8))
    f = learned.Featurizer(4)
    x = f.fit(q[:10], o[:10], q, o, extras)
    assert x.shape == (40, 4 * 3 + 8)
    assert np.allclose(x.mean(axis=0), 0, atol=1e-6)
    new = f.transform(q[:3], o[:3], extras[:3])
    assert new.shape == (3, 20)


def test_study_bootstrap_resamples_studies_not_questions():
    from mavericks.prior_v2 import bigdev_eval as B

    delta = np.array([-0.02] * 30 + [0.05] * 30)  # un estudio muy a favor y otro muy en contra
    studies = ["A"] * 30 + ["B"] * 30
    low, high = B.study_bootstrap(delta, studies, draws=2000)
    assert low == pytest.approx(-0.02) and high == pytest.approx(0.05)  # con 2 estudios el IC es todo el rango
    many = [f"S{i}" for i in range(20) for _ in range(3)]
    low2, high2 = B.study_bootstrap(np.full(60, -0.01), many, draws=500)
    assert low2 == pytest.approx(-0.01) and high2 == pytest.approx(-0.01)


def test_mondrian_bins_merge_and_conformal_radius():
    from mavericks.prior_v2 import task3

    sims = np.concatenate([np.full(5, 0.1), np.full(40, 0.5), np.full(40, 0.9)])
    edges = task3.merge_small([0.3, 0.7], sims, minimum=15)
    assert len(edges) == 1  # el tramo de 5 preguntas se une al contiguo
    rng = np.random.default_rng(0)
    rows = [
        {"study": f"S{i % 5}", "best_sim": float(sims[i % len(sims)]), "n_options": 2,
         "observed": np.array([0.5, 0.5]) + rng.normal(0, 0.05, 2) * np.array([1, -1])}
        for i in range(120)
    ]
    predictor = lambda r: np.array([0.5, 0.5])  # noqa: E731
    stats, per_study = task3.leave_one_study_out(rows, predictor, 2)
    s = task3.summarize(stats)
    assert 0.8 <= s["global"]["coverage"] <= 1.0 and set(per_study) == {f"S{i}" for i in range(5)}
