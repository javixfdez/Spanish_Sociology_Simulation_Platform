import numpy as np
import pytest

from mavericks.sharpen import cross_validate, fit_by_mode, fit_tau, sharpen


def test_sharpen_is_identity_at_one_and_keeps_sum():
    d = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert sharpen(d, 1.0) == pytest.approx(d)
    sharp = sharpen(d, 2.0)
    assert sum(sharp.values()) == pytest.approx(1) and sharp["A"] > d["A"] > sharp["C"] * 2
    flat = sharpen(d, 0.5)
    assert flat["A"] < d["A"]


def _pairs(n=40, true_tau=2.0, seed=0):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        obs = rng.dirichlet([0.6, 0.6, 0.6])
        observed = dict(zip("ABC", obs))
        flat = sharpen(observed, 1 / true_tau)  # el modelo reparte de más
        pairs.append((flat, observed))
    return pairs


def test_fit_tau_recovers_underconfidence():
    assert fit_tau(_pairs(60)) == pytest.approx(2.0, abs=0.15)


def test_cross_validation_reports_out_of_sample_gain():
    result = cross_validate(_pairs(50))
    assert result["useful"] and result["tau"] > 1.5
    assert result["mae_cv"] < result["mae_raw"] * 0.5
    assert len(result["fold_taus"]) == 5 and len(result["corrected_cv"]) == 50


def test_no_correction_when_it_does_not_help_or_too_few_cases():
    rng = np.random.default_rng(3)
    # predicciones ya bien calibradas con ruido simétrico: no debe empeorar
    pairs = []
    for _ in range(40):
        obs = rng.dirichlet([2, 2, 2])
        noisy = np.clip(obs + rng.normal(0, 0.03, 3), 0.01, None)
        pairs.append((dict(zip("ABC", noisy / noisy.sum())), dict(zip("ABC", obs))))
    result = cross_validate(pairs)
    assert result["mae_cv"] <= result["mae_raw"] + 1e-9 or result["tau"] == 1.0
    assert cross_validate(_pairs(10)) is None


def test_fit_by_mode_skips_direct_and_small_groups():
    results = [
        {"status": "completed", "anchor_mode": "none", "evidence_id": f"n{i}", "predicted": p, "observed": o}
        for i, (p, o) in enumerate(_pairs(30))
    ]
    results += [
        {"status": "completed", "anchor_mode": "direct", "evidence_id": f"d{i}", "predicted": p, "observed": o}
        for i, (p, o) in enumerate(_pairs(30, seed=5))
    ]
    results += [
        {"status": "completed", "anchor_mode": "prior", "evidence_id": f"p{i}", "predicted": p, "observed": o}
        for i, (p, o) in enumerate(_pairs(5, seed=7))
    ]
    report = fit_by_mode(results)
    assert set(report) == {"prior", "none"}  # el modo directo nunca se ajusta
    assert report["none"]["useful"] and report["none"]["tau"] > 1.5
    assert report["prior"] == {"tau": 1.0, "n": 5, "useful": False}
