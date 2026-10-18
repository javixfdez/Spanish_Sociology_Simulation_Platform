import numpy as np
import pytest

from mavericks import fastcal


def make_rows(n_direct=30, n_prior=3, n_none=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for mode, n, noise in (("direct", n_direct, 0.02), ("prior", n_prior, 0.1), ("none", n_none, 0.18)):
        for i in range(n):
            observed = rng.dirichlet(np.ones(4))
            predicted = np.clip(observed + rng.normal(0, noise, 4), 0.01, None)
            predicted = predicted / predicted.sum()
            rows.append(
                {
                    "evidence_id": f"{mode}{i}", "topic": "general", "set": "test", "mode": mode, "similarity": 0.9,
                    "n_options": 4, "predicted": predicted.tolist(), "observed": observed.tolist(),
                    "mae": float(np.mean(np.abs(predicted - observed))), "mae_uniform": 0.2,
                }
            )
    return rows


def test_prior_with_few_cases_is_combined_with_none_and_ranges_are_ordered():
    model = fastcal.fit(make_rows())
    assert model["combined"] == ["prior", "none"]
    direct, _ = fastcal.radius(model, "direct", 4, 0.3)
    none, group = fastcal.radius(model, "none", 4, 0.3)
    prior, prior_group = fastcal.radius(model, "prior", 4, 0.3)
    assert direct < none and prior == pytest.approx(none) and prior_group["key"] == "prior+none"
    core, _ = fastcal.radius(model, "none", 4, 0.3, level="q50")
    assert core < none


def test_ranges_shrink_for_extreme_probabilities():
    model = fastcal.fit(make_rows())
    assert fastcal.radius(model, "none", 4, 0.02)[0] < fastcal.radius(model, "none", 4, 0.5)[0]


def test_cross_validated_coverage_is_close_to_nominal():
    cv = fastcal.cross_validate(make_rows(n_direct=60, n_none=60))
    assert 0.8 <= cv["global"]["coverage"] <= 0.97
    assert cv["direct"]["mean_width_pp"] < cv["none"]["mean_width_pp"]


def test_prior_has_its_own_range_with_enough_cases():
    model = fastcal.fit(make_rows(n_prior=20))
    assert "prior" not in model["combined"]
    assert fastcal.radius(model, "prior", 4, 0.3)[1]["key"] == "prior"
