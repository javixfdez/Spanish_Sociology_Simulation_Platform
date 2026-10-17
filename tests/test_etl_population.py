from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mavericks.etl.population import (
    group_index,
    income_tramo,
    ipf,
    municipality_size_class,
    parse_period,
    systematic_sample,
)
from mavericks.population import PopulationLoader


def test_parse_period_rejects_dates_after_cutoff():
    assert str(parse_period("1 de enero de 2026")) == "2026-01-01"
    with pytest.raises(ValueError):
        parse_period("1 de julio de 2026")


def test_group_and_income_and_municipality_bins():
    assert list(group_index([17, 18, 24, 25, 74, 75, 99])) == [-1, 0, 0, 1, 5, 6, 6]
    assert list(income_tramo([6000, 4000, 3000, 2000, 1500, 500])) == [0, 1, 2, 3, 4, 5]
    assert list(municipality_size_class([20_000, 60_000, 500_000, 3_000_000])) == [2, 3, 5, 6]


def test_ipf_matches_margins():
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, 3, 5000), rng.integers(0, 2, 5000)
    margins = [
        {"index": a, "target": np.array([100.0, 300.0, 600.0])},
        {"index": b, "target": np.array([400.0, 600.0])},
    ]
    w, report = ipf(np.ones(5000), margins)
    assert np.allclose(np.bincount(a, w), [100, 300, 600], rtol=1e-4)
    assert np.allclose(np.bincount(b, w), [400, 600], rtol=1e-4)
    assert report["max_relative_gap"] < 1e-4


def test_ipf_reports_empty_cells():
    idx = np.array([0, 0, 1])
    _, report = ipf(np.ones(3), [{"index": idx, "target": np.array([2.0, 1.0, 5.0]), "name": "m"}])
    assert report["empty_cells"][0] == {"margin": "m", "n": 1, "target_lost": 5.0}


def test_systematic_sample_is_reproducible_and_proportional():
    weights = np.array([1.0, 1.0, 8.0])
    first = systematic_sample(weights, 1000, np.random.default_rng(3))
    again = systematic_sample(weights, 1000, np.random.default_rng(3))
    assert (first == again).all()
    assert abs(np.mean(first == 2) - 0.8) < 0.01


@pytest.mark.skipif(
    not Path("data/processed/synthetic_population.parquet").exists(), reason="requiere etl-population"
)
def test_generated_population_meets_contract():
    frame, metadata = PopulationLoader().load("data/processed/synthetic_population.parquet")
    assert metadata.categories and not metadata.synthetic
    assert frame.peso.nunique() == 1 and (frame.edad >= 18).all()
    assert isinstance(frame, pd.DataFrame) and frame.person_id.is_unique
