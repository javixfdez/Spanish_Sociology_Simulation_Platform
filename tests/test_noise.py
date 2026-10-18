import numpy as np
import pytest

from mavericks.noise import bootstrap_floor, summarize


def _sample(n, p=0.3, seed=0):
    rng = np.random.default_rng(seed)
    answers = np.where(rng.random(n) < p, "A", "B")
    return answers, np.ones(n)


def test_floor_shrinks_with_the_square_root_of_n():
    rng = np.random.default_rng(1)
    small = bootstrap_floor(*_sample(100), ["A", "B"], rng, draws=300)
    large = bootstrap_floor(*_sample(1600), ["A", "B"], rng, draws=300)
    assert large < small and large == pytest.approx(small / 4, rel=0.35)  # n×16 -> error/4


def test_floor_is_zero_without_variability_and_none_without_answers():
    rng = np.random.default_rng(0)
    assert bootstrap_floor(np.array(["A"] * 50), np.ones(50), ["A", "B"], rng, draws=50) == 0.0
    assert bootstrap_floor(np.array(["Z"] * 5), np.ones(5), ["A", "B"], rng, draws=10) is None


def test_floor_respects_weights():
    rng = np.random.default_rng(2)
    answers = np.array(["A"] * 50 + ["B"] * 50)
    uniform = bootstrap_floor(answers, np.ones(100), ["A", "B"], rng, draws=300)
    skewed = bootstrap_floor(answers, np.r_[np.full(50, 20.0), np.ones(50)], ["A", "B"], rng, draws=300)
    assert skewed < uniform  # la masa se concentra en A: menos incertidumbre en la proporción


def test_summarize_ignores_missing_values():
    rows = [{"evidence_id": "a", "floor": 0.02, "flat": 0.1, "llm": None}, {"evidence_id": "b", "floor": 0.04, "flat": 0.3, "llm": 0.2}]
    out = summarize(rows, ["flat", "llm"])
    assert out["floor"] == pytest.approx(0.03) and out["flat"] == pytest.approx(0.2) and out["llm"] == 0.2 and out["cases"] == 2
