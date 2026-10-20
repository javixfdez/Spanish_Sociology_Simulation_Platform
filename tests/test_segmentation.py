
import numpy as np
import pandas as pd
import pytest

from mavericks.population import PopulationMetadata, PopulationRepository, build_segments, resolve_dimensions
from mavericks.schemas import Prediction
from mavericks.statistics import aggregate, subgroup_distributions, subgroup_resolution

AGES = ["18-24", "25-34", "35-44", "45-54", "55-64", "65-74", "75+"]
EDU = ["sin_estudios", "primaria", "secundaria_1", "secundaria_2", "fp", "superiores"]
CCAA = ["Andalucía", "Madrid", "Cataluña", "Galicia"]


@pytest.fixture(scope="module")
def repo():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6000):
        rows.append(
            dict(
                person_id=f"P{i}",
                peso=1.0,
                edad=30,
                grupo_edad=rng.choice(AGES),
                sexo=rng.choice(["hombre", "mujer"]),
                nivel_estudios=rng.choice(EDU, p=[0.05, 0.1, 0.25, 0.15, 0.1, 0.35]),
                ccaa=rng.choice(CCAA, p=[0.4, 0.3, 0.2, 0.1]),
                ingresos_hogar_tramo=rng.choice(["bajo", "medio", "alto"]),
                version_poblacion="v1",
            )
        )
    return PopulationRepository(pd.DataFrame(rows), PopulationMetadata(population_version="v1"))


@pytest.mark.parametrize("k", [1, 5, 12, 20, 40])
def test_segment_count_matches_request_with_real_categories(repo, k):
    segments = build_segments(repo, k)
    assert len(segments) == k  # antes: --segments 5 devolvía 1 segmento y 12 devolvía 7
    assert sum(s["population_weight"] for s in segments) == pytest.approx(repo.frame.peso.sum())
    assert sum(s["n_synthetic"] for s in segments) == len(repo.frame)
    assert min(s["n_synthetic"] for s in segments) >= 10


def test_uses_dimensions_beyond_age_sex_education(repo):
    used = {d for s in build_segments(repo, 80, dimensions=["ccaa", "ingresos_hogar_tramo"]) for d in s["demographic_profile"]}
    assert {"ccaa", "ingresos_hogar_tramo"} <= used


def test_requested_dimensions_go_first_and_aliases_work(repo):
    assert resolve_dimensions(["ingresos", "Comunidad autónoma", "inventada"], repo.frame)[:2] == [
        "ingresos_hogar_tramo",
        "ccaa",
    ]
    segments = build_segments(repo, 3, dimensions=["ingresos"])
    assert {tuple(s["demographic_profile"]) for s in segments} == {("ingresos_hogar_tramo",)}


def test_small_cells_are_grouped_not_dropped(repo):
    segments = build_segments(repo, 4, dimensions=["grupo_edad"])
    merged = [s for s in segments if s["merged_categories"]]
    assert merged, "las categorías pequeñas deben agruparse"
    for s in merged:
        assert s["demographic_profile"]["grupo_edad"].startswith("Otros")
        assert "grupo_edad" not in s["homogeneous_dimensions"]
    assert sum(s["population_weight"] for s in segments) == pytest.approx(repo.frame.peso.sum())


def test_segments_are_deterministic(repo):
    first = build_segments(repo, 20)
    second = build_segments(repo, 20)
    assert [s["demographic_profile"] for s in first] == [s["demographic_profile"] for s in second]
    assert [s["representative_profiles"] for s in first] == [s["representative_profiles"] for s in second]


def _predictions(segments, options, key):
    out = {}
    for s in segments:
        p = 0.8 if s["demographic_profile"].get(key) == "mujer" else 0.2
        out[s["segment_id"]] = Prediction(
            segment_id=s["segment_id"], probabilities={options[0]: p, options[1]: 1 - p}, explanation="x"
        )
    return out


def test_subgroups_use_composition_and_hide_unresolved_dimensions(repo):
    options = ["Sí", "No"]
    segments = build_segments(repo, 14, dimensions=["sexo", "grupo_edad"])
    predictions = _predictions(segments, options, "sexo")
    resolution = subgroup_resolution(segments)
    assert resolution["sexo"] == 1.0 and "ccaa" not in resolution
    groups = subgroup_distributions(segments, predictions, options)
    assert groups["sexo"]["mujer"]["Sí"] == pytest.approx(0.8)
    assert groups["sexo"]["hombre"]["Sí"] == pytest.approx(0.2)
    assert "ccaa" not in groups and "ingresos_hogar_tramo" not in groups
    national = aggregate(segments, predictions, options)
    w = {c: repo.frame[repo.frame.sexo == c].peso.sum() / repo.frame.peso.sum() for c in ["mujer", "hombre"]}
    assert sum(w[c] * groups["sexo"][c]["Sí"] for c in w) == pytest.approx(national["Sí"])


def test_unsplit_dimension_published_only_above_min_resolution(repo):
    options = ["Sí", "No"]
    segments = build_segments(repo, 6, dimensions=["sexo"])
    low = subgroup_distributions(segments, _predictions(segments, options, "sexo"), options, min_resolution=1.01)
    assert low == {}
