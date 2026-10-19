import json
from pathlib import Path

import pytest

from mavericks import service as svc

REFERENCE = Path("tests/data/prior_sin_ancla_3cb71f0.json")
REAL = Path("data/processed/cis_responses.parquet").exists() and Path("data/processed_v2/evidence_v2.sqlite").exists() and Path("outputs/models/prior_mlp.pkl").exists()


@pytest.mark.skipif(not REAL, reason="necesita los datos reales")
def test_prior_of_questions_without_anchor_is_identical_to_the_frozen_official_configuration():
    """Blindaje: el reparto nacional de preguntas sin ancla (incluida C2-28, que cambió cuando la tolerancia a N.R./N.S./N.R. se aplicó también al prior) es idéntico, a 1e-9,
    al calculado con 3cb71f0 (la configuración congelada de la evaluación oficial). Los valores de referencia se calcularon con 3cb71f0 en un worktree aparte."""
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    assert len(reference) == 6 and "C2-28" in reference
    resources = svc.load_resources()
    for case_id, ref in reference.items():
        out = svc.predict(ref["pregunta"], ref["opciones"], preset="rapido_sin_llm", resources=resources)
        assert out["mode"] == ref["modo"], case_id
        got = list(out["national"]["distribution"].values())
        assert len(got) == len(ref["nacional"]) and max(abs(a - b) for a, b in zip(got, ref["nacional"], strict=True)) <= 1e-9, case_id
