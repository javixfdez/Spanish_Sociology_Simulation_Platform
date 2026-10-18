from pathlib import Path

import numpy as np
import pytest

from mavericks import final_eval


def test_final_evaluation_requires_confirmation_and_runs_only_once(tmp_path, monkeypatch):
    monkeypatch.setattr(final_eval, "OUT", tmp_path)
    monkeypatch.setattr(final_eval, "LOCK", tmp_path / "DONE")
    with pytest.raises(final_eval.FinalEvalError, match="confirm-final"):
        final_eval.guard(False)
    final_eval.guard(True)  # primera vez: permitido
    (tmp_path / "DONE").write_text("ejecutada 2026-10-19", encoding="utf-8")
    with pytest.raises(final_eval.FinalEvalError, match="ya se ejecutó"):
        final_eval.guard(True)
    (tmp_path / "DONE").unlink()
    with pytest.raises(final_eval.FinalEvalError, match="confirm-final"):
        final_eval.main(["priors"])  # sin --confirm-final: se niega antes de tocar nada


def test_blend_and_rows_follow_the_frozen_weights():
    cases = [{"evidence_id": "A", "estudio_id": 1, "topic": "salud", "options": ["Sí", "No"], "observed_distribution": {"Sí": 0.7, "No": 0.3}}]
    knn = {"A": {"Sí": 0.6, "No": 0.4}}
    mlp = {"A": np.array([0.4, 0.6])}
    llm = {"A": {"Sí": 0.9, "No": 0.1}}
    row = final_eval.build_rows(cases, knn, mlp, llm)[0]
    assert row["rapid"] == pytest.approx([0.5, 0.5])  # media 50/50
    assert row["with_llm"] == pytest.approx(0.3 * np.array([0.9, 0.1]) + 0.7 * np.array([0.5, 0.5]))
    assert "llm" not in final_eval.build_rows(cases, knn, mlp, None)[0]
    assert final_eval.blend({"a": 1.0, "b": 0.0}, {"a": 0.0, "b": 1.0}, 0.3) == pytest.approx({"a": 0.7, "b": 0.3})


def test_frozen_config_and_command_and_no_production_model():
    config = final_eval.frozen_config()
    assert config["llm_mode"]["w_llm"] == 0.3 and config["rapid_mode"]["w_mlp"] == 0.5 and config["mlp"]["pca_dims"] == 64
    assert "--cases outputs/ablation/cases_official_hard.json" in final_eval.llm_command()
    source = Path(final_eval.__file__).read_text(encoding="utf-8")
    assert "ProductionPrior" not in source and "prior_v2.production" not in source and "from .prior_v2 import production" not in source
