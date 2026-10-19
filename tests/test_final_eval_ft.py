import numpy as np
import pytest

from mavericks import final_eval, final_eval_ft as ft


def test_refuses_without_confirmation_or_when_done(tmp_path, monkeypatch):
    with pytest.raises(final_eval.FinalEvalError, match="confirm-final"):
        ft.guard(False)
    lock = tmp_path / "DONE"
    lock.write_text("ya")
    monkeypatch.setattr(ft, "LOCK", lock)
    with pytest.raises(final_eval.FinalEvalError, match="no se repite"):
        ft.guard(True)


def test_bootstrap_resamples_cases_when_few_studies():
    delta = np.array([-0.1, -0.2, 0.0, -0.15] * 5)
    low, high, unit = ft.bootstrap(delta, ["a", "b"] * 10, draws=500)
    assert unit == "casos" and low <= high < 0
    low, high, unit = ft.bootstrap(delta, [str(i % 10) for i in range(20)], draws=500)
    assert unit == "estudios"


def test_report_on_synthetic_cases():
    rng = np.random.default_rng(0)
    cases, stored, llm, b_probs = [], {"modes": {}, "base": {}, "mlp": {}}, {}, {}
    for i in range(24):
        n = 8 if i == 1 else 2 + i % 3
        options = [f"op{j}" for j in range(n)]
        obs = rng.dirichlet(np.ones(n))
        eid = f"CIS{i}_X"
        cases.append({"evidence_id": eid, "estudio_id": i % 9, "topic": "t", "options": options, "observed_distribution": dict(zip(options, obs, strict=True))})
        mode = "direct" if i % 3 == 0 else "none"
        stored["modes"][eid] = mode
        stored["base"][eid] = dict(zip(options, rng.dirichlet(np.ones(n)), strict=True))
        stored["mlp"][eid] = rng.dirichlet(np.ones(n))
        llm[eid] = dict(zip(options, rng.dirichlet(np.ones(n)), strict=True))
        b_probs[eid] = None if n > 7 else list(rng.dirichlet(np.ones(n)))
    text, summary, rows = ft.report(cases, stored, llm, b_probs, [0.3] * 24, [0.2] * 24, 7900, {"vram_gb": 2.6})
    assert "CUMPLE" in text and "Cobertura" in text and summary["casos"] == 24
    direct = [r for r in rows if r["mode"] == "direct"]
    assert all(np.allclose(r["ajustado"], r["rapid"]) for r in direct)  # con ancla directa el modo ajustado no cambia nada
    assert any(not r["supported"] for r in rows) and "más de 7 opciones" in text
    assert all(np.allclose(r["ajustado"], r["rapid"]) for r in rows if not r["supported"])  # sin respuesta de B: modo rápido
