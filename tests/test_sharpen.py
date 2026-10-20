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


def _results(n=40, mode="none", seed=0):
    out = []
    for i, (pred, obs) in enumerate(_pairs(n, seed=seed)):
        sub_obs = {"sexo": {"mujer": {"distribution": dict(obs), "n": 300}}}
        out.append(
            {
                "status": "completed", "anchor_mode": mode, "evidence_id": f"{mode}{i}", "topic": "general",
                "predicted": pred, "observed": obs, "mae": float(np.mean([abs(pred[k] - obs[k]) for k in obs])),
                "mae_uniform_baseline": 0.3, "option_errors": [abs(pred[k] - obs[k]) for k in obs],
                "predicted_subgroups": {"sexo": {"mujer": dict(pred)}}, "subgroup_option_errors": [], "_obs_sub": sub_obs,
            }
        )
    return out


def test_apply_to_results_recomputes_errors_with_cv_taus_and_keeps_raw():
    from mavericks.sharpen import apply_to_results

    results = _results(40)
    cases = [{"evidence_id": r["evidence_id"], "observed_subgroups": r.pop("_obs_sub")} for r in results]
    report = fit_by_mode(results)
    fixed = apply_to_results(results, report, cases)
    assert np.mean([r["mae"] for r in fixed]) < np.mean([r["mae"] for r in results]) * 0.5
    assert fixed[0]["predicted_raw"] == results[0]["predicted"] and fixed[0]["sharpen_tau"] > 1.5
    assert fixed[0]["subgroup_mae"] is not None and fixed[0]["subgroup_option_errors"]
    assert results[0]["predicted"] != fixed[0]["predicted"]  # la lista original no se muta


def test_direct_mode_results_are_never_sharpened():
    from mavericks.sharpen import apply_to_results

    results = _results(30, mode="direct")
    for r in results:
        r.pop("_obs_sub")
    assert apply_to_results(results, fit_by_mode(results), []) == results


def test_interval_coverage_is_out_of_sample():
    from mavericks.sharpen import interval_coverage

    rng = np.random.default_rng(0)
    results = [
        {"status": "completed", "anchor_mode": "none", "option_errors": list(np.abs(rng.normal(0, 0.05, 3)))}
        for _ in range(100)
    ]
    cov = interval_coverage(results)["none"]
    assert 0.8 <= cov["coverage"] <= 0.97 and cov["target"] == 0.9
    assert interval_coverage(results[:5])["none"]["coverage"] is None


def test_graph_applies_sharpening_only_with_flag_and_keeps_raw(tmp_path):
    import asyncio

    from mavericks.evidence import EvidenceRepository
    from mavericks.graph import simulate
    from mavericks.llm import LLMService, MockLLMProvider
    from mavericks.population import PopulationLoader, PopulationRepository, create_demo
    from mavericks.schemas import Question, Settings

    path = create_demo(tmp_path / "fx")
    frame, meta = PopulationLoader().load(path)
    repo = PopulationRepository(frame, meta)
    evidence = EvidenceRepository(tmp_path / "e.sqlite")
    calibration = {"sharpening": {"none": {"tau": 2.0}}}

    def run(sharpen_flag):
        llm = LLMService(MockLLMProvider(), Settings(llm_provider="mock", llm_retries=0, sharpen=sharpen_flag), tmp_path / f"c{sharpen_flag}")
        return asyncio.run(simulate(Question(question="¿Usas aplicaciones raras?", options=["Sí", "No", "Quizá"]), repo, evidence, llm, 3, calibration=calibration))

    off, on = run(False), run(True)
    assert "national_distribution_raw" not in off and off["national_distribution"] != on["national_distribution"]
    assert on["national_distribution_raw"] == off["national_distribution"]
    assert on["sharpening"] == {"tau": 2.0, "mode": "none"}
    assert sum(on["national_distribution"].values()) == pytest.approx(1)
    top = max(off["national_distribution"], key=off["national_distribution"].get)
    assert on["national_distribution"][top] > off["national_distribution"][top]
