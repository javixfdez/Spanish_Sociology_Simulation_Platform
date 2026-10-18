from mavericks.evidence import EvidenceRepository, source_family
from mavericks.report import compare, comparison_table, report


def _r(i, mode, mae, topic="general", nat=None, sec=100.0):
    return {
        "evidence_id": f"c{i}", "status": "completed", "anchor_mode": mode, "topic": topic, "mae": mae,
        "mae_uniform_baseline": 0.2, "mae_national_baseline": nat, "subgroup_mae": mae + 0.01,
        "execution_time": sec, "option_errors": [mae, mae],
    }


def test_report_separates_modes_and_topics_and_baselines():
    results = [_r(i, "direct", 0.02, nat=0.018) for i in range(3)] + [_r(10 + i, "none", 0.15, "consumo", nat=0.16) for i in range(4)]
    results.append({**_r(99, "none", 0.0), "status": "failed", "mae": None})
    text = report(results, "Prueba")
    assert "directo (pregunta repetida)" in text and "sin ancla (pregunta nueva)" in text
    assert "prior (pregunta parecida)" not in text.split("## Por tema")[0].split("## Por modo de ancla")[1]  # sin casos: no aparece
    assert "| consumo |" in text and "directo · general" in text and "sin · consumo" in text or "sin ancla" in text
    assert "1 fallidos" in text and "0.150" in text and "0.160" in text


def test_compare_counts_better_worse_and_ties_on_the_same_cases():
    base = [_r(i, "none", 0.20) for i in range(6)] + [_r(50, "direct", 0.01)]
    other = [_r(0, "none", 0.10), _r(1, "none", 0.15), _r(2, "none", 0.201), _r(3, "none", 0.30), _r(4, "none", 0.20), _r(50, "direct", 0.5)]
    result = compare(base, {"hibrido": other})["hibrido"]
    assert result["cases"] == 5 and result["better"] == 2 and result["worse"] == 1 and result["ties"] == 2  # el directo no cuenta
    assert result["mae"] == (0.10 + 0.15 + 0.201 + 0.30 + 0.20) / 5
    assert "hibrido" in comparison_table({"hibrido": result})


def test_sources_filter_toggles_external_evidence(tmp_path):
    import json

    records = [
        dict(evidence_id="C1", source="CIS 3546", question_text="vivienda alquiler", options=["Sí", "No"]),
        dict(evidence_id="E1", source="eurobarometro", question_text="vivienda alquiler housing", options=["Yes", "No"]),
    ]
    path = tmp_path / "s.json"
    path.write_text(json.dumps(records), encoding="utf-8")
    repo = EvidenceRepository(tmp_path / "s.sqlite")
    repo.ingest(path)
    def ids(**kw):
        return {h["evidence"]["evidence_id"] for h in repo.search("vivienda alquiler", limit=5, **kw)}

    assert ids() == {"C1", "E1"} and ids(sources={"cis"}) == {"C1"} and ids(sources={"eurobarometro", "ess"}) == {"E1"}
    assert source_family(repo.all()[0]) in {"cis", "eurobarometro"}


def test_simbench_and_entropy_ratio():
    from mavericks.report import entropy_ratio, simbench

    exact = {"status": "completed", "predicted": {"A": 0.7, "B": 0.3}, "observed": {"A": 0.7, "B": 0.3}}
    uniform = {"status": "completed", "predicted": {"A": 0.5, "B": 0.5}, "observed": {"A": 0.9, "B": 0.1}}
    assert simbench([exact]) == 100.0 and abs(simbench([uniform])) < 1e-9
    flat = {"status": "completed", "predicted": {"A": 0.5, "B": 0.5}, "observed": {"A": 0.9, "B": 0.1}}
    assert entropy_ratio([flat]) > 1.5  # reparte de más
    assert entropy_ratio([exact]) == 1.0 and simbench([]) is None
