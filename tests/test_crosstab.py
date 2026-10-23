"""Cruce con otra pregunta del mismo estudio: solo dato medido (microdatos ponderados), con n y aviso si n < 50."""

from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from mavericks import crosstab
from mavericks.schemas import Evidence


def evidence(uid, text, options, study="3500", source="CIS", qtype="single"):
    return Evidence(evidence_id=uid, source=source, study_id=study, question_text=text, options=options, metadata={"original_fields": {"question_type": qtype}})


EVIDENCE = [
    evidence("CIS3500_A", "¿Le preocupa el paro?", ["Sí", "No"]),
    evidence("CIS3500_B", "¿Vota habitualmente?", ["Siempre", "A veces", "Nunca", "N.S.", "N.C."]),
    evidence("CIS3500_C", "¿Qué problemas le preocupan? (varios)", ["Paro", "Vivienda"], qtype="multiple"),
    evidence("CIS3500_D", "¿Una sola opción?", ["Sí"]),
    evidence("CIS3501_B", "¿Vota habitualmente? (otro estudio)", ["Siempre", "Nunca"], study="3501"),
    evidence("EB1_X", "¿Confía en la UE?", ["Sí", "No"], study="3500", source="Eurobarómetro"),
]
ANCHOR = {"study_id": "3500", "evidence_id": "CIS3500_A", "mapping": {"Sí": "Sí", "No": "No"}}


def make_resources(n_each=60):
    """Dos preguntas contestadas por los mismos entrevistados: la A y la B (con algún N.S. en B que no cuenta)."""
    a_rows, b_rows, uid = [], [], 0
    plan = [("Sí", "Siempre", 2.0, n_each), ("Sí", "Nunca", 1.0, n_each), ("No", "A veces", 1.0, n_each // 3), ("No", "N.S.", 1.0, 10)]
    for a, b, weight, times in plan:
        for _ in range(times):
            uid += 1
            a_rows.append({"estudio_id": "3500", "respondent_uid": uid, "question_uid": "CIS3500_A", "answer": a, "peso": weight})
            b_rows.append({"estudio_id": "3500", "respondent_uid": uid, "question_uid": "CIS3500_B", "answer": b, "peso": weight})
    frames = {"CIS3500_A": pd.DataFrame(a_rows), "CIS3500_B": pd.DataFrame(b_rows)}
    return SimpleNamespace(evidence=SimpleNamespace(all=lambda: EVIDENCE), anchors=SimpleNamespace(_frame=lambda study, uid: frames[uid]))


def test_companions_are_other_single_cis_questions_of_the_same_study_with_substantive_options():
    found = crosstab.companions(make_resources(), "3500", "CIS3500_A")
    assert [c["evidence_id"] for c in found] == ["CIS3500_B"]  # sin la propia, sin multirrespuesta, sin una opción, sin otro estudio ni otras fuentes
    assert found[0]["options"] == ["Siempre", "A veces", "Nunca"]  # sin N.S./N.C.


def test_joint_table_is_weighted_counts_respondents_once_and_ignores_non_substantive_answers():
    resources = make_resources()
    other = crosstab.companions(resources, "3500", "CIS3500_A")[0]
    table = crosstab.joint(resources, ANCHOR, ["Sí", "No"], other)
    assert table["n"] == 60 + 60 + 20  # los 10 con N.S. en la otra pregunta no entran
    weighted = {("Sí", "Siempre"): 120, ("Sí", "Nunca"): 60, ("No", "A veces"): 20}
    total = sum(weighted.values())
    for (row, column), value in weighted.items():
        assert table["total"].loc[row, column] == pytest.approx(100 * value / total)
        assert table["counts"].loc[row, column] == {("Sí", "Siempre"): 60, ("Sí", "Nunca"): 60, ("No", "A veces"): 20}[(row, column)]
    assert table["by_row"].loc["Sí", "Siempre"] == pytest.approx(100 * 120 / 180) and table["by_row"].loc["Sí"].sum() == pytest.approx(100)
    assert table["total"].to_numpy().sum() == pytest.approx(100)
    assert not table["small"] and table["small_rows"] == ["No"]  # la fila «No» tiene 20 encuestados: aviso por fila


def test_a_small_joint_sample_is_flagged():
    resources = make_resources(n_each=12)
    other = crosstab.companions(resources, "3500", "CIS3500_A")[0]
    table = crosstab.joint(resources, ANCHOR, ["Sí", "No"], other)
    assert table["n"] == 12 + 12 + 4 and table["small"] and set(table["small_rows"]) == {"Sí", "No"}


def test_no_common_respondents_gives_an_empty_flagged_table():
    resources = make_resources()
    frames = {"CIS3500_A": pd.DataFrame({"estudio_id": ["3500"], "respondent_uid": [1], "answer": ["Sí"], "peso": [1.0]}),
              "CIS3500_B": pd.DataFrame({"estudio_id": ["3500"], "respondent_uid": [2], "answer": ["Siempre"], "peso": [1.0]})}
    resources.anchors = SimpleNamespace(_frame=lambda study, uid: frames[uid])
    table = crosstab.joint(resources, ANCHOR, ["Sí", "No"], crosstab.companions(resources, "3500", "CIS3500_A")[0])
    assert table["n"] == 0 and table["small"] and table["total"] is None


def _page(n_each):
    import streamlit as st

    from mavericks.ui_result import render_crosstab
    from tests.test_crosstab import ANCHOR, make_resources

    result = {"anchor": ANCHOR, "measured": {"x": 1}, "options": ["Sí", "No"]}
    render_crosstab(result, make_resources(n_each), key="t")
    st.write("fin")


@pytest.mark.parametrize("n_each", [60, 12])
def test_expander_shows_the_table_with_n_and_the_small_sample_warnings(n_each):
    at = AppTest.from_function(_page, args=(n_each,)).run()
    assert not at.exception and not at.dataframe  # hasta que se elige una pregunta no se calcula nada
    at.selectbox[0].select("CIS3500_B").run()
    assert not at.exception and len(at.dataframe) == 2
    text = " ".join(e.value for e in (*at.markdown, *at.warning))
    assert "encuestados con respuesta a las dos preguntas" in text
    if n_each == 60:
        assert "(n=60)" in at.dataframe[0].value.iloc[0, 0] and any("Filas con menos de 50 encuestados" in w.value and "«No»" in w.value for w in at.warning)
    else:
        assert any("poco fiable" in w.value for w in at.warning)


def test_nothing_is_shown_without_measured_data_or_resources():
    def page():
        import streamlit as st

        from mavericks.ui_result import render_crosstab

        render_crosstab({"anchor": None, "measured": None, "options": []}, object(), key="t")
        render_crosstab({"anchor": {"study_id": "1"}, "measured": {"x": 1}, "options": []}, None, key="u")
        st.write("fin")

    at = AppTest.from_function(page).run()
    assert not at.exception and not at.selectbox
