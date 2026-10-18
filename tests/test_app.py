from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

REAL = (
    Path("data/processed/synthetic_population.parquet").exists()
    and Path("data/processed_v2/evidence_v2.sqlite").exists()
)
pytestmark = pytest.mark.skipif(not REAL, reason="requiere los datos procesados")


ROOT = Path(__file__).resolve().parent.parent


def run(path):
    at = AppTest.from_file(str(ROOT / path), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_simular_page_and_example():
    at = run("views/simular.py")
    assert at.sidebar.selectbox[1].options[0] == "—"
    at.sidebar.selectbox[1].select(at.sidebar.selectbox[1].options[2]).run()
    assert not at.exception
    assert at.session_state["result"]["options"]


def analyse(question):
    at = run("views/simular.py")
    at.sidebar.selectbox[0].select("rapido_sin_llm")
    at.text_area(key="q").set_value(question)
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_simular_flow_suggests_options_and_simulates():
    at = analyse("¿Le preocupa que la inteligencia artificial le quite su empleo?")
    analysis = at.session_state["analysis"]
    assert analysis["suggestion"]["options"] and analysis["coverage"]["category"] != "fuera_de_alcance"
    assert at.text_area(key="opts").value
    at.button[1].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["result"]["mode_label"] in {"directo", "prior", "sin ancla"}
    assert not at.error


def test_out_of_scope_question_is_not_simulated():
    at = analyse("¿Cuántos habitantes tiene Madrid?")
    assert at.session_state["analysis"]["coverage"]["category"] == "fuera_de_alcance"
    assert len(at.button) == 1 and "result" not in at.session_state


def test_bad_options_message():
    at = analyse("¿Está Ud. a favor o en contra de limitar el alquiler?")
    at.text_area(key="opts").set_value("Solo una")
    at.button[1].click().run()
    assert at.error and "opciones" in at.error[0].value


@pytest.mark.parametrize("page", ["lote", "validacion", "metodologia", "poblacion"])
def test_other_pages(page):
    run(f"views/{page}.py")
