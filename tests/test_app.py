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
    at = run("app.py")
    assert at.sidebar.selectbox[1].options[0] == "—"
    at.sidebar.selectbox[1].select(at.sidebar.selectbox[1].options[2]).run()
    assert not at.exception
    assert at.session_state["result"]["options"]


def test_simular_fast_run():
    at = run("app.py")
    at.sidebar.selectbox[0].select("rapido_sin_llm")
    at.text_area(key="q").set_value("¿Le preocupa que la inteligencia artificial le quite su empleo?")
    at.radio(key="template").set_value("Escala Mucho – Nada").run()
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["result"]["mode_label"] in {"directo", "prior", "sin ancla"}
    assert not at.error


def test_bad_options_message():
    at = run("app.py")
    at.sidebar.selectbox[0].select("rapido_sin_llm")
    at.text_area(key="q").set_value("¿Le preocupa la inteligencia artificial?")
    at.radio(key="template").set_value("Personalizada").run()
    at.text_area(key="opts").set_value("Solo una")
    at.button[0].click().run()
    assert at.error and "opciones" in at.error[0].value


@pytest.mark.parametrize("page", ["1_Lote", "2_Validacion", "3_Metodologia", "4_Poblacion"])
def test_other_pages(page):
    run(f"pages/{page}.py")
