"""«¿Quizás querías decir…?»: solo sugiere, nunca cambia la pregunta ni ninguna cifra."""

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from mavericks import paraphrase
from mavericks.schemas import Evidence


class FakeEmbedder:
    """Embeddings de bolsa de palabras con hash (determinista, sin modelo): textos con las mismas palabras dan coseno 1."""

    name = "fake@cpu"

    def __init__(self):
        self.calls = 0

    def encode_queries(self, texts):
        self.calls += len(texts)
        out = np.zeros((len(texts), 256), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in re.findall(r"\w+", text.lower()):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
        return out


def record(evidence_id, text, options=("Sí", "No"), source="CIS", date="2025-01-01", qtype="single", study="3500"):
    return Evidence(
        evidence_id=evidence_id, source=source, study_id=study, question_text=text, options=list(options), fieldwork_date=date,
        metadata={"original_fields": {"question_type": qtype}},
    )


RECORDS = [
    record("CIS3500_ECOESP", "¿Cómo calificaría la situación económica general de España: muy buena, buena, mala o muy mala?", ("Muy buena", "Buena", "Mala", "Muy mala"), date="2025-03-01"),
    record("CIS3400_ECOESP", "¿Cómo calificaría la situación económica general de España: muy buena, buena, mala o muy mala?", ("Muy buena", "Buena", "Mala", "Muy mala"), date="2023-03-01", study="3400"),
    record("CIS3500_VIVIENDA", "¿Cree que el precio de la vivienda subirá durante el próximo año, se mantendrá o bajará?", ("Subirá", "Se mantendrá", "Bajará")),
    record("CIS3500_MULTI", "¿Qué problemas le preocupan más? Puede indicar varios", ("Paro", "Vivienda"), qtype="multiple"),
    record("CIS3500_UNA", "¿Está usted de acuerdo con la pregunta de una sola opción válida?", ("Sí",)),
    record("EB500_X", "¿Cómo calificaría la situación económica general de España: muy buena, buena, mala o muy mala?", ("Muy buena", "Buena"), source="Eurobarómetro"),
]


@pytest.fixture
def index(tmp_path):
    return paraphrase.ParaphraseIndex(RECORDS, FakeEmbedder(), cache_dir=tmp_path)


def test_index_keeps_only_cis_single_questions_with_two_options_and_the_latest_wave_of_each_variable(index):
    ids = {e.evidence_id for e in index.items}
    assert ids == {"CIS3500_ECOESP", "CIS3500_VIVIENDA"}  # sin multirrespuesta, sin una opción, sin otras fuentes y sin la oleada antigua


def test_a_reworded_question_gets_the_cis_question_with_its_study_and_options(index):
    got = paraphrase.suggest(None, "Situación económica general de España: ¿cómo la calificaría? muy buena, buena, mala o muy mala", index.embedder, threshold=0.8, index=index)
    assert got["evidence_id"] == "CIS3500_ECOESP" and got["study_id"] == "3500" and got["fieldwork_date"] == "2025-03-01"
    assert got["options"] == ["Muy buena", "Buena", "Mala", "Muy mala"] and 0.8 <= got["similarity"] <= 1


def test_an_unrelated_or_too_short_question_gets_nothing(index):
    assert paraphrase.suggest(None, "¿Con qué frecuencia usa herramientas de inteligencia artificial en el trabajo?", index.embedder, index=index) is None
    assert paraphrase.suggest(None, "¿Sí?", index.embedder, index=index) is None
    assert paraphrase.suggest(None, "", index.embedder, index=index) is None
    assert paraphrase.suggest(None, "cualquier pregunta larga", None, index=index) is None  # sin e5 no hay sugerencia


def test_the_calibrated_threshold_is_the_default(index):
    exact = RECORDS[2].question_text
    assert paraphrase.suggest(None, exact, index.embedder, index=index)["evidence_id"] == "CIS3500_VIVIENDA"
    assert paraphrase.THRESHOLD == 0.95


def test_vectors_are_cached_on_disk_and_reused(tmp_path):
    first = FakeEmbedder()
    paraphrase.ParaphraseIndex(RECORDS, first, cache_dir=tmp_path)
    assert first.calls == 2 and list(Path(tmp_path).glob("paraphrase_*.npz"))
    second = FakeEmbedder()
    again = paraphrase.ParaphraseIndex(RECORDS, second, cache_dir=tmp_path)
    assert second.calls == 0 and again.matrix.shape == (2, 256)


def _paraphrase_app():
    import streamlit as st

    from mavericks.ui_result import render_paraphrase

    if "start" not in st.session_state:
        st.session_state.update(start=True, q="mi pregunta", analysis={"question": "mi pregunta"})
    render_paraphrase({"question": "¿Cómo está de salud?", "study_id": "3551", "fieldwork_date": "2026-04-01", "similarity": 0.97, "options": ["Muy bueno", "Bueno"]})
    st.write("q=" + st.session_state["q"])
    st.write("analysis" if "analysis" in st.session_state else "sin_analysis")


def test_the_button_fills_the_text_and_asks_to_analyse_again():
    at = AppTest.from_function(_paraphrase_app).run()
    assert not at.exception
    text = " ".join(e.value for e in (*at.markdown,))
    assert "¿Quizás querías decir…?" in text and "estudio 3551" in text and "2026-04-01" in text and "97%" in text and "Muy bueno · Bueno" in text
    assert "q=mi pregunta" in text and "analysis" in text
    at.button[0].click().run()
    text = " ".join(e.value for e in at.markdown)
    assert "q=¿Cómo está de salud?" in text and "sin_analysis" in text  # el texto se rellena; el resultado y el análisis se descartan


CACHED = any(Path("outputs/cache").glob("paraphrase_*.npz"))


@pytest.mark.skipif(not CACHED, reason="necesita los datos reales y el índice ya calculado (abre la app una vez)")
def test_calibration_with_real_questions_has_perfect_precision_and_useful_coverage_at_the_chosen_threshold():
    from mavericks import paraphrase_eval
    from mavericks import service as svc

    index = paraphrase.ParaphraseIndex(svc.load_resources().evidence.all(), svc.get_embedder())
    positives, negatives = paraphrase_eval.build_sets(index)
    pos, neg = paraphrase_eval.evaluate(index, positives, negatives)
    row = next(r for r in paraphrase_eval.table(pos, neg) if r["umbral"] == paraphrase.THRESHOLD)
    assert row["incorrectas"] == 0 and row["negativas_con_sugerencia"] == 0 and row["cobertura"] >= 0.6
