import pytest

from mavericks import assist
from mavericks.llm import LLMService, MockLLMProvider
from mavericks.schemas import ScopeJudgement, Settings
from mavericks.service import OutOfScopeError, Resources, predict


def hit(evidence_id, text, options, study="3546", topic="general", date="2026-03-06"):
    return {
        "retrieval_score": 1.0,
        "evidence": {
            "evidence_id": evidence_id, "study_id": study, "question_text": text, "options": options,
            "fieldwork_date": date, "topic": topic, "metadata": {"original_fields": {"question_type": "single"}},
        },
    }


class StubEvidence:
    def __init__(self, hits):
        self.hits = hits

    def search(self, *args, **kwargs):
        return self.hits


def resources(hits):
    return Resources(None, StubEvidence(hits), None, {}, None)


ECO = "¿Cómo calificaría Ud. la situación económica general de España: muy buena, buena, mala o muy mala?"
ECO_OPTIONS = ["Muy buena", "Buena", "(NO LEER) Regular", "Mala", "Muy mala", "N.S.", "N.C."]


class FakeLLM:
    """Servicio mínimo: devuelve lo que se le indique para cada esquema."""

    def __init__(self, **answers):
        self.answers = answers

    async def call(self, schema, payload, validator=None, **kwargs):
        value = self.answers[schema.__name__]
        if isinstance(value, Exception):
            raise value
        if validator:
            validator(value)
        return value


def test_a_direct_uses_official_options_without_ns_nc():
    out = assist.suggest_options(ECO, resources([hit("CIS1_ECO", ECO, ECO_OPTIONS)]))
    assert out["source"] == "cis_directo"
    assert out["options"] == ["Muy buena", "Buena", "Regular", "Mala", "Muy mala"]
    assert "3546" in out["explanation"]


def test_b_prior_uses_scale_of_similar_question():
    similar = "¿Cómo calificaría Ud. su situación económica personal en la actualidad: muy buena, buena, mala o muy mala?"
    out = assist.suggest_options(
        "¿Cómo calificaría hoy la situación económica de su hogar: muy buena, buena, regular, mala o muy mala?",
        resources([hit("CIS1_ECOPER", similar, ECO_OPTIONS)]),
        topic="general",
    )
    assert out["source"] == "cis_parecida" and "Mala" in out["options"]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("¿Está Ud. a favor o en contra de limitar por ley el precio del alquiler?", ["A favor", "En contra"]),
        ("¿Ha usado alguna vez un patinete eléctrico?", ["Sí", "No"]),
        ("¿Cree que la inteligencia artificial mejorará la sanidad?", ["Sí", "No"]),
    ],
)
def test_c_closed_questions(question, expected):
    assert assist.suggest_options(question, resources([]))["options"] == expected


@pytest.mark.parametrize(
    "question,kind",
    [
        ("¿Está de acuerdo con prohibir la venta de coches de combustión?", "acuerdo"),
        ("¿Le preocupa que la inteligencia artificial le quite su empleo?", "preocupacion"),
        ("¿Con qué frecuencia se siente solo/a?", "frecuencia"),
        ("¿Cómo valora la gestión del Gobierno en vivienda?", "valoracion"),
    ],
)
def test_d_without_llm_uses_standard_scale(question, kind):
    out = assist.suggest_options(question, resources([]))
    assert out["source"] == "escala_estandar" and out["options"] == assist.STANDARD_SCALES[kind]
    assert not any("n.s" in o.lower() or "n.c" in o.lower() for o in out["options"])


def test_d_with_mock_and_failing_model():
    llm = LLMService(MockLLMProvider(), Settings(llm_provider="mock", llm_retries=0), "outputs/cache_test_assist")
    out = assist.suggest_options("¿Le preocupa que la inteligencia artificial le quite su empleo?", resources([]), llm)
    assert out["source"] == "modelo" and 2 <= len(out["options"]) <= 6
    failing = FakeLLM(OptionSuggestion=RuntimeError("sin modelo"))
    out = assist.suggest_options("¿Con qué frecuencia se siente solo/a?", resources([]), failing)
    assert out["source"] == "escala_estandar" and out["warnings"]


def test_prior_guard_rejects_false_analogy():
    fragment = hit("CIS1_P12_1_5", "Otra herramienta de Inteligencia Artificial", ["Sí", "No", "N.S.", "N.C."], topic="ia_tecnologia")
    question = "¿Utiliza habitualmente herramientas de inteligencia artificial como ChatGPT?"
    ref = assist.find_reference(resources([fragment]), question, "ia_tecnologia")
    assert ref["mode"] == "none"
    paraphrase = hit("CIS1_P11", "¿Utiliza Ud. herramientas de inteligencia artificial como ChatGPT en su trabajo o estudios?", ["Sí", "No"], topic="ia_tecnologia")
    assert assist.find_reference(resources([paraphrase]), question, "ia_tecnologia")["mode"] == "prior"


@pytest.mark.parametrize(
    "question,kind",
    [
        ("¿Cuántos habitantes tiene Madrid?", "hechos"),
        ("¿Cuál es la capital de Francia?", "hechos"),
        ("¿Quién ganará las próximas elecciones generales?", "prediccion"),
        ("¿Es feliz mi vecino?", "persona_concreta"),
        ("¿Qué opinan los franceses de la reforma de las pensiones?", "poblacion_distinta"),
        ("Traduce esta frase al inglés", "no_es_pregunta"),
    ],
)
def test_scope_rules_out_of_scope(question, kind):
    out = assist.scope_check(question)
    assert not out["in_scope"] and out["kind"] == kind and out["reformulation"]


@pytest.mark.parametrize(
    "question",
    ["¿Cree que el precio de la vivienda subirá el año que viene?", "¿Está de acuerdo con subir los impuestos?"],
)
def test_scope_keeps_opinion_questions(question):
    assert assist.scope_check(question)["in_scope"]


def test_assess_coverage_categories_and_llm_judgement():
    direct = assist.assess_coverage(ECO, resources([hit("CIS1_ECO", ECO, ECO_OPTIONS)]))
    assert direct["category"] == "respaldada" and direct["reference"]["study_id"] == "3546"
    none = assist.assess_coverage("¿Le gusta el queso?", resources([]))
    assert none["category"] == "sin_datos"
    out = assist.assess_coverage("¿Cuántos habitantes tiene Madrid?", resources([]))
    assert out["category"] == "fuera_de_alcance" and out["reformulation"]
    judge = FakeLLM(ScopeJudgement=ScopeJudgement(category="hechos", reformulation="¿Le parece grande Madrid?"))
    llm_out = assist.assess_coverage("Madrid en cifras", resources([]), judge)
    assert llm_out["category"] == "fuera_de_alcance" and "Madrid" in llm_out["reformulation"]
    mock = LLMService(MockLLMProvider(), Settings(llm_provider="mock", llm_retries=0), "outputs/cache_test_assist")
    assert assist.assess_coverage("¿Le gusta el queso?", resources([]), mock)["category"] == "sin_datos"


def test_predict_refuses_out_of_scope_questions():
    with pytest.raises(OutOfScopeError) as error:
        predict("¿Cuántos habitantes tiene Madrid?", ["Mucho", "Poco"], resources=resources([]))
    assert error.value.coverage["category"] == "fuera_de_alcance"


def test_clean_labels_and_spontaneous_flag():
    assert assist.display_label("(NO LEER) Regular") == "Regular"
    assert assist.spontaneous_labels(["Buena", "(NO LEER) Regular", "N.S."]) == ["Regular"]
    out = assist.suggest_options(ECO, resources([hit("CIS1_ECO", ECO, ECO_OPTIONS)]))
    assert out["spontaneous"] == ["Regular"] and "(NO LEER)" not in " ".join(out["options"])


def test_latest_wave_of_the_same_variable_is_the_reference():
    short = hit("CIS3517_ECOESP", "Valoración de la situación económica general de España", ECO_OPTIONS, date="2025-07-07")
    long = hit("CIS3546_ECOESP", ECO, ECO_OPTIONS, date="2026-03-06")
    ref = assist.find_reference(resources([short, long]), "Valoración de la situación económica general de España")
    assert ref["evidence_id"] == "CIS3546_ECOESP"
