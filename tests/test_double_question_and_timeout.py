"""Aviso de preguntas dobles (no es un rechazo) y aviso del modo ajustado cuando el servidor no contesta a tiempo."""

import pytest

from mavericks import assist
from mavericks.ft.client import FtError, FtTimeout
from mavericks.service import GPU_BUSY_HINT, predict
from tests.test_ft_adjusted import OPTIONS, QUESTION, FakeFt, demo  # noqa: F401  (demo es una fixture)

DOUBLE = [
    "¿Le gusta su trabajo y cuánto cobra?",
    "¿Le gusta su trabajo y por qué?",
    "¿Cuántas personas viven en su hogar y cómo se llaman?",
    "¿Cree que sube el paro y qué hará el Gobierno?",
    "¿Le preocupa la salud? ¿Y la vivienda?",
    "¿Fuma?, ¿bebe? ¿Hace deporte?",
    "Le gusta su trabajo? Cuanto cobra?",
]
SINGLE = [
    "¿Está de acuerdo con subir impuestos y gastar más en sanidad?",  # dudosa: dos acciones bajo una sola interrogación, no se avisa (criterio documentado)
    "¿Cree que el paro sube y que los salarios bajan?",  # «y que» sin tilde es una conjunción
    "¿Qué opina de los hombres y las mujeres en política?",
    "¿Le preocupa que la inteligencia artificial le quite su empleo?",
    "¿Cuántos hijos tiene y vive con ellos?",  # «y vive»: no es una segunda interrogativa
    "¿Hay cuántos niños en casa?",  # «hay cuántos» contiene «y cuántos» dentro de una palabra
    "",
]


@pytest.mark.parametrize("question", DOUBLE)
def test_double_questions_get_a_warning(question):
    assert assist.double_question_warning(question) == assist.DOUBLE_QUESTION_NOTE


@pytest.mark.parametrize("question", SINGLE)
def test_single_questions_get_no_warning(question):
    assert assist.double_question_warning(question) is None


def test_warning_text_is_the_agreed_one():
    assert assist.DOUBLE_QUESTION_NOTE == "Parece que hay dos preguntas en una; el resultado mezclará ambas. Mejor hazlas por separado."


def test_predict_warns_but_does_not_change_the_numbers(demo):  # noqa: F811
    plain = predict(QUESTION, OPTIONS, preset="ajustado", resources=demo, ft=FakeFt())
    double_text = QUESTION[:-1] + " y cuánto le preocupa?"
    double = predict(double_text, OPTIONS, preset="ajustado", resources=demo, ft=FakeFt())
    assert assist.DOUBLE_QUESTION_NOTE in double["warnings"] and assist.DOUBLE_QUESTION_NOTE not in plain["warnings"]
    assert double["national"]["distribution"] == pytest.approx(plain["national"]["distribution"])  # el aviso no es un rechazo ni cambia cifras


def test_suggest_options_shows_the_warning_before_simulating(demo):  # noqa: F811
    out = assist.suggest_options("¿Le gusta su trabajo y cuánto cobra?", demo)
    assert assist.DOUBLE_QUESTION_NOTE in out["warnings"]
    assert assist.DOUBLE_QUESTION_NOTE not in assist.suggest_options("¿Le gusta su trabajo?", demo)["warnings"]


class SlowFt(FakeFt):
    def predict_batch(self, items):
        raise FtTimeout("el servidor del modelo ajustado no respondió (TimeoutError: timed out)")


def test_timeout_fallback_tells_the_user_to_free_the_gpu(demo):  # noqa: F811
    out = predict(QUESTION, OPTIONS, preset="ajustado", resources=demo, ft=SlowFt())
    assert out["fallback"] is True and GPU_BUSY_HINT in " ".join(out["warnings"])
    assert "ciérralo y vuelve a probar" in out["producido_por"]["motivo"] and "Ollama" in out["producido_por"]["motivo"]
    assert out["producido_por"]["motivo"].startswith("el servidor del modelo ajustado no respondió")


def test_other_failures_keep_the_plain_message(demo):  # noqa: F811
    out = predict(QUESTION, OPTIONS, preset="ajustado", resources=demo, ft=FakeFt(fail=True))
    assert "Ollama" not in " ".join(out["warnings"]) and "Ollama" not in out["producido_por"]["motivo"]


def test_client_raises_the_timeout_subclass_only_on_timeouts(monkeypatch):
    import urllib.request

    from mavericks.ft.client import FtClient

    def slow(*args, **kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(urllib.request, "urlopen", slow)
    with pytest.raises(FtTimeout):
        FtClient(timeout=0.1).predict(QUESTION, OPTIONS)

    def refused(*args, **kwargs):
        raise ConnectionRefusedError("rechazada")

    monkeypatch.setattr(urllib.request, "urlopen", refused)
    with pytest.raises(FtError) as caught:
        FtClient(timeout=0.1).predict(QUESTION, OPTIONS)
    assert not isinstance(caught.value, FtTimeout)
