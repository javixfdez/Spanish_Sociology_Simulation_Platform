import pytest

from mavericks.evidence import EvidenceRepository
from mavericks.ft.client import FtClient, FtError, _checked
from mavericks.population import PopulationLoader, PopulationRepository, create_demo
from mavericks.service import Resources, predict

OPTIONS = ["Mucho", "Bastante", "Poco", "Nada"]
QUESTION = "¿Le preocupa que la inteligencia artificial le quite su empleo?"
INTERVALS = {"groups": {"2 opciones": {"q90": 0.8, "q50": 0.3, "preguntas": 10}, "3 o más": {"q90": 0.45, "q50": 0.14, "preguntas": 10}}, "coverage": {}, "calibrada_con": "test"}
VALIDATION = {"error_tipico": {"2 opciones": {"mae": 0.16, "preguntas": 10, "basis": "prueba"}, "3 o más": {"mae": 0.09, "preguntas": 10, "basis": "prueba"}}}


class FakeFt:
    """Servidor simulado: reparto fijo por número de opciones y registro de las peticiones."""

    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def predict_batch(self, items):
        if self.fail:
            raise FtError("el servidor del modelo ajustado no respondió (simulado)")
        self.calls.append(items)
        out = []
        for it in items:
            n = len(it["options"])
            out.append([0.7] + [0.3 / (n - 1)] * (n - 1))
        return out, 0.1


@pytest.fixture
def demo(tmp_path):
    path = create_demo(tmp_path / "fixtures")
    frame, metadata = PopulationLoader().load(path)
    evidence = EvidenceRepository(tmp_path / "evidence.sqlite")
    evidence.ingest(path.parent / "evidence.json")
    return Resources(PopulationRepository(frame, metadata), evidence, None, {}, None, ft_intervals=INTERVALS, ft_validation=VALIDATION)


def test_adjusted_uses_fine_tuned_numbers_and_own_intervals(demo):
    ft = FakeFt()
    out = predict(QUESTION, OPTIONS, preset="ajustado", resources=demo, ft=ft)
    assert out["mode"] != "direct" and out["fallback"] is False
    assert out["national"]["distribution"]["Mucho"] == pytest.approx(0.7)
    assert out["prior_usado"]["metodo"] == "modelo_ajustado"
    assert out["national"]["interval_info"]["calibrated_on"] == "test"
    assert out["ajustado"]["seconds"] == pytest.approx(0.1)
    first = ft.calls[0][0]  # la primera petición es la nacional, con el perfil del entrenamiento y las etiquetas de usuario
    assert first["profile"] == "Población: adultos residentes en España" and first["options"] == OPTIONS


def test_adjusted_falls_back_to_rapid_and_says_so(demo):
    out = predict(QUESTION, OPTIONS, preset="ajustado", resources=demo, ft=FakeFt(fail=True))
    assert out["fallback"] is True and out["ajustado"] is None
    assert out["prior_usado"].get("metodo") != "modelo_ajustado"
    assert sum(out["national"]["distribution"].values()) == pytest.approx(1)
    assert any("modelo ajustado no respondió" in w for w in out["warnings"])


def test_other_presets_unchanged(demo):
    out = predict(QUESTION, OPTIONS, preset="rapido_sin_llm", resources=demo)
    assert out["fallback"] is False and out["ajustado"] is None


def test_client_validates_and_reports_unreachable():
    assert _checked([0.5, 0.5], 2) == [0.5, 0.5]
    for bad in ([0.5], [0.7, 0.7], [float("nan"), 1.0]):
        with pytest.raises(FtError):
            _checked(bad, 2)
    client = FtClient("http://127.0.0.1:9", timeout=0.5)
    assert client.healthy() is False
    with pytest.raises(FtError):
        client.predict("pregunta", ["Sí", "No"])


def test_more_than_seven_options_uses_rapid(demo):
    ft = FakeFt()
    out = predict(QUESTION, [f"opción {i}" for i in range(8)], preset="ajustado", resources=demo, ft=ft)
    assert out["fallback"] is True and not ft.calls
    assert any("2 a 7 opciones" in w for w in out["warnings"])


def test_result_says_which_mode_produced_it_and_whether_there_was_fallback(demo):
    ok = predict(QUESTION, OPTIONS, preset="ajustado", resources=demo, ft=FakeFt())["producido_por"]
    assert ok["modo"] == "ajustado" and not ok["fallback"] and "no cumplió el criterio prerregistrado" in ok["experimental"]
    bad = predict(QUESTION, OPTIONS, preset="ajustado", resources=demo, ft=FakeFt(fail=True))["producido_por"]
    assert bad["modo"] == "rapido" and bad["fallback"] and "no respondió" in bad["motivo"] and bad["experimental"] is None
    plain = predict(QUESTION, OPTIONS, preset="rapido_sin_llm", resources=demo)["producido_por"]
    assert plain["modo"] == "rapido" and not plain["fallback"]


def test_client_retries_once_on_reset(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self, *a):
            return b'{"ok": true}'

    def flaky(request, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionResetError("simulado")
        return Response()

    monkeypatch.setattr("mavericks.ft.client.urllib.request.urlopen", flaky)
    monkeypatch.setattr("mavericks.ft.client.time.sleep", lambda s: None)
    assert FtClient("http://x").healthy() is True and len(calls) == 2
    calls.clear()
    monkeypatch.setattr("mavericks.ft.client.urllib.request.urlopen", lambda r, timeout=None: (_ for _ in ()).throw(ConnectionResetError("siempre")))
    assert FtClient("http://x").healthy() is False


def test_banner_shows_mode_fallback_and_experimental_note():
    from streamlit.testing.v1 import AppTest

    def app():
        import streamlit as st

        from mavericks.ui_result import render_produced_by

        render_produced_by(st.session_state["r"])

    ok = {"modo": "ajustado", "etiqueta": "Ajustado (experimental)", "solicitado": "ajustado", "fallback": False, "motivo": None, "nota": None,
          "experimental": "Experimental: menor error medido, no cumplió el criterio prerregistrado en el examen."}
    at = AppTest.from_function(app)
    at.session_state["r"] = {"producido_por": ok}
    at.run()
    assert "Ajustado (experimental)" in at.markdown[0].value and "no cumplió el criterio prerregistrado" in at.info[0].value and not at.warning
    bad = {**ok, "modo": "rapido", "etiqueta": "Rápido", "fallback": True, "motivo": "el servidor del modelo ajustado no respondió", "experimental": None}
    at.session_state["r"] = {"producido_por": bad}
    at.run()
    assert "Rápido" in at.markdown[0].value and "solicitado: Ajustado" in at.markdown[0].value
    assert "Se ha vuelto al modo rápido: el servidor del modelo ajustado no respondió" in at.warning[0].value
