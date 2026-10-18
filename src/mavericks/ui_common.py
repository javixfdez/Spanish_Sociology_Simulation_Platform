"""Piezas comunes de la interfaz Streamlit: recursos en caché, estilo y avisos de error."""

import httpx
import streamlit as st

from .llm import LLMService, check_ollama, create_provider
from .schemas import Settings
from .service import ServiceError, load_resources

PRESET_LABELS = {
    "recomendado": "Recomendado (estadística + modelo de lenguaje si está disponible)",
    "rapido_sin_llm": "Rápido, sin modelo de lenguaje (instantáneo, sin GPU)",
    "llm_completo": "Modelo de lenguaje completo (necesita GPU y Ollama)",
}
CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.card {border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;}
.pill {display:inline-block; padding: 2px 12px; border-radius: 999px; font-weight: 600; color: #fff; font-size: .9rem;}
.pill.alta {background:#1e8449;} .pill.media {background:#d68910;} .pill.baja {background:#c0392b;}
.muted {color: rgba(128,128,128,.95); font-size: .9rem;}
</style>
"""


def configure():
    """Configuración global de la aplicación (solo la llama el punto de entrada ``app.py``)."""
    st.set_page_config(page_title="Simulación sociológica", page_icon="📊", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)


def setup(title=None, icon=None):
    """Estilo de cada página (la configuración global se hace en ``app.py``)."""
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data(ttl=30, show_spinner=False)
def ollama_ready(base_url, model, timeout=2.0):
    """¿Responde Ollama en menos de ``timeout`` segundos y tiene el modelo? Comprobación rápida, sin generar nada."""
    try:
        response = httpx.get(base_url.rstrip("/") + "/api/tags", timeout=timeout)
        response.raise_for_status()
        return model in [m["name"] for m in response.json()["models"]]
    except (httpx.HTTPError, ValueError, KeyError):
        return False


@st.cache_resource(show_spinner="Cargando población y datos del CIS…")
def cached_resources():
    return load_resources()


def get_resources():
    """Recursos o un mensaje de error comprensible (y se detiene la página)."""
    try:
        return cached_resources()
    except ServiceError as exc:
        st.error(str(exc))
        st.stop()


def llm_service(preset, model="qwen3:8b"):
    """Devuelve (servicio o None, aviso). Solo se conecta a Ollama si el preset lo necesita."""
    if preset == "rapido_sin_llm":
        return None, None
    settings = Settings(llm_provider="ollama", ollama_model=model)
    try:
        check_ollama(settings)
    except RuntimeError:
        if preset == "llm_completo":
            raise ServiceError(
                "El modelo de lenguaje no está disponible. Arranca Ollama o elige el preset «Rápido, sin modelo de lenguaje»."
            ) from None
        return None, "El modelo de lenguaje no está disponible: se ha usado el modo rápido, basado solo en datos del CIS."
    return LLMService(create_provider(settings), settings), None
