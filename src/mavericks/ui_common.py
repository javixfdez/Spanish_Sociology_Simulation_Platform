"""Piezas comunes de la interfaz Streamlit: recursos en caché, estilo y avisos de error."""

import time

import httpx
import streamlit as st

from .llm import LLMService, check_ollama, create_provider
from .schemas import Settings
from .service import ServiceError, get_embedder, load_resources, predict

PRESET_LABELS = {"rapido_sin_llm": "Rápido", "llm_completo": "Con LLM", "ajustado": "Ajustado"}
PRESET_HELP = (
    "**Rápido**: estimación estadística validada, al instante. "
    "**Con LLM**: añade el criterio del modelo de lenguaje y una explicación redactada; tarda más. "
    "**Ajustado** (si el servidor está arrancado): un modelo pequeño ajustado con encuestas reales da el reparto de las preguntas sin equivalente en el CIS."
)
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


WARMUP_QUESTION = ("¿Cree Ud. que el precio de la vivienda durante el próximo año subirá, se mantendrá o bajará?", ["Subirá", "Se mantendrá", "Bajará"])


def warm_up(resources):
    """Carga e5 y el MLP y ejecuta una pregunta de prueba para construir los índices: la primera pregunta real tarda como las demás."""
    started = time.monotonic()
    try:
        get_embedder()
        predict(WARMUP_QUESTION[0], WARMUP_QUESTION[1], resources=resources)
    except Exception:  # sin e5 o sin modelo la aplicación sigue con el KNN; el aviso lo da cada resultado
        pass
    return time.monotonic() - started


@st.cache_resource(show_spinner=False)
def cached_resources():
    """Recursos + calentamiento, una vez por proceso. Devuelve (recursos, segundos de calentamiento)."""
    started = time.monotonic()
    resources = load_resources()
    seconds = warm_up(resources)
    return resources, seconds, time.monotonic() - started


def preload():
    """Se llama al cargar la página principal: carga todo mientras se muestra «Cargando modelos…»."""
    try:
        with st.spinner("Cargando modelos…"):
            return cached_resources()
    except ServiceError:
        return None


@st.cache_resource(show_spinner="Cargando el modelo de lenguaje e5 (solo la primera vez)…")
def warm_embedder():
    """Carga e5 una sola vez por proceso (singleton de service.py); devuelve el modelo o None si no está disponible."""
    return get_embedder()


def get_resources():
    """Recursos o un mensaje de error comprensible (y se detiene la página)."""
    try:
        return cached_resources()[0]
    except ServiceError as exc:
        st.error(str(exc))
        st.stop()


def llm_service(preset, model="qwen3:8b"):
    """(servicio o None, aviso). Solo se conecta a Ollama en el modo con LLM; si no responde, se usa el modo rápido."""
    if preset == "rapido_sin_llm":
        return None, None
    settings = Settings(llm_provider="ollama", ollama_model=model)
    try:
        check_ollama(settings)
    except RuntimeError:
        if preset == "ajustado":
            return None, "El modelo de lenguaje no responde: la explicación será la automática."
        return None, "El modelo de lenguaje no responde: se muestra el resultado del modo rápido, con explicación automática."
    return LLMService(create_provider(settings), settings), None
