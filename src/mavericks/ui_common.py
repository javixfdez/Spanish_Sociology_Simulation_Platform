"""Piezas comunes de la interfaz Streamlit: recursos en caché, estilo y avisos de error."""

import time
from pathlib import Path

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
ROOT = Path(__file__).resolve().parents[2]          # carpeta raíz del proyecto
LOGO = ROOT / "customization" / "logo.svg"
TEAM = "Miguel Alsina · Itziar Carro · Javier Fernández · Javier Picazo · Alejandro Useros"

CSS = """
<style>
.stApp::before {content:""; position:fixed; top:0; left:0; right:0; height:4px; z-index:1000002;
  background:linear-gradient(90deg,#A100FF 0%,#7C3AED 50%,#A100FF 100%);}
header[data-testid="stHeader"] {background:rgba(255,255,255,.95); backdrop-filter:blur(8px);
  border-bottom:1px solid #E9D5FF;}
.block-container {padding-top:3.5rem; padding-bottom:3.5rem; max-width:1200px;}

h1, h2, h3 {color:#3B0764; font-weight:800; letter-spacing:-.01em;}
header[data-testid="stHeader"]::after {
  content:"GenAI Mavericks 2026";
  position:absolute; left:50%; top:50%; transform:translate(-50%,-50%);
  padding:3px 14px; border-radius:999px; background:#F3E8FF; border:1px solid #D8B4FE;
  color:#581C87; font-size:.78rem; font-weight:700; letter-spacing:.04em; pointer-events:none;
}

.card {background:#fff; border:1px solid #E9D5FF; border-radius:16px; padding:14px 18px;
  margin-bottom:10px; box-shadow:0 1px 2px rgba(59,7,100,.06);}
.pill {display:inline-block; padding:2px 12px; border-radius:999px; font-weight:600; color:#fff; font-size:.9rem;}
.pill.alta {background:#047857;} .pill.media {background:#B45309;} .pill.baja {background:#BE123C;}  /* texto blanco: contraste ≥ 5 (AA) */
[data-testid="stCaptionContainer"] {opacity:.85;}  /* los textos pequeños de apoyo (rangos, métodos) se leen sobre el fondo morado */
.muted {color:#4B5563; font-size:.9rem;}

[data-testid="stMetric"] {background:#fff; border:1px solid #E9D5FF; border-radius:12px; padding:12px 16px;}
[data-testid="stMetricValue"] {color:#3B0764; font-weight:900;}

.brand-footer {position:fixed; bottom:0; left:0; right:0; z-index:99; background:#fff;
  border-top:1px solid #E9D5FF; padding:6px 24px; font-size:.75rem; color:#64748b;
  display:flex; justify-content:space-between;}
.brand-footer b {color:#7e22ce;}
body:has([data-testid="stSidebar"][data-collapsed="false"]) .brand-footer {left:300px;}  /* con la barra lateral abierta (300 px) no tapa los nombres */
</style>
"""

FOOTER_HTML = f"""
<div class='brand-footer'><span>{TEAM}</span>
<span>GenAI Mavericks Accenture</span></div>
"""


def configure():
    """Configuración global de la aplicación (solo la llama el punto de entrada ``app.py``)."""
    st.set_page_config(page_title="Synthetic Spain · Accenture GenAI Mavericks",
                       page_icon=":material/chevron_right:", layout="wide")
    if LOGO.exists():
        st.logo(str(LOGO), size="large")
    st.markdown(CSS, unsafe_allow_html=True)


def setup(title=None, icon=None):
    """Estilo de cada página (la configuración global se hace en ``app.py``)."""
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

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
