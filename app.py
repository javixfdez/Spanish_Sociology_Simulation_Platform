import asyncio
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from mavericks.anchor import load_anchor_store
from mavericks.evidence import EvidenceRepository
from mavericks.graph import simulate
from mavericks.llm import LLMService, create_provider
from mavericks.population import PopulationLoader, PopulationRepository
from mavericks.schemas import Question, Settings
from mavericks.statistics import load_calibration

DEFAULT_POPULATION = (
    "data/processed/synthetic_population.parquet"
    if Path("data/processed/synthetic_population.parquet").exists()
    else "data/fixtures/population.csv"
)


@st.cache_resource(show_spinner=False)
def load_population(path):
    frame, metadata = PopulationLoader().load(path)
    return PopulationRepository(frame, metadata)


@st.cache_resource(show_spinner=False)
def load_anchors(directory):
    return load_anchor_store(Settings(), directory)


st.set_page_config(page_title="GenAI Mavericks", layout="wide")
st.title("GenAI Mavericks · Simulación poblacional")
st.caption("Estimaciones de modelo, no encuestas reales. Sin calibración histórica no se atribuye precisión.")
question = st.text_input("Pregunta", "¿Pagarías por una aplicación con inteligencia artificial?")
options_text = st.text_area("Una opción por línea", "Sí\nNo\nNo estoy seguro")
question_type = st.radio(
    "Tipo de respuesta", ["single", "multiple"], horizontal=True, format_func=lambda v: {"single": "Única", "multiple": "Múltiple"}[v]
)
left, right = st.columns(2)
with left:
    provider = st.selectbox("Proveedor", ["mock", "ollama", "external"])
    model = st.text_input("Modelo Ollama", "qwen3:8b")
    segments = st.slider("Máximo de segmentos", 1, 80, 5)
    runs = st.slider("Repeticiones por segmento (estabilidad)", 1, 5, 1)
with right:
    population = st.text_input("Archivo de población", DEFAULT_POPULATION)
    evidence_db = st.text_input("Base de evidencia", "data/processed/evidence.sqlite")
    cis_dir = st.text_input("Carpeta de microdatos del CIS procesados", "data/processed")
    calibration_path = st.text_input("Calibración (JSON)", "data/processed/calibration.json")
rag = st.checkbox("Usar evidencia", value=True)
use_anchor = st.checkbox("Anclar a las distribuciones reales del CIS", value=True)
if st.button("Ejecutar"):
    try:
        settings = Settings(llm_provider=provider, ollama_model=model, stability_runs=runs)
        repository = load_population(population)
        anchors = load_anchors(cis_dir) if use_anchor else None
        with st.spinner("Ejecutando agentes…"):
            result = asyncio.run(
                simulate(
                    Question(
                        question=question,
                        options=[o.strip() for o in options_text.splitlines() if o.strip()],
                        question_type=question_type,
                    ),
                    repository,
                    EvidenceRepository(evidence_db),
                    LLMService(create_provider(settings), settings),
                    segments,
                    rag,
                    anchors=anchors,
                    calibration=load_calibration(calibration_path),
                )
            )
        st.session_state["result"] = result
    except Exception as exc:
        st.error(str(exc))
if "result" in st.session_state:
    result = st.session_state["result"]
    if result["execution_metadata"]["synthetic"]:
        st.warning("DEMO / SYNTHETIC — sin validez sociológica ni representatividad")
    for warning in result["warnings"]:
        st.warning(warning)
    if result.get("errors"):
        st.error("Simulación incompleta: no se publica distribución nacional")
        st.json(result["errors"])
    else:
        trust = result["trust_metrics"]
        intervals = (trust.get("predictive_intervals") or {}).get("intervals", {})
        table = pd.DataFrame({"estimación %": pd.Series(result["national_distribution"]) * 100})
        if intervals:
            table["mínimo %"] = [intervals[o][0] * 100 for o in table.index]
            table["máximo %"] = [intervals[o][1] * 100 for o in table.index]
        st.bar_chart(table["estimación %"])
        st.dataframe(table.round(1))
        st.info(trust["reliability"])
        if result["subgroup_distributions"]:
            with st.expander("Subgrupos"):
                for dimension, groups in result["subgroup_distributions"].items():
                    st.write(f"**{dimension}** (resolución {result['subgroup_resolution'][dimension]:.0%})")
                    st.dataframe((pd.DataFrame(groups).T * 100).round(1))
        st.dataframe(
            pd.DataFrame({k: p["probabilities"] for k, p in result["segment_predictions"].items()}).T
        )
    st.write(result.get("final_report", {}).get("narrative", {}).get("summary", ""))
    with st.expander("Evidencia y evaluación"):
        st.json(result["retrieved_evidence"])
        st.json(result["evidence_assessment"])
    with st.expander("Indicadores de confianza"):
        st.json(result["trust_metrics"])
    st.download_button(
        "Descargar JSON",
        json.dumps(result, ensure_ascii=False, indent=2),
        "simulation.json",
        mime="application/json",
    )
