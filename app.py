import asyncio

import pandas as pd
import streamlit as st

from mavericks.evidence import EvidenceRepository
from mavericks.graph import simulate
from mavericks.llm import LLMService, create_provider
from mavericks.population import PopulationLoader, PopulationRepository
from mavericks.schemas import Question, Settings

st.set_page_config(page_title="GenAI Mavericks", layout="wide")
st.title("GenAI Mavericks · Simulación poblacional")
st.caption("Estimaciones de modelo, no encuestas reales. Sin calibración histórica no se atribuye precisión.")
question = st.text_input("Pregunta", "¿Pagarías por una aplicación con inteligencia artificial?")
options_text = st.text_area("Una opción por línea", "Sí\nNo\nNo estoy seguro")
provider = st.selectbox("Proveedor", ["mock", "ollama", "external"])
model = st.text_input("Modelo Ollama", "qwen3:8b")
population = st.text_input("Archivo de población", "data/fixtures/population.csv")
segments = st.slider("Máximo de segmentos", 1, 30, 5)
rag = st.checkbox("Usar evidencia", value=True)
if st.button("Ejecutar"):
    try:
        settings = Settings(llm_provider=provider, ollama_model=model)
        frame, metadata = PopulationLoader().load(population)
        with st.spinner("Ejecutando agentes…"):
            result = asyncio.run(
                simulate(
                    Question(question=question, options=options_text.splitlines()),
                    PopulationRepository(frame, metadata),
                    EvidenceRepository(),
                    LLMService(create_provider(settings), settings),
                    segments,
                    rag,
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
        st.bar_chart(pd.Series(result["national_distribution"]) * 100)
        st.dataframe(
            pd.DataFrame({k: p["probabilities"] for k, p in result["segment_predictions"].items()}).T
        )
    st.write(result.get("final_report", {}).get("narrative", {}).get("summary", ""))
    with st.expander("Evidencia y evaluación"):
        st.json(result["retrieved_evidence"])
        st.json(result["evidence_assessment"])
    with st.expander("Indicadores de confianza"):
        st.json(result["trust_metrics"])
    import json

    st.download_button(
        "Descargar JSON",
        json.dumps(result, ensure_ascii=False, indent=2),
        "simulation.json",
        mime="application/json",
    )
