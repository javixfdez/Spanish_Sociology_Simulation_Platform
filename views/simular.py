import streamlit as st

from mavericks.examples import load_examples
from mavericks.schemas import Settings
from mavericks.service import (
    OutOfScopeError,
    ServiceError,
    assess_coverage,
    predict,
    suggest_options,
)
from mavericks.topics import TOPICS
from mavericks.ui_common import PRESET_LABELS, get_resources, llm_service, ollama_ready, setup
from mavericks.ui_result import render, render_coverage

setup()

AUTO = "(automático)"
examples = load_examples()
by_label = {item["question"][:90]: item for item in examples["items"]}

st.title("Simulación de opinión pública en España")
st.markdown(
    "Escribe una pregunta de encuesta y estima cómo respondería la población adulta residente en España, "
    "con un intervalo de confianza y el detalle por grupos. **Son estimaciones, no una encuesta real.**"
)


def pick_example():
    item = by_label.get(st.session_state.get("example"))
    if item:
        st.session_state.update(
            q=item["question"],
            opts="\n".join(item["options"]),
            result=item,
            analysis={"question": item["question"], "coverage": item["coverage"], "suggestion": item["suggestion"]},
            topic_select=item["topic"],
        )


with st.sidebar:
    st.header("Opciones")
    defaults = Settings()
    ready = ollama_ready(defaults.ollama_base_url, defaults.ollama_model)
    preset = st.selectbox(
        "Modo de cálculo",
        ["rapido_sin_llm", "recomendado", "llm_completo"],
        index=1 if ready else 0,
        format_func=PRESET_LABELS.get,
        key="preset",
    )
    if not ready:
        st.caption("El modelo de lenguaje no responde: se usa el modo rápido.")
    if examples["items"]:
        st.selectbox("Ejemplos listos", ["—", *by_label], key="example", on_change=pick_example)
        st.caption("Los ejemplos se muestran al instante porque ya están calculados.")
    with st.expander("Avanzado"):
        topic = st.selectbox("Tema (si no se indica, se detecta)", [AUTO, *TOPICS], key="topic_select")
        model = st.text_input("Modelo de lenguaje", "qwen3:8b")
        segments = st.slider("Grupos de población simulados", 5, 40, 10)

question = st.text_area(
    "Pregunta", key="q", placeholder="¿Le preocupa que la inteligencia artificial le quite su empleo?", height=90
)
chosen_topic = None if topic == AUTO else topic

if st.button("Analizar pregunta", type="primary"):
    st.session_state.pop("result", None)
    try:
        resources = get_resources()
        service, notice = llm_service(preset, model)
        if notice:
            st.info(notice)
        with st.spinner("Analizando la pregunta…"):
            coverage = assess_coverage(question, resources, service, chosen_topic)
            suggestion = None
            if coverage["category"] != "fuera_de_alcance":
                suggestion = suggest_options(question, resources, service, chosen_topic)
                st.session_state["opts"] = "\n".join(suggestion["options"])
        st.session_state["analysis"] = {"question": question, "coverage": coverage, "suggestion": suggestion}
    except ServiceError as exc:
        st.session_state.pop("analysis", None)
        st.error(str(exc))
    except Exception:
        st.session_state.pop("analysis", None)
        st.error("No se pudo analizar la pregunta. Prueba con el modo rápido o revisa el texto.")

analysis = st.session_state.get("analysis")
if analysis and analysis["question"].strip() != question.strip():
    analysis = None  # la pregunta ha cambiado: hay que volver a analizarla
if analysis:
    render_coverage(analysis["coverage"])
    if analysis["suggestion"]:
        st.markdown("**Opciones sugeridas; puedes cambiarlas** (una por línea)")
        st.caption(analysis["suggestion"]["explanation"])
        options_text = st.text_area("Opciones de respuesta", key="opts", height=130, label_visibility="collapsed")
        if analysis["suggestion"].get("spontaneous"):
            names = ", ".join(f"«{o}»" for o in analysis["suggestion"]["spontaneous"])
            st.caption(f"{names}: opción no leída por el entrevistador en el CIS.")
        for warning in analysis["suggestion"].get("warnings", []):
            st.info(warning)
        if st.button("Simular", type="primary"):
            options = [o.strip() for o in options_text.splitlines() if o.strip()]
            try:
                resources = get_resources()
                service, notice = llm_service(preset, model)
                if notice:
                    st.info(notice)
                with st.spinner("Calculando…"):
                    st.session_state["result"] = predict(
                        question, options, "single", preset, topic=chosen_topic, resources=resources,
                        service=service, n_segments=segments,
                    )
            except OutOfScopeError as exc:
                st.session_state.pop("result", None)
                render_coverage(exc.coverage)
            except ServiceError as exc:
                st.session_state.pop("result", None)
                st.error(str(exc))
            except Exception:
                st.session_state.pop("result", None)
                st.error("No se pudo completar la simulación. Prueba con el modo rápido o revisa las opciones.")

if "result" in st.session_state and analysis:
    render(st.session_state["result"])
elif not analysis:
    st.info("Escribe una pregunta y pulsa **Analizar pregunta**, o abre un ejemplo en la barra lateral.")
