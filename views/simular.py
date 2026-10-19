import streamlit as st

from mavericks.examples import load_examples
from mavericks.ft.client import FtClient
from mavericks.schemas import Settings
from mavericks.service import (
    OutOfScopeError,
    ServiceError,
    assess_coverage,
    predict,
    suggest_options,
)
from mavericks.topics import TOPICS
from mavericks.ui_common import PRESET_HELP, PRESET_LABELS, get_resources, llm_service, setup, warm_embedder
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
    modes = ["rapido_sin_llm", "llm_completo", "ajustado"]  # siempre los tres: si el servidor cae, la selección no cambia en silencio, el resultado avisa del fallback
    preset = st.radio("Modo", modes, index=modes.index(defaults.default_preset), format_func=PRESET_LABELS.get, key="preset")
    st.caption(PRESET_HELP)
    if preset == "ajustado":
        if FtClient(timeout=1.0).healthy():
            st.caption("Servidor del modelo ajustado: activo. **Experimental**: menor error medido, no cumplió el criterio prerregistrado en el examen.")
        else:
            st.warning("El servidor del modelo ajustado no responde (`outputs\\ft\\run_server.ps1`): se usará el modo rápido y el resultado lo indicará.")
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
        warm_embedder()
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
                    result = predict(
                        question, options, "single", preset if (service or preset == "ajustado") else "rapido_sin_llm", topic=chosen_topic,
                        resources=resources, service=service, n_segments=segments,
                    )
                    if preset == "llm_completo" and not service and result["mode"] != "direct":  # Ollama no respondía: queda constancia en el resultado
                        result["producido_por"].update(solicitado=preset, fallback=True, motivo="Ollama no responde")
                    st.session_state["result"] = result
                st.session_state["text_service"] = service
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
    render(st.session_state["result"], llm=st.session_state.get("text_service"), timeout=Settings().explanation_timeout)
elif not analysis:
    st.info("Escribe una pregunta y pulsa **Analizar pregunta**, o abre un ejemplo en la barra lateral.")
