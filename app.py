import streamlit as st

from mavericks.examples import load_examples
from mavericks.service import PRESETS, ServiceError, predict
from mavericks.topics import TOPICS
from mavericks.ui_common import PRESET_LABELS, get_resources, llm_service, setup
from mavericks.ui_result import render

setup("Simular", "📊")

TEMPLATES = {
    "Sí / No": "Sí\nNo",
    "Escala Mucho – Nada": "Mucho\nBastante\nPoco\nNada",
    "Acuerdo (1–5)": "Muy de acuerdo\nBastante de acuerdo\nRegular\nPoco de acuerdo\nNada de acuerdo",
    "Personalizada": None,
}
AUTO = "(automático)"
examples = load_examples()
by_label = {item["question"][:90]: item for item in examples["items"]}

st.title("Simulación de opinión pública en España")
st.markdown(
    "Estima cómo respondería la población adulta residente en España a una pregunta de encuesta, "
    "con un intervalo de confianza y el detalle por grupos. **Son estimaciones, no una encuesta real.**"
)


def pick_example():
    item = by_label.get(st.session_state.get("example"))
    if item:
        st.session_state.update(
            q=item["question"], opts="\n".join(item["options"]), template="Personalizada", result=item,
            opts_template="Personalizada", topic_select=item["topic"],
        )


with st.sidebar:
    st.header("Opciones")
    preset = st.selectbox("Modo de cálculo", PRESETS, format_func=PRESET_LABELS.get)
    if examples["items"]:
        st.selectbox("Ejemplos listos", ["—", *by_label], key="example", on_change=pick_example)
        st.caption("Los ejemplos se muestran al instante porque ya están calculados.")
    with st.expander("Avanzado"):
        topic = st.selectbox("Tema (si no se indica, se detecta)", [AUTO, *TOPICS], key="topic_select")
        model = st.text_input("Modelo de lenguaje", "qwen3:8b")
        segments = st.slider("Grupos de población simulados", 5, 40, 10)

question = st.text_area(
    "Pregunta", key="q", placeholder="¿Le preocupa que la inteligencia artificial le quite su empleo?", height=80
)
template = st.radio("Tipo de respuesta", list(TEMPLATES), horizontal=True, key="template")
if TEMPLATES[template] and st.session_state.get("opts_template") != template:
    st.session_state["opts"] = TEMPLATES[template]
    st.session_state["opts_template"] = template
options_text = st.text_area("Opciones de respuesta (una por línea)", key="opts", height=130)

if st.button("Simular", type="primary"):
    options = [o.strip() for o in options_text.splitlines() if o.strip()]
    try:
        resources = get_resources()
        service, notice = llm_service(preset, model)
        if notice:
            st.info(notice)
        with st.spinner("Calculando…"):
            st.session_state["result"] = predict(
                question,
                options,
                "single",
                preset,
                topic=None if topic == AUTO else topic,
                resources=resources,
                service=service,
                n_segments=segments,
            )
    except ServiceError as exc:
        st.session_state.pop("result", None)
        st.error(str(exc))
    except Exception:
        st.session_state.pop("result", None)
        st.error("No se pudo completar la simulación. Prueba con el modo rápido o revisa la pregunta y las opciones.")

if "result" in st.session_state:
    render(st.session_state["result"])
else:
    st.info("Escribe una pregunta, elige las opciones y pulsa **Simular**, o abre un ejemplo en la barra lateral.")
