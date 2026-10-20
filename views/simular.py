import streamlit as st

from mavericks import assist, mrp_service
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
from mavericks.ui_result import render, render_coverage, render_synthetic

setup()

AUTO = "(automático)"
CROSS_ANY = "(cualquiera)"
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
            st.warning("El servidor del modelo ajustado no responde (`scripts\\run_server.ps1`): se usará el modo rápido y el resultado lo indicará.")
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
            suggestion, plan = None, None
            if coverage["category"] != "fuera_de_alcance":
                suggestion = suggest_options(question, resources, service, chosen_topic)
                st.session_state["opts"] = "\n".join(suggestion["options"])
                for dim in mrp_service.DIMENSIONS:
                    st.session_state[f"cross_{dim}"] = CROSS_ANY
                if service is not None:  # plan estructurado (20 s como máximo); si falla, se usan las reglas y los selectores manuales
                    plan = assist.plan_question(question, service, mrp_service.schema_of(resources.population.frame))
                    if plan["ok"] and plan["plan"]["cruce"] and coverage["category"] == "respaldada":
                        for item in plan["plan"]["cruce"]:
                            st.session_state[f"cross_{item['var']}"] = item["val"]
        st.session_state["analysis"] = {"question": question, "coverage": coverage, "suggestion": suggestion, "plan": plan}
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
        plan = analysis.get("plan")
        if plan and plan["ok"]:
            st.caption(f"Plan del modelo de lenguaje (es una sugerencia: revisa los selectores): tema «{plan['plan']['tema']}»" + (f"; grupo pedido: {mrp_service.cross_label([(i['var'], i['val']) for i in plan['plan']['cruce']])}" if plan["plan"]["cruce"] else "") + ".")
            if plan.get("aviso"):
                st.warning(plan["aviso"])
            if not plan["plan"]["respondible"]:
                st.info(f"El modelo duda de que sea una pregunta de opinión: {plan['plan']['motivo']}")
        elif plan:
            st.caption(f"Plan del modelo no usado ({plan['reason']}); se aplican las reglas de siempre.")
        crosses = []
        if analysis["coverage"]["category"] == "respaldada":
            with st.expander("Subgrupo o cruce de varias variables (experimental)"):
                st.caption("Elige una o más variables para ver un grupo concreto (por ejemplo, mujeres de 18-24 años de Andalucía). Experimental: los cruces de varias variables no están validados; la validación cubre subgrupos de una variable.")
                schema = mrp_service.schema_of(get_resources().population.frame)
                chosen = []
                columns = st.columns(len(schema))
                for column, (dim, values) in zip(columns, schema.items(), strict=True):
                    value = column.selectbox(mrp_service.DIMENSION_LABELS[dim], [CROSS_ANY, *values], key=f"cross_{dim}")
                    if value != CROSS_ANY:
                        chosen.append({"var": dim, "val": value})
                if chosen:
                    crosses = [chosen]
        else:
            st.caption("Los cruces de varias variables solo se ofrecen cuando la pregunta existe en el CIS (modo directo): hacen falta los microdatos de la misma pregunta para ajustar el modelo.")
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
                        resources=resources, service=service, n_segments=segments, crosses=crosses or None, intent=(plan["plan"]["intencion"] if plan and plan["ok"] else None),
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
    render_synthetic(st.session_state["result"], get_resources())
elif not analysis:
    st.info("Escribe una pregunta y pulsa **Analizar pregunta**, o abre un ejemplo en la barra lateral.")
