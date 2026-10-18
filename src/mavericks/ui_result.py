"""Presentación de un resultado de ``predict`` en Streamlit."""

import pandas as pd
import streamlit as st

from . import batch
from .ui_charts import distribution_bars, heatmap, stacked_groups

CONFIDENCE_TEXT = {"alta": "Confianza alta", "media": "Confianza media", "baja": "Confianza baja"}


def _points(value):
    return "—" if value is None else f"{value * 100:.1f} puntos"


COVERAGE_ICON = {"respaldada": "🟢", "parecidas": "🟡", "sin_datos": "🟠", "fuera_de_alcance": "🔴"}


def render_coverage(card):
    """Tarjeta de cobertura: si la pregunta se puede responder con las encuestas y con qué respaldo."""
    extra = ""
    if card.get("reformulation"):
        extra = f"<br><b>Cómo reformularla:</b> {card['reformulation']}"
    st.markdown(
        f"<div class='card'><b>{COVERAGE_ICON[card['category']]} {card['label']}</b><br>{card['explanation']}{extra}</div>",
        unsafe_allow_html=True,
    )


def render_header(result):
    left, right = st.columns([3, 2])
    with left:
        st.markdown(
            f"<div class='card'><b>Cómo se ha calculado: {result['mode_label']}</b><br>{result['explanation']}"
            f"<br><span class='muted'>Tema: {result['topic'].replace('_', ' ')} · calculado en {result['seconds']:.1f} s</span></div>",
            unsafe_allow_html=True,
        )
    with right:
        conf = result["confidence"]
        level = conf["level"]
        detail = (
            "En preguntas del mismo tipo, el error típico medido con estudios que el sistema no había visto fue de "
            f"{_points(conf['typical_error'])} por opción ({conf['backtest_cases']} preguntas)."
            if conf["typical_error"] is not None
            else "No hay validación suficiente para este tipo de pregunta."
        )
        st.markdown(
            f"<div class='card'><span class='pill {level}'>{CONFIDENCE_TEXT[level]}</span><br>{detail}</div>",
            unsafe_allow_html=True,
        )


def render_national(result):
    node = result["national"]
    st.subheader("Distribución de respuestas en la población adulta")
    st.plotly_chart(distribution_bars(node["distribution"], node["interval"]), use_container_width=True)
    if node["radius"] is not None:
        st.caption(
            "Las líneas negras marcan un intervalo del 90 %: en preguntas de este tipo, el valor real quedó dentro de "
            f"±{node['radius'] * 100:.0f} puntos aproximadamente 9 de cada 10 veces."
        )


def render_subgroups(result, key="sub"):
    st.subheader("Por grupos de población")
    subgroups = result["subgroups"]
    if result.get("subgroup_note"):
        st.info(result["subgroup_note"])
    names = {k: v["label"] for k, v in subgroups.items()}
    left, right = st.columns([2, 3])
    dimension = left.selectbox("Dimensión", list(names), format_func=names.get, key=f"{key}_dim")
    view = right.radio(
        "Vista",
        ["Barras apiladas", "Mapa de calor", "Tabla con intervalos"],
        horizontal=True,
        key=f"{key}_view",
    )
    groups = subgroups[dimension]["groups"]
    options = result["options"]
    if view == "Barras apiladas":
        st.plotly_chart(stacked_groups(groups, options), use_container_width=True)
    elif view == "Mapa de calor":
        st.plotly_chart(heatmap(groups, options), use_container_width=True)
    else:
        table = batch.subgroup_frame(result, dimension).rename(
            columns={
                "grupo": "Grupo",
                "opcion": "Opción",
                "porcentaje": "%",
                "ic90_min": "IC 90 % mín.",
                "ic90_max": "IC 90 % máx.",
                "peso_poblacion": "% de la población",
            }
        )
        option = st.selectbox("Opción", options, key=f"{key}_opt")
        shown = table[table["Opción"] == option].drop(columns=["dimension"])
        st.dataframe(shown.round(1), hide_index=True, use_container_width=True)
    if result["mode"] == "direct":
        st.caption(
            "Los grupos con pocas entrevistas en el CIS se acercan al conjunto de la población "
            "para no sacar conclusiones de muestras pequeñas."
        )


def render_evidence(result):
    st.subheader("Preguntas del CIS en las que se apoya")
    if not result["evidence"]:
        st.write("No se ha encontrado ninguna pregunta relacionada.")
    for item in result["evidence"]:
        similarity = f" · similitud {item['similarity']:.0%}" if item.get("similarity") else ""
        title = f"{item['evidence_id']} · estudio {item['study_id']} · {item['fieldwork_end'] or 's/f'}{similarity}"
        with st.expander(title):
            st.write(item["question"])
            if item["distribution"]:
                frame = pd.DataFrame(
                    {
                        "Opción": list(item["distribution"]),
                        "%": [v * 100 for v in item["distribution"].values()],
                    }
                )
                st.dataframe(frame.round(1), hide_index=True, use_container_width=True)


def render_downloads(result, key="dl"):
    left, right, _ = st.columns([1, 1, 3])
    left.download_button("Descargar CSV", batch.to_csv(result), "resultado.csv", "text/csv", key=f"{key}_csv")
    right.download_button(
        "Descargar JSON", batch.to_json(result), "resultado.json", "application/json", key=f"{key}_json"
    )


def render(result, key="res"):
    st.markdown(f"### {result['question']}")
    if result.get("coverage"):
        render_coverage(result["coverage"])
    render_header(result)
    for warning in result["warnings"]:
        st.warning(warning)
    render_national(result)
    render_subgroups(result, key)
    render_evidence(result)
    render_downloads(result, key)
