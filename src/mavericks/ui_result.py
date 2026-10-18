"""Presentación de un resultado de ``predict`` en Streamlit."""

import pandas as pd
import streamlit as st

from . import batch
from .assist import SPONTANEOUS_NOTE, display_label, is_spontaneous
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
            f"En las preguntas de validación resueltas en modo {result['mode_label']}, el error típico fue de "
            f"{_points(conf['typical_error'])} por opción ({conf['backtest_cases']} preguntas)."
            if conf["typical_error"] is not None
            else "No hay validación suficiente para este tipo de pregunta."
        )
        st.markdown(
            f"<div class='card'><span class='pill {level}'>{CONFIDENCE_TEXT[level]}</span><br>{detail}</div>",
            unsafe_allow_html=True,
        )


def _pct(value):
    return "—" if value is None else f"{100 * value:.0f} %"


def render_national(result):
    node = result["national"]
    st.subheader("Distribución de respuestas en la población adulta")
    st.plotly_chart(distribution_bars(node["distribution"], None, core=node.get("interval_core")), use_container_width=True)
    info = node.get("interval_info")
    if node.get("interval_core") is not None:
        text = "La banda gruesa es el **rango habitual** de cada opción: la mitad de las veces el resultado real cae dentro"
        if info and info.get("coverage_50") is not None:
            text += f" (en preguntas nuevas no vistas cayó dentro el {_pct(info['coverage_50'])} de las veces)"
        if info and info.get("mean_halfwidth_pp") is not None:
            text += f"; de media ±{info['mean_halfwidth_pp']:.0f} puntos"
        st.caption(text + ".")
    if node.get("interval") is not None:
        with st.expander("Rango del 90 %"):
            rows = [
                {"Opción": o, "Estimación (%)": 100 * node["distribution"][o], "Mínimo (%)": 100 * lo, "Máximo (%)": 100 * hi}
                for o, (lo, hi) in node["interval"].items()
            ]
            st.dataframe(pd.DataFrame(rows).round(1), hide_index=True, use_container_width=True)
            cover = f" (en preguntas nuevas no vistas: {_pct(info['coverage_90'])})" if info and info.get("coverage_90") is not None else ""
            st.caption(f"Con este rango el resultado real debería quedar dentro 9 de cada 10 veces{cover}.")
    if result.get("spontaneous"):
        names = ", ".join(f"«{o}»" for o in result["spontaneous"])
        st.caption(f"{names}: {SPONTANEOUS_NOTE}.")
    for note in result.get("notes", []):
        st.info(note)
    used = result.get("prior_usado") or {}
    if used.get("metodo") == "media_knn_mlp":
        st.caption("Estimación: media de las preguntas parecidas del CIS y de un modelo aprendido de todas las preguntas anteriores del CIS.")
    elif used.get("metodo") == "knn":
        st.caption(f"Estimación basada solo en preguntas parecidas del CIS ({used.get('motivo')}).")


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
                "rango_min": "Rango probable mín. (%)",
                "rango_max": "Rango probable máx. (%)",
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
                        "Opción": [
                            display_label(o) + (" (no leída)" if is_spontaneous(o) else "") for o in item["distribution"]
                        ],
                        "%": [v * 100 for v in item["distribution"].values()],
                    }
                )
                st.dataframe(frame.round(1), hide_index=True, use_container_width=True)
                if any(is_spontaneous(o) for o in item["distribution"]):
                    st.caption("(no leída): " + SPONTANEOUS_NOTE + ".")


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
