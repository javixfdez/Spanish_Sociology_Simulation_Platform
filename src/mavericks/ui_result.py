"""Presentación de un resultado de ``predict`` en Streamlit."""

import pandas as pd
import streamlit as st

from . import batch, explain
from .assist import SPONTANEOUS_NOTE, display_label, is_spontaneous
from .ui_charts import distribution_bars, heatmap, stacked_groups

SMALL_MEASURED = "Muestra pequeña: el dato medido es poco fiable; mira la estimación del modelo."
PRESET_LABELS = {"rapido_sin_llm": "Rápido", "llm_completo": "Con LLM", "recomendado": "Con LLM", "ajustado": "Ajustado"}

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


def render_produced_by(result):
    """Qué modo produjo las cifras y, si hubo que volver al modo rápido, el motivo (siempre visible)."""
    info = result.get("producido_por")
    if not info:
        return
    st.markdown(f"**Modo que ha producido este resultado: {info['etiqueta']}**" + (f" (solicitado: {PRESET_LABELS.get(info['solicitado'], info['solicitado'])})" if info.get("solicitado") else ""))
    if info.get("fallback"):
        st.warning(f"Se ha vuelto al modo rápido: {info['motivo']}. Las cifras son las del modo rápido.")
    if info.get("experimental"):
        st.info(info["experimental"])
    if info.get("nota"):
        st.caption(info["nota"])


def render_header(result):
    render_produced_by(result)
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


def render_measured(result):
    """«Lo que midió el CIS»: reparto ponderado de los microdatos de la oleada de la pregunta, sin ningún modelo."""
    m = result["measured"]
    nat = m["national"]
    st.subheader("Lo que midió el CIS")
    st.caption(
        f"Estudio {m['study_id']} · trabajo de campo {m['fieldwork_end'] or 's/f'} · {nat['n']} encuestados (n efectivo {nat['n_eff']:.0f}). "
        "Reparto ponderado de los microdatos, sin ningún modelo; la barra es el margen de error muestral al 95 % (±1,96·√(p(1−p)/n efectivo))."
    )
    st.plotly_chart(distribution_bars(nat["distribution"], nat["interval"]), use_container_width=True, key="measured_national")
    table = pd.DataFrame([{"Opción": o, "Medido (%)": 100 * nat["distribution"][o], "± puntos": 100 * nat["margin"][o]} for o in nat["distribution"]])
    st.dataframe(table.round(1), hide_index=True, use_container_width=True)
    for n, cross in enumerate(m["crosses"]):
        st.markdown(f"**{cross['label']}** · {cross['n']} encuestados")
        if cross["distribution"] is None:
            st.warning("No hay encuestados con respuesta en este grupo en la oleada de la pregunta.")
            continue
        if cross["small"]:
            st.warning(SMALL_MEASURED)
        st.dataframe(
            pd.DataFrame([{"Opción": o, "Medido (%)": 100 * cross["distribution"][o], "± puntos": 100 * cross["margin"][o]} for o in cross["distribution"]]).round(1),
            hide_index=True, use_container_width=True, key=f"measured_cross_{n}",
        )


def render_national(result):
    node = result["national"]
    st.subheader("Estimación del modelo para hoy" if "measured_note" in result else "Distribución de respuestas en la población adulta")
    st.plotly_chart(distribution_bars(node["distribution"], None, core=node.get("interval_core")), use_container_width=True)
    info = node.get("interval_info")
    if node.get("interval_core") is not None:
        text = "La banda gruesa es el **rango habitual** de cada opción: la mitad de las veces el resultado real cae dentro"
        if info and info.get("coverage_50") is not None:
            text += f" (medido dejando fuera una ventana temporal entera, cayó dentro el {_pct(info['coverage_50'])} de las veces)"
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
            cover = f" (medido dejando fuera una ventana temporal entera: {_pct(info['coverage_90'])})" if info and info.get("coverage_90") is not None else ""
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
    method = result.get("subgroups_method")
    if method:
        extra = ""
        if method["clave"] == "mrp":
            extra = " Los rangos son los del método anterior (calibrados con él, por tanto prudentes). Validado entre oleadas de la misma pregunta: mejora la edad, sexo, estudios e ingresos; en comunidad autónoma empata con el método anterior."
        st.caption(f"Método de los subgrupos: **{method['etiqueta']}**.{extra}")
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


def render_crosses(result):
    """Cruces de varias variables (experimentales, solo MRP en modo directo) o el motivo por el que no se ofrecen."""
    crosses = result.get("crosses") or []
    if not crosses:
        if result.get("cross_note"):
            st.info(result["cross_note"])
        return
    st.subheader("Cruce de varias variables")
    st.info(result["cross_experimental"])
    for n, cross in enumerate(crosses):
        st.markdown(f"**{cross['label']}** · {cross['n_cis']} encuestados del CIS en la oleada de la pregunta · método {cross['method']}")
        if cross["small"]:
            st.warning(f"Solo {cross['n_cis']} encuestados en este cruce: la estimación se apoya sobre todo en el modelo.")
        st.plotly_chart(distribution_bars(cross["distribution"], cross.get("interval")), use_container_width=True, key=f"cross_{n}")


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


def show_text(slot, result):
    info = result.get("explanation_text")
    if not info:
        return
    with slot.container():
        st.markdown("**Resumen**")
        st.write(info["texto"])
        if info["origen"] == "llm":
            st.caption("Texto redactado por el modelo de lenguaje a partir de las cifras calculadas; las cifras se han comprobado.")
        elif info.get("motivo"):
            st.caption(f"Resumen automático (no se usó el texto del modelo: {info['motivo']}).")
        else:
            st.caption("Resumen automático a partir de las cifras calculadas.")


def render(result, key="res", llm=None, timeout=20.0):
    """Muestra el resultado. Los números salen primero; si hay ``llm`` (modo con LLM), el texto redactado sustituye
    a la plantilla al final, sin retrasar nada de lo anterior."""
    st.markdown(f"### {result['question']}")
    if result.get("coverage"):
        render_coverage(result["coverage"])
    render_header(result)
    for warning in result["warnings"]:
        st.warning(warning)
    measured = result.get("measured")
    if measured and result.get("intent") == "consulta_dato":
        render_measured(result)  # se preguntó por lo medido: va primero
    elif result.get("measured_note"):
        st.info(result["measured_note"])
    render_national(result)
    if measured and result.get("intent") != "consulta_dato":
        render_measured(result)
    slot = st.empty()
    show_text(slot, result)
    render_subgroups(result, key)
    render_crosses(result)
    render_evidence(result)
    render_downloads(result, key)
    info = result.get("explanation_text") or {}
    if llm is not None and "segundos" not in info:
        with st.spinner("Redactando la explicación con el modelo de lenguaje…"):
            explain.add_llm_text(result, llm, timeout)
        show_text(slot, result)
