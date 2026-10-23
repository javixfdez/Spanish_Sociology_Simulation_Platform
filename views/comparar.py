import pandas as pd
import streamlit as st

from mavericks import compare
from mavericks.ui_charts import comparison_bars
from mavericks.ui_common import get_resources, setup

setup()
st.title("Comparar formulaciones")
st.markdown(
    "Escribe de 2 a 10 maneras de plantear la misma pregunta (o mensajes distintos) con **las mismas opciones** y mira cuál recibe más apoyo estimado. "
    "Cada una se calcula en modo rápido, igual que en «Simular». **Son estimaciones, no una encuesta real**: cuando las diferencias son menores que los rangos, no hay diferencia."
)

formulations_text = st.text_area(
    "Formulaciones (una por línea)", key="cmp_formulations", height=190,
    placeholder="¿Está de acuerdo con limitar el precio del alquiler?\n¿Apoyaría una ley que limite los alquileres para que las familias puedan pagarlos?",
)
options_text = st.text_area("Opciones de respuesta (una por línea, las mismas para todas)", key="cmp_options", height=110, placeholder="A favor\nEn contra\nNi a favor ni en contra")
formulations, options = compare.parse_lines(formulations_text), compare.parse_lines(options_text)
chosen = st.selectbox("Ordenar por esta opción", options, key="cmp_option") if len(options) >= 2 else None
st.caption(f"{len(formulations)} formulaciones · {len(options)} opciones")

if st.button("Comparar", type="primary"):
    st.session_state.pop("cmp_rows", None)
    if not compare.MIN_FORMULATIONS <= len(formulations) <= compare.MAX_FORMULATIONS:
        st.error(f"Escribe entre {compare.MIN_FORMULATIONS} y {compare.MAX_FORMULATIONS} formulaciones, una por línea.")
    elif len(options) < 2 or chosen is None:
        st.error("Escribe al menos dos opciones, una por línea.")
    else:
        resources = get_resources()
        bar = st.progress(0.0, text="Calculando…")
        rows = compare.compare(formulations, options, chosen, resources, progress=lambda n, total: bar.progress(n / total, text=f"Calculando {n} de {total}…"))
        bar.empty()
        st.session_state["cmp_rows"] = {"rows": rows, "option": chosen}

saved = st.session_state.get("cmp_rows")
if saved:
    frame = compare.to_frame(saved["rows"])
    done = frame[frame["Estado"] == "ok"]
    st.subheader(f"Apoyo estimado a «{saved['option']}»")
    if len(done):
        labels = [f"{i}. {text[:70]}{'…' if len(text) > 70 else ''}" for i, text in enumerate(done["Formulación"], start=1)]
        def values(column):
            return [None if pd.isna(v) else float(v) for v in done[column]]

        st.plotly_chart(comparison_bars(labels, values("% estimado"), values("Rango habitual mín. (%)"), values("Rango habitual máx. (%)")), use_container_width=True)
        st.caption("La barra gruesa es el rango habitual (la mitad de las veces el resultado real cae dentro).")
    st.dataframe(frame, hide_index=True, use_container_width=True)
    for _, row in frame[frame["Estado"] != "ok"].iterrows():
        st.warning(f"«{row['Formulación']}»: {row['Estado']}")
    st.download_button("Descargar CSV", compare.to_csv(frame), "comparacion_formulaciones.csv", "text/csv")
