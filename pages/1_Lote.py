import streamlit as st

from mavericks import batch
from mavericks.service import PRESETS, ServiceError
from mavericks.ui_common import PRESET_LABELS, get_resources, llm_service, setup

setup("Lote", "🗂️")
st.title("Procesar varias preguntas a la vez")
st.markdown(
    "Sube un archivo CSV o Excel con una fila por pregunta y descarga un Excel con el total y el detalle por grupos."
)
st.markdown("**Columnas:** `pregunta`, `opciones` (separadas por `|`) y `tipo` (`unica`).")
st.download_button(
    "Descargar plantilla de ejemplo",
    batch.TEMPLATE.to_csv(index=False).encode("utf-8-sig"),
    "plantilla_preguntas.csv",
    "text/csv",
)

preset = st.selectbox("Modo de cálculo", PRESETS, format_func=PRESET_LABELS.get)
uploaded = st.file_uploader("Archivo de preguntas", type=["csv", "xlsx", "xls"])

if uploaded and st.button("Procesar", type="primary"):
    try:
        rows = batch.read_table(uploaded, uploaded.name)
        resources = get_resources()
        service, notice = llm_service(preset)
        if notice:
            st.info(notice)
        bar = st.progress(0.0, text="Preparando…")

        def progress(done, total):
            bar.progress(done / total, text=f"Pregunta {done} de {total}")

        st.session_state["batch"] = batch.run_batch(rows, preset, resources, service, progress)
        bar.empty()
    except ServiceError as exc:
        st.session_state.pop("batch", None)
        st.error(str(exc))

results = st.session_state.get("batch")
if results:
    ok = [r for r in results if r["ok"]]
    st.success(f"{len(ok)} de {len(results)} preguntas procesadas.")
    for item in results:
        if not item["ok"]:
            st.warning(f"Fila {item['row']}: {item['error']}")
    if ok:
        import pandas as pd

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Fila": r["row"], "Pregunta": r["result"]["question"], "Tema": r["result"]["topic"],
                        "Cálculo": r["result"]["mode_label"], "Confianza": r["result"]["confidence"]["level"],
                        "Opción más probable": max(
                            r["result"]["national"]["distribution"], key=r["result"]["national"]["distribution"].get
                        ),
                    }
                    for r in ok
                ]
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.download_button(
            "Descargar Excel con todos los resultados",
            batch.to_excel(results),
            "resultados.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
