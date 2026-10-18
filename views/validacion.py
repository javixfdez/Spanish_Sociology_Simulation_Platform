import pandas as pd
import streamlit as st

from mavericks import validation_view as view
from mavericks.ui_charts import grouped_bars, scatter_pred_obs
from mavericks.ui_common import setup

setup()
st.title("¿Cuánto podemos fiarnos?")
st.markdown(
    "Probamos el sistema con **preguntas de estudios del CIS que no conocía** (excluidas de toda la evidencia), "
    "comparamos lo que predijo con lo que respondió la gente en realidad y medimos el error. "
    "Todos los errores están en **puntos porcentuales** por opción."
)

fast = view.load_fast()
baseline = view.load_baseline()
floors = view.load_floors()
if fast is None and baseline is None:
    st.warning("Aún no hay resultados de validación. Ejecuta `mavericks fast-validation` y `mavericks backtest`.")
    st.stop()

# Cifras destacadas
cols = st.columns(3)
if fast:
    coverage = view.fast_coverage(fast)
    cols[0].metric("Cobertura del rango probable (modo rápido)", f"{coverage['global'][0]:.1f} %")
if baseline:
    base_cov = view.baseline_coverage(baseline)["global"]
    cols[1].metric("Cobertura del rango probable (con LLM)", f"{base_cov[0]:.1f} %")
if fast:
    rows = view.fast_rows(fast)
    cols[2].metric(
        "Error medio, modo rápido",
        f"{100 * sum(r['mae'] for r in rows) / len(rows):.1f} puntos",
    )
st.caption(
    "Cobertura: el «rango probable» del 90 % debe contener el valor real en torno al 90 % de las veces. "
    "Se mide dejando cada pregunta fuera al calcular su rango."
)

st.subheader("Qué configuración respalda lo que ves en la demo")
st.markdown(
    "La demo usa por defecto el **modo rápido sin LLM**: su validación es la fila «Modo rápido sin LLM». "
    "La línea «Con LLM» es la validación del pipeline completo con modelo de lenguaje, solo con las preguntas reservadas."
)
config = view.configurations(fast, baseline, floors)
shown = config.drop(columns=["SimBench"]).round(1)
st.dataframe(
    shown[["Configuración", "Grupo", "Preguntas", "Error del modelo (pp)", "Reparto uniforme (pp)", "Suelo de ruido (pp)"]],
    hide_index=True,
    use_container_width=True,
)
st.caption(
    "«Suelo de ruido»: el error que tendría incluso una predicción perfecta, solo por haber entrevistado a una muestra."
)
st.plotly_chart(
    grouped_bars(
        config[config["Configuración"].str.startswith("Modo rápido sin LLM (test + desarrollo)")],
        "Grupo",
        ["Error del modelo (pp)", "Reparto uniforme (pp)", "Suelo de ruido (pp)"],
        ytitle="Error medio (puntos)",
    ),
    use_container_width=True,
)

if fast:
    rows = view.fast_rows(fast)
    st.subheader("Según el tema")
    test_rows = [r for r in rows if r["set"] == "test"]
    dev_rows = [r for r in rows if r["set"] == "dev"]
    st.markdown("**Preguntas reservadas (test)**")
    st.dataframe(
        view.summary(test_rows, floors, "topic").drop(columns=["SimBench"]).round(1),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown("**Preguntas de desarrollo**")
    st.info(view.DEV_ONLY_NOTE)
    st.dataframe(
        view.summary(dev_rows, floors, "topic").drop(columns=["SimBench"]).round(1),
        hide_index=True,
        use_container_width=True,
    )

    st.subheader("Fiabilidad de los rangos probables")
    cv = fast["by_mode"]["cv"]
    names = {"direct": "Directo", "prior": "Prior", "none": "Sin ancla", "global": "Global"}
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Modo": names[m],
                    "Opciones evaluadas": v["options"],
                    "Cobertura (%)": 100 * v["coverage"],
                    "Anchura media (puntos)": v.get("mean_width_pp"),
                }
                for m, v in cv.items()
            ]
        ).round(1),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "En prior hay muy pocas preguntas de validación, por eso su rango se calcula junto con el de sin ancla "
        "(más amplio). En preguntas nuevas el error es grande y el rango también: es una limitación real del método."
    )

    st.subheader("Predicho frente a observado")
    points = view.points(view.fast_rows(fast))
    mode = st.radio("Mostrar", ["Todos", *sorted(points["modo"].unique())], horizontal=True)
    selected = points if mode == "Todos" else points[points["modo"] == mode]
    st.plotly_chart(
        scatter_pred_obs(selected["predicho"], selected["observado"], selected["etiqueta"]), use_container_width=True
    )

with st.expander("Detalle técnico (SimBench)"):
    st.markdown(
        "SimBench = 100·(1 − TVD del modelo / TVD del reparto uniforme): 100 es perfecto y 0 no mejora al reparto uniforme."
    )
    st.dataframe(config[["Configuración", "Grupo", "SimBench"]].round(1), hide_index=True, use_container_width=True)

with st.expander("Cómo se hace esta validación"):
    st.markdown(
        "1. Se reservan estudios del CIS publicados a partir de abril de 2026; **nunca** se usan para construir la evidencia.\n"
        "2. Para cada pregunta, el sistema solo puede usar encuestas anteriores a su fecha y nunca la propia pregunta.\n"
        "3. Se compara el reparto predicho con el observado (con los pesos del CIS) y se registra el error.\n"
        "4. Los errores históricos construyen el rango probable y el nivel de confianza que ves en «Simular»."
    )
