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

prior = view.load_prior_validation()
if prior:
    st.subheader("Validación con datos no vistos")
    st.markdown(
        "Para las preguntas **nuevas** (sin ninguna parecida en el pasado) la demo usa la **media de las preguntas parecidas del CIS y de un "
        "modelo aprendido** de todas las preguntas anteriores. Así se comprobó, con tres conjuntos de preguntas y estudios distintos:"
    )
    cols = st.columns(3)
    for col, (title, delta, ci, n) in zip(cols, view.prior_replication(prior), strict=True):
        col.metric(title.split(" (")[0], f"{100 * delta:+.1f} puntos", help=f"IC 95 % por estudio [{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]; {n} preguntas")
    st.caption(
        "Mejora de la media frente a usar solo las preguntas parecidas (error en puntos porcentuales; negativo = mejor). "
        "La mejora se repite en los tres conjuntos."
    )
    st.dataframe(view.prior_table(prior).round(1), hide_index=True, use_container_width=True)
    official = view.load_official()
    if official:
        st.markdown("### Evaluación final oficial (estudios reservados)")
        st.markdown(
            f"Una sola ejecución, con la configuración congelada de antemano, sobre los {official['casos']} casos de los estudios reservados "
            "(que nunca se usaron para construir el sistema). Error en puntos porcentuales; menos es mejor."
        )
        st.dataframe(view.official_table(official).round(1), hide_index=True, use_container_width=True)
        st.dataframe(view.official_deltas(official).round(1), hide_index=True, use_container_width=True)
        st.caption(
            "Las preguntas repetidas se resuelven con la estadística directa en los dos modos. " + official["nota"]
        )
    st.markdown("**Frente al reparto uniforme**, por número de opciones:")
    st.dataframe(view.prior_vs_uniform(prior).round(1), hide_index=True, use_container_width=True)
    st.info(
        "Honestidad: frente al reparto uniforme la mejora es clara en preguntas de 3 o más opciones; en las de sí/no ningún método "
        "se separa del uniforme de forma fiable."
    )
    with st.expander("Cómo se validó (protocolo)"):
        st.markdown(
            "1. Los ajustes se eligieron con un conjunto de desarrollo grande (1.460 preguntas de 2025-07 a 2026-03), prediciendo cada pregunta solo con lo anterior a su fecha y sin su estudio.\n"
            "2. La media se confirmó con una comparación **prerregistrada** (fijada en git antes de ejecutarla) en estudios no vistos de 2025-01 a 2025-06: mejora de 2,4 puntos.\n"
            "3. Una segunda mejora propuesta (usar solo el modelo aprendido en preguntas de sí/no) se probó en estudios de 2024-07 a 2024-12 y **no se confirmó**; se descartó.\n"
            "4. Una variante que parecía ganar en 44 casos dejó de ganar al ampliar el conjunto («maldición del ganador»): por eso se exigen confirmaciones externas.\n"
            "5. Los intervalos de confianza se calculan remuestreando estudios completos, porque las preguntas de un mismo estudio están correlacionadas."
        )

with st.expander("Modo ajustado (opcional): un modelo pequeño ajustado con encuestas reales"):
    st.markdown(
        "Qwen3-4B-Instruct-2507 con un adaptador QLoRA (r=8, alpha=32) entrenado para que la probabilidad de su primer token sobre las opciones A, B, C… coincida con el reparto **real** "
        "(CIS, Eurobarómetro y ESS; método de Cao et al. 2025). Dos modelos: **a1** (trabajo de campo anterior a 2025-07-01) para evaluar en datos de desarrollo y **B** (anterior a 2026-04-01) para la evaluación final.\n\n"
        "**1) Desarrollo (a1, prerregistrado):** 703 preguntas nuevas de 23 estudios (2025-07 a 2026-03), error en puntos; Δ frente al prior de producción con IC 95 % por estudio."
    )
    st.dataframe(
        pd.DataFrame(
            [
                ["Reparto uniforme", 24.22, "+5,60 [+3,85; +7,04]"],
                ["KNN actual", 20.72, "+2,09 [+0,61; +3,24]"],
                ["Prior de producción (50/50 KNN + MLP)", 18.63, "—"],
                ["Qwen base sin ajustar", 34.19, "+15,56 [+13,76; +18,87]"],
                ["a1 (ajustado)", 12.44, "−6,18 [−8,67; −2,37]  → CUMPLE"],
                ["Media 50/50 a1 + prior", 14.78, "−3,85 [−5,00; −2,07]  → CUMPLE"],
            ],
            columns=["Sistema", "Error (pp)", "Δ frente al prior [IC 95 %]"],
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Búsqueda de fuga (la mejora supera los 4 puntos): entrenamiento hasta 2025-06-30 y dev_grande desde 2025-07-07; ningún identificador ni estudio compartido; 36 preguntas tienen texto idéntico a una de entrenamiento y, sin ellas, la mejora es de −5,93 puntos."
    )
    st.markdown("**2) Evaluación final oficial (B, una sola ejecución, 88 casos de 11 estudios reservados):** Δ frente al modo rápido.")
    st.dataframe(
        pd.DataFrame(
            [
                ["Reparto uniforme", 17.13, "+9,02 [+6,72; +11,11]"],
                ["Sistema anterior", 8.76, "+0,65 [−0,19; +1,56]"],
                ["Modo rápido", 8.11, "—"],
                ["Modo con LLM", 7.91, "−0,20 [−0,54; +0,13]"],
                ["Modo ajustado (B)", 6.84, "−1,28 [−2,62; +0,01]  → NO CUMPLE"],
            ],
            columns=["Sistema", "Error (pp)", "Δ frente al modo rápido [IC 95 %]"],
        ),
        hide_index=True,
        use_container_width=True,
    )
    st.markdown(
        "Desglose (modo rápido → ajustado): directos (47) 2,49 → 2,49; sin ancla (40) 14,57 → 12,05; no directos (41) 14,57 → 11,82; 2 opciones (13) 17,41 → 13,91; 3 o más (28) 13,25 → 10,85. "
        "Cobertura del rango del 90 % en los no directos: 86,5 %. Los casos de 8 opciones (el modelo cubre de 2 a 7) usan el modo rápido; la regla se fijó antes de ver resultados."
    )
    st.info(
        "Decisión: el criterio prerregistrado exigía que el intervalo quedara entero por debajo de cero y llega a +0,01, así que **el modo rápido sigue por defecto**; el ajustado es opcional. "
        "La comparación «B también con ancla directa» (6,55 puntos) es secundaria y no se adopta: queda como trabajo futuro. "
        "Limitaciones: poca potencia (41 casos informativos de 9 estudios), solo 2–7 opciones, a1 y B no son el mismo modelo (bf16 frente a fp16) y no se conservaron los puntos intermedios de a1."
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
