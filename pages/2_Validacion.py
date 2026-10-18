import streamlit as st

from mavericks import validation_view as view
from mavericks.ui_charts import grouped_bars, scatter_pred_obs
from mavericks.ui_common import setup

setup("Validación", "✅")
st.title("¿Cuánto podemos fiarnos?")
st.markdown(
    "Probamos el sistema con **estudios del CIS que no conocía** (reservados y excluidos de toda la evidencia), "
    "comparamos lo que predijo con lo que respondió la gente en realidad y medimos el error."
)

data = view.load_backtest()
if data is None:
    st.warning("Aún no hay resultados de validación. Ejecuta `mavericks backtest` para generarlos.")
    st.stop()

results = data["results"]
done = [r for r in results if r["status"] == "completed"]
c1, c2, c3 = st.columns(3)
c1.metric("Preguntas evaluadas", len(done))
modes = view.by_mode(data)
c2.metric("Error medio del modelo", f"{sum(r['mae'] for r in done) / len(done) * 100:.1f} puntos")
c3.metric("Reparto uniforme (referencia)", f"{sum(r['mae_uniform_baseline'] for r in done) / len(done) * 100:.1f} puntos")
st.caption(
    "Error = diferencia media, en puntos porcentuales, entre el % predicho y el % real de cada opción. "
    "El «suelo de ruido» es el error que tendría incluso una predicción perfecta, solo por haber entrevistado a una muestra."
)

st.subheader("Según cómo se calcula")
st.markdown(
    "- **Directo:** la pregunta ya se hizo antes en el CIS; partimos de su última medición.\n"
    "- **Prior:** hay una pregunta muy parecida.\n"
    "- **Sin ancla:** pregunta nueva; es la situación más difícil."
)
st.dataframe(modes.round(3), hide_index=True, use_container_width=True)
st.plotly_chart(
    grouped_bars(
        modes, "Grupo", ["Error del modelo", "Reparto uniforme", "Resultado nacional previo", "Suelo de ruido"],
        ytitle="Error medio (0-1)",
    ),
    use_container_width=True,
)

st.subheader("Según el tema")
topics = view.by_topic(data)
st.dataframe(topics.round(3), hide_index=True, use_container_width=True)
st.plotly_chart(
    grouped_bars(topics, "Grupo", ["Error del modelo", "Reparto uniforme", "Suelo de ruido"], ytitle="Error medio (0-1)"),
    use_container_width=True,
)

st.subheader("Fiabilidad de los intervalos")
st.markdown("Un intervalo del 90 % debería contener el valor real en torno al 90 % de las veces.")
st.dataframe(view.coverage(data), hide_index=True, use_container_width=True)

st.subheader("Predicho frente a observado")
points = view.points(data)
mode = st.radio("Mostrar", ["Todos", *sorted(points["modo"].unique())], horizontal=True)
shown = points if mode == "Todos" else points[points["modo"] == mode]
st.plotly_chart(scatter_pred_obs(shown["predicho"], shown["observado"], shown["etiqueta"]), use_container_width=True)

with st.expander("Cómo se hace esta validación"):
    st.markdown(
        "1. Se reservan estudios del CIS publicados a partir de abril de 2026; **nunca** se usan para construir la evidencia.\n"
        "2. Para cada pregunta de esos estudios, el sistema solo puede usar encuestas anteriores a su fecha.\n"
        "3. Se compara el reparto predicho con el observado (con los pesos del CIS) y se registra el error.\n"
        "4. Los errores históricos construyen los intervalos y el nivel de confianza que ves en «Simular»."
    )
