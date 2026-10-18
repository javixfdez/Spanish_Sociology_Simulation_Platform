import streamlit as st

from mavericks.ui_common import setup

setup("Metodología", "🧭")
st.title("Cómo funciona")
st.markdown("Principio de diseño: **la estadística va primero y el modelo de lenguaje solo ajusta.**")

st.graphviz_chart(
    """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eaf2f8", color="#1f4e79", fontname="Helvetica"];
  Q [label="Pregunta\\ny opciones"];
  R [label="Búsqueda de preguntas\\nparecidas en el CIS"];
  M [label="¿Ya se hizo?"];
  D [label="Directo:\\núltima medición real"];
  P [label="Prior:\\npregunta parecida"];
  N [label="Sin ancla:\\nmedia de análogas"];
  S [label="Ajuste por grupo\\n(sexo, edad, estudios…)"];
  A [label="Población sintética\\n50 000 adultos"];
  I [label="Intervalos y confianza\\n(errores históricos)"];
  Q -> R -> M; M -> D; M -> P; M -> N; D -> S; P -> S; N -> S; S -> A -> I;
}
"""
)

st.subheader("Fuentes")
st.markdown(
    "| Fuente | Uso | Fecha |\n|---|---|---|\n"
    "| CIS, barómetros y estudios | Evidencia sobre lo que la gente responde | Hasta marzo de 2026 (estudios posteriores reservados para validar) |\n"
    "| INE, Censo 2021 | Estructura de la población | 2021 |\n"
    "| INE, Estadística Continua de Población | Totales por sexo, edad y comunidad | 1 de enero de 2026 |\n"
    "| INE, EPA y ECV | Estudios e ingresos del hogar | 2.º trim. 2026 y 2025 |\n"
    "| Eurobarómetro y ESS | Evidencia complementaria (solo España) | Según estudio |"
)

st.subheader("Reglas")
st.markdown(
    "- Solo se usa información **anterior** a la fecha del estudio que se predice.\n"
    "- Si la pregunta ya existe en el CIS, se parte de su **oleada más reciente** y se ajusta con las diferencias reales entre grupos.\n"
    "- Si solo existe una pregunta parecida, se usa como punto de partida con más incertidumbre.\n"
    "- Si no existe nada parecido, se promedian las preguntas más cercanas y se indica baja confianza.\n"
    "- Los intervalos salen de los errores reales medidos en la validación, no de supuestos teóricos.\n"
    "- Los grupos con pocas entrevistas se acercan al conjunto de la población (reducción jerárquica)."
)

st.subheader("Limitaciones")
st.markdown(
    "- Son **estimaciones**; no sustituyen a una encuesta real.\n"
    "- Para preguntas nuevas el error es mucho mayor que para preguntas repetidas y no se distinguen bien los grupos.\n"
    "- La población sintética refleja variables sociodemográficas, no actitudes individuales.\n"
    "- Las opiniones cambian: una medición antigua puede quedarse desfasada tras un acontecimiento reciente.\n"
    "- El CIS pregunta por teléfono o cara a cara; hay sesgos de respuesta que no corregimos."
)
