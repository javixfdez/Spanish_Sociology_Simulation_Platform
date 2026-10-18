import streamlit as st

from mavericks.ui_common import setup

setup()
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

st.subheader("Dos modos de cálculo")
st.markdown(
    "- **Rápido** (por defecto): los números salen del prior estadístico (preguntas parecidas del CIS + modelo aprendido), sin modelo de lenguaje; "
    "la explicación escrita usa plantillas con las cifras calculadas. Responde al instante.
"
    "- **Con LLM**: los números mezclan el modelo de lenguaje (peso 0,3) con el prior (peso 0,7), y el modelo redacta la explicación a partir "
    "de las cifras ya calculadas; si algún número del texto no coincide con los calculados, o tarda más de 20 s, se muestra la plantilla.
"
    "- **Precisión**: con el peso elegido por validación cruzada dejando un estudio fuera, la mezcla da 15,06 puntos de error frente a 14,84 del prior solo "
    "(44 casos de desarrollo). En precisión, los dos modos son equivalentes dentro del error de medida; el valor del modo con LLM está "
    "en la explicación y en el criterio cualitativo del modelo."
)

st.subheader("Reglas")
st.markdown(
    "- Solo se usa información **anterior** a la fecha del estudio que se predice.\n"
    "- Si la pregunta ya existe en el CIS, se parte de su **oleada más reciente** y se ajusta con las diferencias reales entre grupos (modo directo).\n"
    "- Si es nueva, la estimación es la **media 50/50 de dos predictores**: las preguntas más parecidas del CIS (5 vecinos ponderados por parecido) y un **modelo aprendido** (red pequeña sobre embeddings multilingües e5) entrenado con todas las preguntas anteriores del CIS. Si el modelo no está disponible, se usan solo las parecidas y se indica.\n"
    "- Los rangos se calibran por separado para preguntas de 2 y de 3 o más opciones: el **rango habitual** (50 %) es el principal y el del 90 % está en un desplegable.\n"
    "- Los grupos con pocas entrevistas se acercan al conjunto de la población (reducción jerárquica); en preguntas nuevas las diferencias entre grupos se toman de preguntas parecidas, contraídas a la mitad.\n"
    "- Cada mejora se valida con estudios que el sistema no había visto, con confirmaciones **prerregistradas** (ver «Validación»)."
)

st.subheader("Limitaciones")
st.markdown(
    "- Son **estimaciones**; no sustituyen a una encuesta real.\n"
    "- Para preguntas nuevas el error es mucho mayor que para repetidas; en las de sí/no ningún método se separa del reparto uniforme de forma fiable.\n"
    "- La población sintética refleja variables sociodemográficas, no actitudes individuales.\n"
    "- Las opiniones cambian: una medición antigua puede quedarse desfasada tras un acontecimiento reciente.\n"
    "- El CIS pregunta por teléfono o cara a cara; hay sesgos de respuesta que no corregimos."
)
