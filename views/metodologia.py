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
    "- **Rápido** (por defecto): los números salen del prior estadístico (preguntas parecidas del CIS + modelo aprendido), sin modelo de lenguaje; la explicación escrita usa plantillas con las cifras calculadas. Responde al instante.\n"
    "- **Con LLM**: los números mezclan el modelo de lenguaje (peso 0,3) con el prior (peso 0,7), y el modelo redacta la explicación a partir de las cifras ya calculadas; si algún número del texto no coincide con los calculados, o tarda más de 20 s, se muestra la plantilla.\n"
    "- **Precisión**: con el peso elegido por validación cruzada dejando un estudio fuera, la mezcla da 15,06 puntos de error frente a 14,84 del prior solo (44 casos de desarrollo). En precisión, los dos modos son equivalentes dentro del error de medida; el valor del modo con LLM está en la explicación y en el criterio cualitativo del modelo."
)

st.markdown(
    "- **Ajustado** (opcional): un modelo de lenguaje pequeño (Qwen3-4B) ajustado con repartos reales del CIS, el Eurobarómetro y la ESS da el reparto de las preguntas sin equivalente; las repetidas siguen con la estadística directa. "
    "Cuesta menos de un segundo por pregunta, pero necesita un servidor local con GPU; si no responde, se usa el modo rápido. "
    "**No es el modo por defecto**: en la evaluación final de los 88 casos reservados mejora 1,28 puntos frente al rápido, pero el intervalo roza el cero y no cumple el criterio fijado de antemano (ver «Validación»)."
)

st.subheader("Subgrupos y cruces (MRP)")
st.markdown(
    "- **Cuando la pregunta existe en el CIS (modo directo)**, los repartos por grupo salen de un **MRP**: un modelo multinomial con efectos de sexo, edad, comunidad, estudios e ingresos (los que haya en esa oleada) y cuatro variables por comunidad, "
    "con penalización L2 en lugar de MCMC; se ajusta con los microdatos de la oleada de la pregunta, se postestratifica con la población sintética y se recalibra para que el total coincida con el nacional del modo activo.\n"
    "- **Validación** (prerregistrada, 279 pares de preguntas repetidas del CIS, 7.542 subgrupos): el MRP reduce el error por grupo de 3,22 a 2,64 puntos (−0,58, IC 95 % por series [−0,93; −0,32]). Mejora sobre todo la edad; en comunidad autónoma empata. "
    "Solo se validó **entre oleadas de la misma pregunta**, con subgrupos de **una** variable.\n"
    "- **Sin ancla** (pregunta nueva) se mantiene el método anterior (diferencias de preguntas análogas); en el **modo ajustado** los grupos los da el modelo ajustado. La interfaz dice siempre qué método produjo los grupos. "
    "Los rangos de los grupos siguen siendo los del método anterior (conservadores).\n"
    "- **Cruces de varias variables** (por ejemplo, mujeres de 18-24 años de Andalucía): solo con MRP y con pregunta ya existente en el CIS. **Experimentales**: no están validados. Se muestra el número de encuestados del CIS en ese cruce y se avisa si es menor de 50.\n"
    "- Con Ollama, un **plan estructurado** (qwen3:8b, 20 s como máximo) puede proponer opciones, tema y el grupo que pide la pregunta; solo se aceptan variables y valores del esquema del CIS y, si falla, se usan las reglas y los selectores manuales."
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
    "- El CIS pregunta por teléfono o cara a cara; hay sesgos de respuesta que no corregimos.\n"
    "- Modo ajustado: solo preguntas de 2 a 7 opciones; la evaluación en los 88 casos reservados tiene poca potencia (41 casos informativos de 9 estudios); sus rangos quedan algo cortos (86,5 % de cobertura al 90 %)."
)
