# «¿Quizás querías decir…?»: sugerencia de paráfrasis

Cuando la pregunta escrita no tiene un equivalente directo en el CIS (la tarjeta de cobertura sale «parecidas» o «sin datos comparables»), «Analizar pregunta» busca por **similitud de embeddings e5** entre las preguntas del CIS y, si la mejor supera el umbral, la muestra con su estudio, su fecha de campo, sus opciones y un botón **«Usar esta pregunta»** que solo rellena el cuadro de texto: hay que volver a pulsar «Analizar pregunta». No cambia nada automáticamente ni toca el ancla, el prior o ninguna cifra (`src/mavericks/paraphrase.py`).

- **Qué se compara:** el coseno e5 («query: » contra «query: ») entre la pregunta y las 4.207 preguntas de respuesta única del CIS del almacén de evidencia con al menos dos opciones sustantivas. De cada variable del CIS solo entra la oleada más reciente (la que usaría el modo directo). No hay estudios reservados en el almacén.
- **Índice:** se calcula una vez (≈ 2 min en CPU la primera vez; la app lo hace al arrancar) y se guarda en `outputs/cache/paraphrase_*.npz`; después carga en 0,4 s.
- **Opciones compatibles:** al analizar todavía no hay opciones, así que se muestran las del CIS junto a la sugerencia; si el usuario usa la pregunta, esas son las opciones sugeridas del modo directo.

## Calibración del umbral

`python -m mavericks.paraphrase_eval [--detalle]`. Conjuntos:

- **36 positivas:** 30 paráfrasis escritas a mano de preguntas reales del CIS (`tests/data/parafrasis_calibracion.json`) y las 6 de la batería (`subtipo: parafrasis`). Es correcta si la pregunta propuesta es la misma variable del CIS, o contiene el enunciado de la original (la misma pregunta aparece con otro identificador en otros barómetros).
- **52 negativas:** las 40 preguntas «de predicción» de la batería (no existen en el CIS) y 12 preguntas cercanas pero distintas («¿Cree que el precio de los alquileres subirá el año que viene?», frente a la del precio de la vivienda).

| Umbral (coseno) | Sugerencias | Correctas | Incorrectas | Precisión | Cobertura de positivas | Negativas con sugerencia |
|---|---|---|---|---|---|---|
| 0,88 | 60 | 36 | 24 | 60 % | 100 % | 24 |
| 0,90 | 45 | 35 | 10 | 78 % | 97 % | 10 |
| 0,92 | 38 | 35 | 3 | 92 % | 97 % | 3 |
| 0,94 | 29 | 27 | 2 | 93 % | 75 % | 2 |
| **0,95** | **23** | **23** | **0** | **100 %** | **64 %** | **0** |
| 0,96 | 17 | 17 | 0 | 100 % | 47 % | 0 |
| 0,97 | 10 | 10 | 0 | 100 % | 28 % | 0 |

**Umbral elegido: 0,95.** La negativa más parecida llega a 0,947 («¿Qué riesgo de la inteligencia artificial le preocupa más?» frente a «¿Cree Ud. que existen riesgos en el uso de la inteligencia artificial?»), y las positivas están entre 0,89 y 0,985; por debajo de 0,95 empiezan a salir sugerencias de otra pregunta del mismo tema. Preferimos no sugerir a sugerir mal: se pierde un tercio de las paráfrasis (las más lejanas, 0,89–0,95).

## Límites

- Muestra pequeña (36 + 52): el umbral se eligió con los mismos casos con los que se mide; la precisión del 100 % no es una garantía, solo que no hay falsos en este conjunto.
- Los aciertos se juzgaron con criterio algo generoso (una pregunta equivalente de otra serie cuenta como correcta).
- Solo preguntas de respuesta única del CIS (sin Eurobarómetro ni ESS). En las baterías de ítems (enunciado común con un ítem entre corchetes) la sugerencia puede apuntar a otro ítem de la misma batería; no se ha medido.
