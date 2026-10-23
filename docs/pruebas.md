# Batería de pruebas

Batería de unos 150 casos (`tests/battery/preguntas.yaml`, 150 casos y 176 ejecuciones en los tres modos) y su ejecutor (`python -m mavericks.battery run --modes rapido,llm,ajustado`; los resultados van a `outputs/battery/`, fuera de git).
**Solo se arreglaron fallos** (errores, rechazos que faltaban, etiquetas y textos incorrectos): no se cambió ninguna predicción, modelo, rango ni el modo por defecto para que las pruebas salieran mejor.
Nada de estudios reservados: las preguntas reales del CIS de la categoría 1 son de estudios no reservados (trabajo de campo desde 2024-07).

Invariantes comprobados en cada ejecución: porcentajes que suman 100 ± 0,1 sin negativos y con el punto dentro de su rango (nacional, subgrupos, cruces y dato medido); cifras de la explicación coherentes con las calculadas y sin mezclar lo medido con lo estimado;
etiquetas de modo, método de subgrupos y fallback presentes y coherentes; sin N.S./N.C. en lo estimado; tiempo por modo; y, en consulta de dato, **igualdad exacta (1e-12) del bloque medido con un cálculo independiente de pandas sobre los microdatos**.
Un caso «pasa» si cumple todo, «aviso» si hay algo a vigilar que no es un fallo (p. ej., una paráfrasis que no encuentra el ancla directo) y «falla» si rompe un invariante o una expectativa.

## Resultado (última ejecución completa)

| Categoría | Modo | Casos | Pasa | Aviso | Falla | Invariantes sin fallo |
|---|---|---|---|---|---|---|
| consulta_dato | rapido | 20 | 20 | 0 | 0 | 100 % |
| prediccion | rapido | 40 | 40 | 0 | 0 | 100 % |
| robustez | rapido | 25 | 21 | 4 | 0 | 100 % |
| subgrupo_lenguaje_natural | rapido | 15 | 10 | 5 | 0 | 100 % |
| fuera_de_alcance | rapido | 20 | 20 | 0 | 0 | 100 % |
| limite | rapido | 15 | 15 | 0 | 0 | 100 % |
| modos | rapido | 13 | 13 | 0 | 0 | 100 % |
| modos | ajustado | 15 | 15 | 0 | 0 | 100 % |
| modos | llm | 13 | 13 | 0 | 0 | 100 % |

Ejecución: rápido y ajustado con el servidor del modelo ajustado y Ollama a la vez (como se midió, caben en la VRAM); con LLM aparte, con el servidor apagado y la caché del LLM vacía. El modo con LLM y el ajustado solo se ejecutan en la muestra de la categoría 7
(13 y 15 casos), más los rápidos de las otras categorías. El plan del LLM de la categoría 4 se evalúa con Ollama (qwen3:8b) sobre el texto en lenguaje natural.

## Tiempos por modo (solo ejecuciones sin error)

| Modo | Ejecuciones | Mediana (s) | Máximo (s) |
|---|---|---|---|
| ajustado | 15 | 3.7 | 11.1 |
| llm | 13 | 8.8 | 147.6 |
| rapido | 122 | 2.4 | 5.2 |

- Con LLM, sin caché y con la GPU para Ollama solo: las 6 preguntas sin ancla tardaron 89–148 s (mediana 116 s); las que tienen ancla directa, 2–4 s (estadística directa).
- Con el servidor del modelo ajustado cargado a la vez, el modo con LLM fue **mucho más lento** (338–558 s en las preguntas sin ancla de una ejecución previa): con los dos modelos en la VRAM de 8 GB, lo más probable es que Ollama tenga que descargar capas a la CPU (no se ha comprobado la causa). Limitación conocida: no usar los dos a la vez para el modo con LLM completo.

## Lo que se midió sin forzarlo

- **Permutación del orden de las opciones** (la misma pregunta con las opciones invertidas, rotadas o barajadas; el reparto debe moverse con las opciones): diferencia máxima por opción **6.4 pp** (0,0 en las preguntas con ancla directo; 6.4 en el caso sin ancla con escala de acuerdo). Casos: R3-07 0.0, R3-08 0.0, R3-09 0.0, R3-10 6.4, R3-11 0.0, R3-12 4.8.
- **Preguntas negadas** (¿a favor de…? / ¿en contra de…?): R3-14 NO invierte (Sí base 58 % → Sí en la negada 50 % (no invierte el sentido)), R3-16 invierte (Sí base 68 % → Sí en la negada 49 % (invierte el sentido)). El sistema no entiende la negación: en una pregunta sin ancla, negar el enunciado puede no invertir el reparto.
- **Paráfrasis de preguntas del CIS:** 2 de 6 mantienen el ancla directo; las otras 4 caen al modo prior (pregunta parecida) con una diferencia de 4–9 puntos de TVD respecto a la pregunta original. Erratas, mayúsculas y sin tildes: el ancla se mantiene en 4 de 6 casos (las otras dos son preguntas sin ancla de partida).
- **Pregunta en inglés:** no falla; se trata como pregunta nueva sin ancla (sin traducir).

## Fallos encontrados y arreglados

- `0896127 — ronda 1: reglas de alcance que faltaban (elecciones futuras, personas concretas, otros países por gentilicio, periodo posterior a junio de 2026, datos personales, enunciados ofensivos, inyecciones de instrucciones), N.S./N.C. introducidas como opción ahora se ignoran con aviso, texto sin letras o solo espacios rechazado, avisos de pregunta muy larga o con varias preguntas, «Según el CIS, …» ya no impide encontrar el ancla directo, y falso positivo del comprobador de explicaciones (la palabra «primaria» de una opción tomada por el grupo de estudios)`
- `c369944 — la batería: rechazar una inyección es un acierto`
- `0fa0026 — el plan del LLM avisa cuando traduce un grupo vago («jóvenes», «mayores», «ingresos altos») a un valor concreto`
- `83a986a — ronda 2: el juicio «hechos» del LLM ya no rechaza preguntas sobre el comportamiento del encuestado («¿Dónde compra usted…?»); regla «¿qué día prefiere…?» ya no es un dato objetivo`
- `0ceeadf — ronda 3: tampoco rechaza experiencias personales («¿Ha tenido ansiedad o tristeza…?», «¿Ha comprado…?»)`

Antes de los arreglos, la primera ejecución en modo rápido tenía **15 fallos de 148** (13 de la categoría de fuera de alcance, 1 caso límite y 1 falso positivo del comprobador). Tras la ronda 1, 0 fallos en rápido; las rondas 2 y 3 cerraron los fallos que aparecieron al añadir los modos con LLM y ajustado.

## Los 10 casos más raros

1. **C1-15** [rapido] «¿Ud. considera útil que le envíen el aviso a su móvil en el caso de una emergencia extraor» → Lo considera útil — 95 % en una sola opción. Explicación del sistema: La opción más probable es «Lo considera útil», con un 95 %; rango habitual 95–96 %. Se apoya en la propia pregunta del CIS (estudio 3527, 2025-09-12), con su resultado real más reciente. Dato medido por el CIS (estudio 3527, campo 2025-09-12, 2982 encuestados)
2. **C2-05** [rapido] «¿En qué tipo de vivienda le gustaría vivir dentro de diez años?» → Piso en el centro — tres opciones con el mismo porcentaje. Explicación del sistema: La opción más probable es «Piso en el centro», con un 11 %; rango habitual 3–20 %. Es una pregunta nueva, sin equivalente en el CIS; la estimación se apoya en preguntas parecidas del CIS y tiene confianza baja. No hay dato medido: la pregunta no se ha hecho en
3. **C2-19** [rapido] «¿Para qué usaría principalmente un asistente de inteligencia artificial?» → Buscar información — tres opciones con el mismo porcentaje. Explicación del sistema: La opción más probable es «Buscar información», con un 11 %; rango habitual 3–20 %. Es una pregunta nueva, sin equivalente en el CIS; la estimación se apoya en preguntas parecidas del CIS y tiene confianza baja. No hay dato medido: la pregunta no se ha hecho e
4. **C2-35** [rapido] «¿Ha comprado ropa de segunda mano en el último año?» → No — 91 % en una sola opción. Explicación del sistema: La opción más probable es «No», con un 91 %; rango habitual 76–100 %. Es una pregunta nueva, sin equivalente en el CIS; la estimación se apoya en preguntas parecidas del CIS y un modelo aprendido de todas las preguntas anteriores y tiene confianza baja. No hay
5. **L6-04** [rapido] «¿Cuál es su número favorito del uno al quince?» → 1 — tres opciones con el mismo porcentaje. Explicación del sistema: La opción más probable es «1», con un 7 %; rango habitual 0–14 %. Es una pregunta nueva, sin equivalente en el CIS; la estimación se apoya en preguntas parecidas del CIS y tiene confianza baja. No hay dato medido: la pregunta no se ha hecho en el CIS.
6. **L6-07** [rapido] «¿Cree usted que la situación económica, social y política del país, la situación económica» → No — 90 % en una sola opción. Explicación del sistema: La opción más probable es «No», con un 90 %; rango habitual 74–100 %. Es una pregunta nueva, sin equivalente en el CIS; la estimación se apoya en preguntas parecidas del CIS y un modelo aprendido de todas las preguntas anteriores y tiene confianza baja. No hay
7. **N4-01** [rapido] «¿Cómo describiría Ud. su estado de salud en general: muy bueno, bueno, regular, malo o muy» → Bueno — sin motivo concreto. Explicación del sistema: La opción más probable es «Bueno», con un 50 %; rango habitual 49–52 %. Las mayores diferencias entre grupos: estudios sin_estudios (37 %, -14 puntos); edad 75+ (39 %, -12 puntos) (método: MRP (modelo multinivel con los microdatos de la oleada de la pregunta))
8. **N4-02** [rapido] «Según el CIS, cambiando de tema, ¿cómo calificaría Ud. su situación económica personal en » → Buena — sin motivo concreto. Explicación del sistema: Dato medido por el CIS (estudio 3546, campo 2026-03-06, 3998 encuestados): «Buena» 61 % (±2.1 puntos). En Estudios: primaria el CIS midió «Buena» 49 % (±10.1 puntos, 109 encuestados). Estimación del modelo para hoy: la opción más probable es «Buena», con un 61
9. **N4-03** [rapido] «El presidente del Gobierno, Pedro Sánchez, ¿le inspira a Ud., personalmente, mucha confian» → Ninguna confianza — sin motivo concreto. Explicación del sistema: La opción más probable es «Ninguna confianza», con un 47 %; rango habitual 45–49 %. Las mayores diferencias entre grupos: comunidad autónoma País Vasco (30 %, -17 puntos); comunidad autónoma Cataluña (33 %, -14 puntos) (método: MRP (modelo multinivel con los m
10. **N4-10** [rapido] «Según el CIS, cambiando de tema, ¿cómo calificaría Ud. su situación económica personal en » → Buena — sin motivo concreto. Explicación del sistema: Dato medido por el CIS (estudio 3546, campo 2026-03-06, 3998 encuestados): «Buena» 61 % (±2.1 puntos). En Comunidad autónoma: Castilla y León el CIS midió «Buena» 66 % (±9.0 puntos, 203 encuestados). Estimación del modelo para hoy: la opción más probable es «B

## Regresión frente a la configuración oficial

Comprobación antes de congelar: el reparto **nacional** (modo rápido, sin Ollama ni GPU) de las mismas preguntas calculado con el código de la configuración congelada de la evaluación oficial (`3cb71f0`, en un worktree aparte) y con el código actual (`HEAD`),
leyendo los mismos datos y modelos. Preguntas: las 20 de «consulta de dato» y las 40 de «predicción» de la batería, más 50 casos de desarrollo no reservados (`data/processed_v2/backtest_cases_dev.json`, tomados a intervalos regulares). No se han usado los 88 oficiales ni estudios reservados.
Tolerancia 1e-9 sobre cada porcentaje; también se exige el mismo modo y las mismas opciones.

| Grupo | Preguntas | Coinciden exactamente | Distintas |
|---|---|---|---|
| Consulta de dato (batería) | 20 | 19 | 1 |
| Predicción (batería) | 40 | 40 | 0 |
| Desarrollo no reservado | 50 | 45 | 5 |
| **Total** | **110** | **104** | **6** |

**Las 6 que no coinciden son intencionadas:**

1. **C1-08** (prefijo «Según el CIS, …»): con `3cb71f0` el prefijo impedía encontrar la pregunta repetida y salía como pregunta sin ancla; ahora se quita para buscar y sale en modo directo (diferencia máxima 3,3 pp).
2. **5 casos de desarrollo con una opción no sustantiva escrita por el usuario** (`N.R.` en los tres de participación electoral y en la consulta P5C; `N.S./N.R.` en CIS3531_P5D_3): las opciones N.S./N.C./N.R. nunca se estiman, así que se ignoran con aviso y el reparto se renormaliza sin ellas
   (p. ej., «Votó» 80,1 % → 80,7 %). El modo sigue siendo directo.

Ninguna pregunta cambió por las reglas de alcance nuevas, el MRP no altera ningún reparto nacional (solo toca subgrupos) y **el prior de las preguntas sin ancla era idéntico** en la comprobación del 19-10: las 40 de predicción coincidían (tras reconstruir el entorno desde `uv.lock` coinciden 39 de 40; ver la nota del 22-10 más abajo). Lo blinda `tests/test_prior_unchanged.py` (6 preguntas sin ancla, entre ellas C2-28, con valores calculados con `3cb71f0`, a 1e-9).

### Corrección posterior a la evaluación oficial (N.S./N.R.)

Al comparar apareció un fallo de fondo de `anchor.py`: la búsqueda del ancla solo toleraba como opciones extra de la pregunta del CIS N.S., N.C., «No sabe», «No contesta» y N.P.; no reconocía «N.S./N.R.», «N.R.», «No responde» ni «No sabe/No contesta». Por eso **unas 138 preguntas del CIS no reservadas
(de 6.829, ≈ 2 %) con «N.S./N.R.» entre sus opciones solo se encontraban como ancla si el usuario escribía también esa opción; con «Sí / No», lo normal, nunca**, tampoco en `3cb71f0`. Se corrige con una única función compartida (`anchor.is_non_substantive`, la misma que usa `assist.is_non_substantive` para las opciones del usuario)
que reconoce todas las variantes (con o sin puntos, mayúsculas o tildes). Ahora esas preguntas se responden con el dato directo (p. ej., CIS3531_P5D_3 con «Sí / No»: antes modo sin ancla, Sí 37,6 %; ahora modo directo, Sí 20,3 %).

- La tolerancia ampliada se aplica **solo a la búsqueda del ancla** (`option_mapping` por defecto). El prior por analogía (`analog.py`), la evaluación (`evaluation.py`) y el conjunto prerregistrado del MRP (`mrp_eval.py`) siguen con el criterio de `3cb71f0` (`legacy_extras=True`). Una primera versión del arreglo, sin esta separación, cambiaba el prior de C2-28 en 0,6 pp; se detectó con esta regresión y se corrigió antes de publicar.
- **La evaluación oficial (modo rápido 8,11 pp, modo con LLM 7,91 pp) se hizo con el comportamiento anterior y no se repite.** Esta corrección es posterior a ella: no cambia sus cifras ni su configuración congelada.

**Batería en modo rápido tras la corrección:** 148 ejecuciones, 144 pasan, 4 avisos (paráfrasis que dejan de ser directas) y 0 fallos; el reparto nacional, el modo y la etiqueta de las 148 son idénticos a los de la ejecución anterior a la corrección (los cinco casos de lenguaje natural que antes marcaban aviso por el plan del LLM ahora salen como «plan no evaluado», porque esta ejecución no usa Ollama).

Condiciones: el worktree de `3cb71f0` usó los mismos `data/processed*` y `outputs/` (enlazados), de modo que cualquier diferencia viene del código. Las comparaciones son del reparto nacional en modo rápido; no cubren subgrupos (ahí el MRP sí cambia las cifras a propósito) ni los modos con LLM y ajustado, que no existían en la configuración congelada.

### Nota del 22-10: regresión repetida con el entorno reconstruido desde `uv.lock`

Repetida la misma regresión (mismas 110 preguntas, mismos datos y modelos, tolerancia 1e-9) con el código actual y el entorno reconstruido desde `uv.lock`, da **103 iguales de 110** (la tabla de arriba, del 19-10, no se modifica). Las 6 diferencias intencionadas siguen idénticas, con los mismos tamaños, y aparece una séptima: **C2-25** («¿Quién cree que debería hacer más contra el cambio climático?», pregunta sin ancla), con 2,94 pp de diferencia máxima en el reparto nacional; las de predicción coinciden por tanto 39 de 40.

- **No la causa el código:** `HEAD` y los commits anteriores (hasta `a62f4ad`) dan exactamente las mismas cifras con los datos y el entorno actuales.
- **Es el entorno numérico:** numpy, scipy, sentence-transformers y tokenizers del `.venv` local se reinstalaron el 21-10; `uv.lock` no cambió (es idéntico al de `3cb71f0`), de modo que el entorno anterior no coincidía con el del lock. La causa exacta (probablemente el cálculo de los embeddings e5) no se ha investigado.
- **Un clon limpio reproduce las cifras actuales:** un clon nuevo con los datos del paquete (sin estudios reservados) da el mismo reparto en las 148 ejecuciones de la batería en modo rápido que la carpeta local, y 47,5 / 28,2 / 8,2 / 9,2 / 6,9 en C2-25 (antes, 45,9 / 26,4 / 8,4 / 12,1 / 7,2).
- **La evaluación oficial no se ha repetido** y sus cifras (8,11 pp en modo rápido, 7,91 pp con LLM) no cambian; se calcularon con el entorno anterior.
- `tests/test_prior_unchanged.py` sigue pasando (sus 6 preguntas no incluyen C2-25).

## Limitaciones conocidas (no se arreglan porque implicarían tocar modelos o predicciones)

- **Empates exactos y reparto casi uniforme con 8 o más opciones** (p. ej., tres opciones al 11 %): la lectura del prior no distingue opciones sin información; el modo ajustado, además, devuelve logits en bf16, y los empates son por resolución.
- **Paráfrasis fuera de umbral:** una pregunta parecida a una del CIS con otra redacción puede caer al modo prior (similitud < 0,85) y dejar de mostrar el dato medido.
- **Negación:** el sistema no invierte el sentido de forma fiable en preguntas sin ancla.
- **Preguntas en inglés** o con muchas erratas se tratan como preguntas nuevas.
- **Modo con LLM:** de 100 a 135 s por pregunta sin ancla con la GPU libre, y varios minutos si el servidor del modelo ajustado está cargado a la vez.
- **Plan del LLM:** puede tardar más de 20 s la primera vez (carga del modelo; entonces se usan los selectores) y traduce «jóvenes» o «mayores» a un valor concreto; ahora lo avisa, pero la suposición sigue siendo del modelo.
- **Cruces de varias variables:** los de dos variables (sexo×edad, edad×CCAA, estudios×edad) están validados entre oleadas (MRP 3,65 pp frente a 7,00 del reparto nacional, ver `docs/mrp_resultados.md`); los de tres o más siguen sin validar y con n pequeño el dato medido es poco fiable.
- **Patrón `_INTERROGATIVES` (`assist.py`):** contiene dos caracteres de retroceso donde debería haber `\b` (desde el commit `7df2202`), por lo que nunca casa y `closed_kind` solo decide por «o»/escalas; corregirlo cambiaría las opciones sugeridas de algunas preguntas, así que se deja como está.
- **Sugerencia de paráfrasis («¿Quizás querías decir…?»):** el umbral (coseno e5 0,95) se calibró con 36 paráfrasis y 52 negativas (muestra pequeña, mismo conjunto para elegir y medir); ver `docs/parafrasis.md`.
