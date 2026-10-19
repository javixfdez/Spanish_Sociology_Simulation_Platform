# Preregistro — fine-tuning de un LLM pequeño (fijado en git ANTES de evaluar)

**Modelo base:** `Qwen/Qwen3-4B-Instruct-2507` (publicado en julio de 2025). dev_grande (trabajo de campo 2025-07-01 a 2026-03-26) es, con alta
probabilidad, posterior a sus datos de preentrenamiento, lo que favorece la limpieza de la evaluación (el 2025-H1 y el 2024-H2 pueden estar en esos datos y
por eso ya no se usan para evaluar).

**Entrenamiento (corte A'):** ejemplos con trabajo de campo anterior a 2025-07-01 (CIS, Eurobarómetro y ESS filtrados), objetivos siempre reales,
conservando todos los nacionales y como máximo 6 subgrupos por pregunta y dimensión; orden de opciones barajado al vuelo. Sin dev_grande ni nada posterior
al corte en el entrenamiento (assert). La pérdida es KL(reparto real || softmax del primer token restringido a las letras válidas), ponderada por
min(n, 1000)/1000 y ×0,5 en Eurobarómetro y ESS. El corte B (< 2026-04-01) queda para el modelo final si se adopta.

**Evaluación única:** dev_grande, solo reparto nacional, preguntas nuevas (sin pregunta pasada con similitud léxica ≥ 0,90): 703 preguntas de 23 estudios.

**Comparación principal:** «LLM ajustado» frente al «prior de producción (media 50/50 KNN + MLP, el MLP entrenado por mes con su corte)».
Se adopta si la mejora es ≥ 0,5 puntos con el IC 95 % bootstrap POR ESTUDIO entero por debajo de cero.
**Segunda comparación declarada:** «media 50/50 LLM ajustado + prior de producción» frente al prior, con el mismo criterio.

**Referencias en las mismas preguntas:** uniforme, KNN actual, prior de producción y el mismo modelo base sin ajustar con el mismo método de lectura.
**Métricas:** MAE en puntos, IC por estudio, mejoran/empeoran, Wilcoxon, desglose 2 frente a 3 o más opciones y segundos por pregunta.
**Descriptivo:** error por subgrupo del LLM ajustado frente al método actual de subgrupos (desviaciones empíricas contraídas) en las mismas preguntas.
Si algo mejora más de 4 puntos sobre el prior, se busca primero una fuga. Los 88 oficiales no se tocan; si se adopta, tendrá su propia evaluación única.

## Anexo (2026-10-18): modelo B y Kaggle
El modelo B (corte < 2026-04-01) se prepara en paralelo en Kaggle (2×T4, fp16 porque la T4 no admite bf16; resto de la configuración idéntico a a1:
mismo modelo base y revisión `cdbee75f…`, QLoRA nf4 r=8, alpha=32, lr 1e-4, KL ponderada, barajado al vuelo, máx. 6 subgrupos por pregunta y dimensión, semilla 0).
**Solo se usará si a1 cumple el criterio prerregistrado arriba.** Si a1 no lo cumple, B se descarta sin evaluarse. Si a1 lo cumple, B tendrá su única evaluación
prerregistrada sobre los 88 casos oficiales. Diferencias declaradas: fp16 en lugar de bf16, y si las 9,5 h de presupuesto no bastan, se submuestrean solo subgrupos
(la cifra queda en el status.log de Kaggle). El dataset del paquete se comprueba con assert: sin estudios reservados, sin ESS11, fecha de campo < 2026-04-01.

## Anexo (2026-10-19): selección del checkpoint y resultado de los entrenamientos
Escrito ANTES de cualquier evaluación de a1 o b1.

- **Checkpoints.** `save_state` sobrescribía `runs/<name>/adapter` en cada guardado, así que los checkpoints intermedios de a1 no se conservaron. El candidato único
  para evaluar es `a1/adapter_final` (paso 3315, fin de la única época). No hay selección entre checkpoints y, por tanto, no hay sesgo de selección por checkpoint.
  (Para futuros entrenamientos `save_state` guarda también una copia por paso.)
- **Pérdidas de validación** (KL ponderada). Los guardados usan `limit=1000` ejemplos (a1) o los 1000 primeros de la muestra fija (b1); el FIN usa toda la validación
  (a1: 1767 ejemplos de estudios no vistos; b1: 2292), por eso difieren.

  | modelo | paso | validación |
  |---|---|---|
  | a1 | 663 | 0,1458 |
  | a1 | 1326 | 0,1337 |
  | a1 | 1989 | 0,1463 |
  | a1 | 2652 | 0,1377 |
  | a1 | 3315 | 0,1459 |
  | a1 | FIN (completa) | 0,1506 (entrenamiento final 0,0159, media de los últimos 100 pasos) |
  | b1 | 1158 | 0,1623 |
  | b1 | 2316 | 0,1648 |
  | b1 | 3474 | 0,1397 |
  | b1 | 4632 | 0,1417 |
  | b1 | 5790 | 0,1386 |
  | b1 | FIN (completa) | 0,1453 |
- **b1.** Terminó sin submuestreo (5792 pasos), en fp16 en 2×T4, con 4 pasos omitidos por desbordamiento de fp16, todos al inicio. Su script es el de
  `src/mavericks/ft/train_kaggle.py` tras sincronizarlo con la versión ejecutada (opción `--compute`, comprobación numérica sobre las letras válidas, aborto si >20 % de
  pasos omitidos). La versión 1 falló en la comprobación inicial porque miraba también los −inf de las letras no válidas, enmascarados a propósito.
- **Uso de b1.** b1 sigue sin usarse salvo que a1 cumpla el criterio prerregistrado. No se ha descargado ni evaluado.
- **Evaluación de a1.** Se hace una sola vez sobre dev_grande, tal como está descrito arriba; no se cambia nada tras ver resultados.
