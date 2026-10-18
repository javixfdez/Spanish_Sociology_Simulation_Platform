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
