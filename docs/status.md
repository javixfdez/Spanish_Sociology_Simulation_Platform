# Estado de entrega (rama `javier-picazo`)

## Implementado y verificado con datos reales

- **ETL del CIS** (`etl-cis`): 108 estudios (11 reservados), 7.523 preguntas, 425.588 entrevistados. Fechas de campo de la
  ficha técnica y de los microdatos con aviso de discrepancias (3438: la ficha dice 2023, los microdatos 2024).
  Sin `PESO`: 3503 y 3528. Sin `REGISTRO` único: 3495.
- **Población sintética** (`etl-population`): Censo 2021 → IPF a Cifras de Población (1/1/2026) y EPA 2T26 → n = 50.000,
  semilla 42, ingresos imputados de la ECV 2025. Error máximo de margen CCAA × sexo × edad: 0,1 puntos de cuota.
  Estudios y ingresos coinciden con la EPA y la ECV (`population_validation.json`).
- **Segmentación jerárquica** por pesos: devuelve el número de segmentos pedido (5 → 5, 80 → 80) usando edad, sexo,
  estudios, CCAA, hábitat e ingresos según la prioridad del metodólogo.
- **Ancla empírica**: estadística directa si existe la misma pregunta (probado con CIS 3546 P4, 19,5 s con `llama3`) y prior
  acotado a ±0,15 si es parecida (6 segmentos en 77 s).
- **Backtest con los 11 estudios reservados**: 88 casos generados; corte estricto por fecha de campo, sin reservados en el
  RAG, MAE/JSD/MAE por subgrupo con línea base uniforme y calibración (`calibration.json`).
- Resiliencia: sin evidencia no se llama al evaluador; reintentos informados con temperatura creciente; alternativa
  determinista para evaluador, metodólogo y analista.
- Estabilidad (`--stability-runs`) e intervalos calibrados conectados al auditor.
- Tests: 89 (`uv run pytest -q`), ruff limpio. Los 31 tests originales se mantienen.

## Verificado solo con el Mock o con muestras pequeñas

- Backtest completo de los 88 casos con el **Mock**: sirve para probar el flujo y la calibración, no es una métrica
  predictiva (51 casos en modo directo: MAE 0,028; 37 sin ancla: MAE 0,164; línea base uniforme ≈ 0,17). El modo directo
  equivale a repetir la oleada anterior.
- Backtest real con `llama3` de 4 casos (1 directo MAE 0,026; 3 sin ancla MAE 0,148; uniforme 0,184). Muestra demasiado
  pequeña para concluir.
- `calibration.json` no se ha generado con un LLM real: hacen falta los 88 casos (≈ 45 min con `llama3`). Hasta
  entonces el auditor devuelve «Fiabilidad predictiva todavía no calibrada».

## Pendiente o con riesgo

- Calibración con LLM real y con `--stability-runs` ≥ 3; revisar los casos del backtest (equivalencias humanas).
- `qwen3:8b` no estaba instalado ni se probó; con `llama3` el evaluador de evidencia suele fallar tras los reintentos y
  entra la alternativa determinista (aviso en `warnings`). Con `mistral` fallaron 2 de 5 segmentos en una prueba previa.
- Los enunciados de las preguntas del CIS son la etiqueta de la variable (sin el enunciado completo del cuestionario); en
  baterías de ítems el texto es solo el ítem. Mejorar: extraer el enunciado de los PDF `cues*.pdf`.
- `ingresos_hogar_tramo`: 8 estudios usan otra escala y quedan sin esa variable. `tamuni` solo con la escala de 7 tramos.
- `situacion_laboral` de la población proviene del Censo 2021 sin reponderar.
- El tamaño de municipio > 10.000 habitantes es una estimación a partir de la muestra del Censo.
- Embeddings semánticos (Chroma + sentence-transformers) implementados pero no ejecutados con un modelo real.
- Chunks de documentos grandes, focus group y despliegue no incluidos.

## Cómo reproducir las cifras

```sh
uv run python main.py etl-population && uv run python main.py etl-cis
uv run python main.py ingest-surveys --path data/processed/cis_questions.parquet
uv run python main.py make-backtest-cases
uv run python main.py backtest --provider mock --path data/processed/synthetic_population.parquet --cases data/processed/backtest_cases.json
```
