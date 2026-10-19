# Datos y modelos fuera de git (inventario)

Lo que la app necesita además del código. Nada de esto está en git (`data/processed*`, `outputs/` y `maverick_data/` están ignorados). Se obtiene de dos formas:
**descargándolo** (`mavericks fetch-assets` y `mavericks fetch-adapter`, desde los repositorios privados de Hugging Face `JaviPica/mavericks-assets` y `JaviPica/mavericks-qwen3-4b-cis`)
o **regenerándolo** con los ETL del README a partir de `maverick_data/raw` (que no se redistribuye). El paquete lo crea `mavericks pack-assets` (versión 1, 2026-10-19 15:37); su `manifest.json`
lleva ruta, tamaño y sha256 de cada archivo.

## 1. Modo rápido (por defecto): sin Ollama ni GPU

Archivos del paquete de ejecución (81.7 MB sin comprimir; el zip pesa unos 43 MB):

| Archivo | Tamaño | Para qué sirve | Fuente |
|---|---|---|---|
| `data/processed/synthetic_population.parquet` | 528 KB | Población sintética de 50.000 adultos (segmentos y cuotas de los subgrupos) | Generada con etl-population: Censo 2021, Cifras de Población 1/1/2026, EPA 2T26 y ECV 2025 (INE) |
| `data/processed/population_metadata.json` | 4 KB | Metadatos y procedencia de la población sintética | etl-population |
| `data/processed/population_targets.json` | 1 KB | Objetivos de la ECP para la página de población | population-targets (INE) |
| `data/processed/population_validation.json` | 3 KB | Validación de márgenes de la población sintética | etl-population (INE) |
| `data/processed/cis_respondents.parquet` | 4.2 MB | Entrevistados del CIS (variables sociodemográficas y peso) para las diferencias por grupo | CIS, microdatos de estudios NO reservados, procesados con etl-cis |
| `data/processed/cis_responses.parquet` | 43.5 MB | Respuestas del CIS por entrevistado y pregunta (modo directo y subgrupos) | CIS, microdatos de estudios NO reservados, procesados con etl-cis |
| `data/processed_v2/evidence_v2.sqlite` | 29.7 MB | Almacén de evidencia (preguntas, opciones y distribuciones del CIS, Eurobarómetro y ESS de España) | ingest-surveys sobre CIS y etl-external (agregados), sin reservados |
| `data/processed_v2/cis_national.json` | 2.3 MB | Distribución nacional de cada pregunta del CIS (prior por analogía) | build-national (CIS no reservado) |
| `outputs/models/prior_mlp.pkl` | 708 KB | Modelo aprendido del prior (MLP sobre embeddings e5) | train-prior (CIS no reservado) |
| `outputs/models/prior_mlp.json` | 1 KB | Metadatos del modelo del prior (corte, preguntas, hash del almacén) | train-prior |
| `outputs/models/prior_intervals.json` | 3 KB | Cuantiles de los rangos del prior por número de opciones | calibrate-prior (dev_grande y confirmaciones) |
| `outputs/models/prior_validation.json` | 9 KB | Tabla «Validación con datos no vistos» y error típico del prior | calibrate-prior |
| `outputs/fast_validation.json` | 75 KB | Validación del modo rápido (153 casos) y rangos por modo | fast-validation |
| `outputs/baseline_v1/calibration.json` | 232 KB | Calibración de la confianza por modo (errores históricos del backtest) | backtest sobre los casos oficiales y de desarrollo |
| `outputs/baseline_v1/backtest.json` | 520 KB | Resultados del backtest con LLM (página Validación) | backtest (agregados por caso) |
| `outputs/case_floors.json` | 20 KB | Suelo de ruido por caso (página Validación) | noise-floor |
| `outputs/final_eval/report.json` | 2 KB | Resumen de la evaluación final oficial (página Validación) | final_eval (una sola ejecución) |

Además, fuera del paquete:

| Qué | Tamaño | Para qué sirve | Fuente |
|---|---|---|---|
| Modelo `intfloat/multilingual-e5-base` en la caché de Hugging Face | 1,1 GB | Embeddings de la pregunta para el prior aprendido (se carga con `local_files_only`) | Hugging Face (descarga única, ver README) |
| `data/examples/ejemplos_demo.json` | pequeño | Ejemplos precalculados de la barra lateral | Está en git |

## 2. Modo con LLM

Los mismos archivos del modo rápido más **Ollama** con `qwen3:8b` (5,2 GB, `ollama pull qwen3:8b`, Ollama 0.40.6 con KV q8_0 y atención flash). No hay archivos propios adicionales
(la caché de respuestas del LLM, `outputs/cache`, se crea sola y no es necesaria).

## 3. Modo ajustado (experimental)

Los mismos archivos del modo rápido, más:

| Archivo | Tamaño | Para qué sirve | Fuente |
|---|---|---|---|
| `data/processed/synthetic_population.parquet` | 528 KB | Población sintética de 50.000 adultos (segmentos y cuotas de los subgrupos) | Generada con etl-population: Censo 2021, Cifras de Población 1/1/2026, EPA 2T26 y ECV 2025 (INE) |
| `data/processed/population_metadata.json` | 4 KB | Metadatos y procedencia de la población sintética | etl-population |
| `data/processed/population_targets.json` | 1 KB | Objetivos de la ECP para la página de población | population-targets (INE) |
| `data/processed/population_validation.json` | 3 KB | Validación de márgenes de la población sintética | etl-population (INE) |
| `data/processed/cis_respondents.parquet` | 4.2 MB | Entrevistados del CIS (variables sociodemográficas y peso) para las diferencias por grupo | CIS, microdatos de estudios NO reservados, procesados con etl-cis |
| `data/processed/cis_responses.parquet` | 43.5 MB | Respuestas del CIS por entrevistado y pregunta (modo directo y subgrupos) | CIS, microdatos de estudios NO reservados, procesados con etl-cis |
| `data/processed_v2/evidence_v2.sqlite` | 29.7 MB | Almacén de evidencia (preguntas, opciones y distribuciones del CIS, Eurobarómetro y ESS de España) | ingest-surveys sobre CIS y etl-external (agregados), sin reservados |
| `data/processed_v2/cis_national.json` | 2.3 MB | Distribución nacional de cada pregunta del CIS (prior por analogía) | build-national (CIS no reservado) |
| `outputs/models/prior_mlp.pkl` | 708 KB | Modelo aprendido del prior (MLP sobre embeddings e5) | train-prior (CIS no reservado) |
| `outputs/models/prior_mlp.json` | 1 KB | Metadatos del modelo del prior (corte, preguntas, hash del almacén) | train-prior |
| `outputs/models/prior_intervals.json` | 3 KB | Cuantiles de los rangos del prior por número de opciones | calibrate-prior (dev_grande y confirmaciones) |
| `outputs/models/prior_validation.json` | 9 KB | Tabla «Validación con datos no vistos» y error típico del prior | calibrate-prior |
| `outputs/fast_validation.json` | 75 KB | Validación del modo rápido (153 casos) y rangos por modo | fast-validation |
| `outputs/baseline_v1/calibration.json` | 232 KB | Calibración de la confianza por modo (errores históricos del backtest) | backtest sobre los casos oficiales y de desarrollo |
| `outputs/baseline_v1/backtest.json` | 520 KB | Resultados del backtest con LLM (página Validación) | backtest (agregados por caso) |
| `outputs/case_floors.json` | 20 KB | Suelo de ruido por caso (página Validación) | noise-floor |
| `outputs/final_eval/report.json` | 2 KB | Resumen de la evaluación final oficial (página Validación) | final_eval (una sola ejecución) |
| `outputs/models/ft_intervals.json` | 1 KB | Rangos del modo ajustado (calibración Mondrian con los residuos de a1 en dev_grande) | python -m mavericks.ft.artifacts |
| `outputs/models/ft_validation.json` | 1 KB | Error típico del modo ajustado en dev_grande | python -m mavericks.ft.artifacts |
| `outputs/ft/runs/b1/adapter_final/` (`adapter_config.json`, `adapter_model.safetensors`) | 63 MB | Adaptador LoRA B (sha256 `f6a4dd94…`) | `mavericks fetch-adapter` (repo `JaviPica/mavericks-qwen3-4b-cis`) |
| `Qwen/Qwen3-4B-Instruct-2507`, revisión `cdbee75f…`, en la caché de Hugging Face | 7,6 GB | Modelo base | Hugging Face (descarga única, ver `docs/finetuning.md`) |
| Entorno `.venv_ft` (torch, transformers, peft, bitsandbytes) | varios GB | Servidor del modelo (GPU NVIDIA con ~3 GB libres) | `docs/finetuning.md` |

## MRP (subgrupos y cruces)

No necesita archivos nuevos: usa `cis_respondents.parquet`, `cis_responses.parquet` y `synthetic_population.parquet` del paquete. Guarda una caché por pregunta en `outputs/cache/mrp/*.npz` (probabilidades por persona, ≈ 1 MB;
se crea sola, no se versiona ni se empaqueta). Por eso el paquete sigue siendo `mavericks_assets_v1.zip`.

## Lo que NO es necesario para ejecutar la app

`maverick_data/` (6,6 GB de originales), `data/processed/cis_questions.parquet` y `cis_studies.parquet` (solo para regenerar), `data/processed_v2/` embeddings y cachés de análogos, `data/processed_v2/frozen/`
(evidencia congelada de la evaluación final), `outputs/prior_v2/` (investigación del prior: la app solo importa su código), `outputs/ft/` salvo el adaptador, y `outputs/final_eval*`, `outputs/demo`.

## Qué se excluye del paquete y por qué

- `maverick_data/` y los ficheros originales del Eurobarómetro (GESIS) y la ESS: el paquete solo lleva agregados procesados (`pack-assets` falla si encuentra rutas de `maverick_data` o formatos `.sav`/`.dta`/`.zip`).
- Los **estudios reservados** (publicados desde 2026-04-01): se eliminan de los microdatos del CIS y de la tabla nacional, y se comprueba que el almacén de evidencia no tiene ninguno. Filtros de la versión 1:
  - `data/processed/cis_respondents.parquet`: 41248 filas de 11 estudios reservados eliminadas (384340 conservadas)
  - `data/processed/cis_responses.parquet`: 3080993 filas de 11 estudios reservados eliminadas (28223350 conservadas)
  - `data/processed_v2/cis_national.json`: 663 preguntas de estudios reservados eliminadas (6536 conservadas)
- Regla del reto: ningún dato publicado después del 30/06/2026.
