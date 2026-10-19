# Modelos de lenguaje del proyecto y solape con los datos de evaluación

Tabla para responder a la pregunta «¿puede algún modelo haber visto los datos con los que lo evaluamos?». Se parte de una regla sencilla: **un modelo no puede haber visto datos publicados después de su
lanzamiento**; la fecha de corte de entrenamiento es anterior o igual a la de publicación. Cuando no conozco con seguridad una fecha, pone «sin confirmar»: ni los autores de Qwen, Gemma ni Mistral publican un corte
exacto de forma verificable aquí.

Ventanas de evaluación: **dev_grande** = trabajo de campo desde 2025-07 hasta 2026-03 (estudios del CIS usados para desarrollar y validar); **88 oficiales** = estudios reservados, publicados desde 2026-04-01.
Equipo: Ollama 0.40.6 (KV q8_0, atención flash, paralelo 2). Las fechas de publicación salen de la fecha de creación del repositorio del modelo en Hugging Face (consultada el 2026-10-19), no de una nota oficial de lanzamiento.

| Modelo | Para qué se usa | Fecha de publicación | Versión / digest de Ollama | ¿Puede solapar con dev_grande (desde 2025-07)? | ¿Puede solapar con los 88 oficiales (desde 2026-04)? |
|---|---|---|---|---|---|
| **Qwen3-4B-Instruct-2507** (`Qwen/Qwen3-4B-Instruct-2507`, revisión `cdbee75f…`) | Base del fine-tuning (modo ajustado: adaptadores a1 y B). No pasa por Ollama; se carga con transformers y cuantización nf4 | Julio de 2025 según el nombre «2507»; repositorio creado el 2025-08-05 (última modificación 2025-09-17); fecha exacta de lanzamiento sin confirmar | No aplica (no está en Ollama) | **Poco probable pero no descartable** para los primeros estudios de la ventana (2025-07/08); corte exacto sin confirmar | No: se publicó antes de abril de 2026 |
| **qwen3:8b** | Modo «Con LLM» (números 0,3·LLM + 0,7·prior y redacción de la explicación); modelo elegido | Qwen3-8B: repositorio creado el 2025-04-27 (familia Qwen3, abril de 2025); corte de entrenamiento sin confirmar | digest `500a1f067a9f` (8,2 B, Q4_K_M); descargado en este equipo el 2026-10-17 | No: se publicó antes de la ventana | No |
| **qwen3.5:9b** | Alternativa comparada con qwen3:8b y **descartada** (la elección usó los 41 casos no repetidos de la evaluación oficial; el prior, w y los rangos no) | Qwen3.5-9B: repositorio creado el 2026-02-27; corte de entrenamiento sin confirmar | digest `56671c2ab938` (9,0 B, Q4_K_M); descargado el 2026-10-18 | **Sí, posible**: se publicó dentro de la ventana de dev_grande y su corte podría llegar a ella (sin confirmar) | No: se publicó antes de abril de 2026 |
| **gemma4:12b** | Alternativa comparada y **descartada** | Familia Gemma 4: repositorios en Hugging Face desde 2026-03-02 (E4B) y 2026-03-11 (26B-A4B y 31B); la variante de 12 B que sirve Ollama, **fecha y origen sin confirmar** | digest `6114515d63c1` (11,9 B, Q4_K_M); descargado el 2026-10-18 | **Sí, posible** (publicado dentro de la ventana, corte sin confirmar) | Improbable (lanzamiento de la familia en marzo de 2026), pero **sin confirmar** para la variante 12 B |
| **llama3:latest** | Primeras pruebas del pipeline con LLM (README y `docs/status.md`); sustituido por qwen3:8b | Llama 3 8B: repositorio creado el 2024-04-17; corte sin confirmar | digest `365c0bd3c000` (8,0 B, Q4_0) | No | No |
| **mistral:latest** | Una prueba previa (más lento y falló más a menudo); **descartado** | Mistral 7B; la versión exacta de la etiqueta `latest` está sin confirmar (la Instruct v0.3 tiene el repositorio creado el 2024-05-22) | digest `6577803aa9a0` (7,2 B, Q4_K_M) | No | No |
| `intfloat/multilingual-e5-base` (no es un LLM generativo) | Embeddings e5 para el KNN y el MLP del prior | Repositorio creado el 2023-05-19 | No está en Ollama | No | No |

Instalados en el equipo pero **sin registro de uso en el proyecto** (no se evaluaron ni se usaron): `llama2:latest` (`78e26419b446`), `deepseek-r1:32b` (`edba8017331d`), `gemma4:26b` (`5571076f3d70`) y `qwen-local:latest` (`dbaf6dd33b99`).

## Lectura

- Los modelos que **ponen los números** de la demo y de la evaluación final (qwen3:8b en el modo con LLM; Qwen3-4B-Instruct-2507 en el modo ajustado) se publicaron antes de abril de 2026, así que no pudieron ver los 88 estudios reservados.
- Solo Qwen3-4B-Instruct-2507 podría rozar el inicio de dev_grande (julio-agosto de 2025). Para ese modelo se comprobó además que ningún estudio de dev_grande entrara en el ajuste (ver `docs/finetuning.md`),
  pero no se puede descartar que su preentrenamiento incluya alguna encuesta de esas semanas (el corte exacto de Qwen no es público).
- qwen3.5:9b y gemma4:12b se descartaron; si se retomaran, habría que tratar dev_grande como posiblemente visto por ellos.
- Para cerrar los «sin confirmar», consulta la ficha o el informe técnico de cada modelo: no se han inventado fechas.
