# Estado de entrega

## Implementado y verificado

- Entorno `.venv` con uv y `uv.lock`, paquete instalable y CLI.
- StateGraph completo con Send, reducer de predicciones, concurrencia configurable, timeout, reintentos y caché.
- Población CSV/Parquet, validadores, metadatos/procedencia, segmentación determinista y perfiles heterogéneos.
- Agentes metodólogo, intérprete, respondente y analista con esquemas Pydantic.
- Mock explícito DEMO / SYNTHETIC; no hay fallback silencioso.
- Agregación ponderada, subgrupos soportados y bloqueo de estimación nacional ante fallos.
- Ingesta JSON/CSV/Parquet/XLSX/PDF, idempotencia, SQLite y TF-IDF con metadatos.
- Microdatos: pesos nacionales/autonómicos explícitos, unión por estudio/persona, NS/NC, respuestas múltiples y Kish.
- Estadísticas calculadas antes de suministrar evidencia al LLM.
- Auditor descriptivo sin puntuación inventada; MAE, JSD, utilidad de intervalos y backtesting con exclusiones/corte temporal.
- Streamlit básico, resultado nacional, segmentos, evidencia, confianza, advertencias y descarga JSON.

## Verificado mediante dobles; requiere servicio/modelo real

- ChatOllama con structured output JSON Schema: pruebas sin servidor, incluyendo indisponibilidad. Ollama no responde en localhost:11434 y no se encontró ejecutable en PATH.
- No se han descargado modelos LLM ni embeddings. No se verificó GPU: consulta de hardware denegada.

## Implementado pero no ejecutado con dependencias opcionales

- sentence-transformers + Chroma persistente y combinación con recuperación léxica: requiere extra semantic y modelo local.
- SAV/SPSS con pyreadstat: requiere extra spss.
- API externa compatible con OpenAI: requiere extra external y credenciales.

## Pendiente de datos y validación empírica

- Revisar contrato definitivo Accenture.md y categorías reales; no estaban disponibles.
- Integrar población Fase 1 y datos CIS reales; verificar nombres y opciones originales.
- Preguntas múltiples y únicas mezcladas en una misma ingesta necesitan ingestiones/configuración por tipo.
- Estudios: configurar pesos según metodología; no se infiere automáticamente peso frente a peso_ccaa.
- Calibración real, error por tema/segmento e intervalos validados. No se publican métricas predictivas a partir de la demo.
- Repeticiones de estabilidad por temperatura/paráfrasis y revisión humana de equivalencias históricas.
- Indexación por chunks para documentos grandes y optimización del índice léxico: el MVP recalcula TF-IDF y carga registros recuperables en memoria.
- Interpretar population_validation.json cuando llegue su contrato.
- Focus group futuro, clustering avanzado y despliegue no incluidos.

## Evidencias de ejecución

Suite pytest: 31 pruebas, sin llamadas LLM reales. Comprende integración del grafo secuencial/concurrente, integridad, ingesta heterogénea, backtesting y comprobación de arranque Streamlit. Ruff sin errores.

Demo CLI: `outputs/demo.json`, estado completed, cobertura 1.0, 3 segmentos para máximo 5, todas las etapas ejecutadas y salida marcada synthetic. Backtesting de prueba: `outputs/backtest-demo.json`; evidencia objetivo excluida. Estas cifras verifican el software, no la representatividad ni precisión.

`check-model`: error claro indicando que Ollama no es accesible; no se ejecutó inferencia real.
