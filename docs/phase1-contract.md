# Integración provisional con Fase 1

No se encontró `Accenture.md`. Este documento hace explícitas las decisiones provisionales; debe revisarse contra ese contrato antes de afirmar compatibilidad con los datasets reales.

## Población

Archivos esperados: `synthetic_population_sample.parquet` o `synthetic_population.parquet`, acompañados de `population_metadata.json`. Se acepta CSV. Se conserva toda columna de entrada y no se recodifican categorías.

Mínimo obligatorio para ejecutar: `person_id`, `peso`, `sexo`, `edad`, `grupo_edad`, `nivel_estudios`, `version_poblacion`. El resto de las columnas enumeradas en el prompt es opcional; su ausencia no bloquea el MVP. Edad finita >=18, identificadores únicos, pesos positivos finitos, ausencia de nulos en obligatorias y versión única son requisitos estrictos.

Metadatos mínimos:

```json
{
  "population_version": "VERSION_PROPORCIONADA_POR_FASE1",
  "synthetic": false,
  "variable_provenance": {"sexo": "observed", "ingresos_hogar_tramo": "imputed"},
  "categories": {},
  "coverage": "Adultos en hogares privados; excluye establecimientos colectivos"
}
```

`synthetic` identifica fixtures de demostración, no el hecho de que los registros de Fase 1 sean una población sintetizada estadísticamente. Añade a `categories` los catálogos reales: sin ellos se advierte que la validación categórica no está verificada. Valores extra en metadatos se conservan. Indica `--metadata` si el archivo está en otra ubicación. `population_validation.json` puede conservarse como artefacto de Fase 1; no se interpreta su esquema, aún desconocido.

## Encuestas

`cis_questions`: registros con `question_uid` (alias explícito de `evidence_id`), `estudio_id` (alias de `study_id`), `question_text` literal, `question_summary` abreviada, `options` como lista, y campos conocidos del esquema Evidence. Campos desconocidos se conservan en `metadata.original_fields`. No se inventan textos, fechas, universos o modos ausentes. Para CSV, normaliza previamente listas a JSON/Parquet; no se interpreta una cadena arbitraria como opciones.

`cis_respondents`: `estudio_id`, `respondent_uid`, peso muestral `peso` o `peso_ccaa`, y variables demográficas disponibles.

`cis_responses`: `estudio_id`, `respondent_uid`, `question_uid`, columna de respuesta configurable (`--answer-column`, por defecto provisional `answer`). No se presume que el contrato final denomine así esa columna. Preguntas múltiples: `selected` booleano y todas las opciones por persona; no se usan porcentajes que deban sumar 100%.

La clave compuesta evita mezclar entrevistas de estudios distintos. Duplicados y respuestas huérfanas producen error. El peso de encuestas nunca se usa como peso poblacional. Los resultados derivados contienen pregunta y estudio originales, peso seleccionado, n observado, sumas de pesos y n efectivo de Kish. La fecha procede del registro Evidence, no se deduce del identificador.

La ingesta de microdatos adjunta distribuciones nacionales y por las variables disponibles de edad, sexo y educación. Cruces específicos de segmentos y modos de respuesta mixtos por pregunta requieren configuración adicional conforme al contrato definitivo; no se infieren automáticamente.

## Backtesting

Formato `--cases`:

```json
[
  {
    "evidence_id": "ID_RETENIDO",
    "question": "Texto de la pregunta retenida",
    "options": ["A", "B"],
    "fieldwork_date": "2025-01-01",
    "equivalent_ids": ["DUPLICADO_REVISADO"],
    "equivalent_texts": ["Paráfrasis revisada"],
    "observed_distribution": {"A": 0.5, "B": 0.5}
  }
]
```

Las cifras del ejemplo son ilustrativas, no CIS. Los resultados observados se leen para comparar después de ejecutar la simulación y no entran en payloads LLM. Reserva casos y equivalencias antes de indexar/evaluar. La caché incluye payload, esquema, proveedor, endpoint, temperatura y prompt; los resultados retenidos no se pasan a la caché de inferencia. El backtesting Mock es exclusivamente una prueba del pipeline.
