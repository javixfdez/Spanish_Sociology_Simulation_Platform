# Contrato de datos Fase 1 → Fase 2

Este contrato lo producen los comandos `etl-population` y `etl-cis` a partir de `maverick_data/` (ver `README.md`).
Todo se escribe en `data/processed/` (ignorado por git). Los datos publicados después del 30/06/2026 no se usan y
los estudios con `reservado_examen = si` solo sirven para el backtest.

## Esquema común de categorías (el del CIS)

Definido en `src/mavericks/categories.py`; la población y los entrevistados del CIS se recodifican al mismo esquema.

| Variable | Categorías |
|---|---|
| `sexo` | hombre, mujer |
| `grupo_edad` | 18-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75+ |
| `nivel_estudios` | sin_estudios, primaria, secundaria_1, secundaria_2, fp, superiores (+ `otros`, `nc` solo en el CIS) |
| `ccaa` | las 19 comunidades y ciudades autónomas con la etiqueta del CIS (p. ej. «Madrid (Comunidad de)») |
| `tamuni` | 7 tramos de tamaño de municipio del CIS (de «Menos o igual a 2.000 habitantes» a «Más de 1.000.000 habitantes») |
| `ingresos_hogar_tramo` | 6 tramos de ingresos netos mensuales del hogar del CIS (+ `N.S.`, `N.C.` solo en el CIS) |

Estudios del CIS con otra escala (p. ej. tamaño de municipio de 5 tramos o ingresos personales) dejan esa variable a
nulo: no se fuerza una equivalencia.

## Población (`etl-population`)

Ficheros: `synthetic_population.parquet`, `population_metadata.json`, `population_validation.json`.

- Mínimo obligatorio: `person_id`, `peso`, `sexo`, `edad`, `grupo_edad`, `nivel_estudios`, `version_poblacion`.
  Además se generan `ccaa`, `tamuni`, `ingresos_hogar_tramo`, `nacionalidad` (española/extranjera) y
  `situacion_laboral` (ocupado, parado, pensionista, estudiante, otra_inactividad).
- Método: microdatos del Censo 2021 (adultos) → IPF a las Cifras de Población (ECP) a 1 de enero de 2026
  (CCAA × sexo × edad, CCAA × nacionalidad, nacionalidad × sexo × edad) y a la EPA 2T26 (estudios por sexo × edad y por
  CCAA) → muestreo sistemático con probabilidad proporcional al peso (`--n`, `--seed`) → imputación de ingresos del hogar
  con la ECV 2025. El `peso` es uniforme y suma la población adulta objetivo (~41,4 M).
- Las referencias de la ECP posteriores al 30/06/2026 (p. ej. «1 de julio de 2026») se rechazan.
- `population_metadata.json`: `population_version`, `synthetic: false`, `coverage`, `variable_provenance`
  (`observed`, `derived` para `tamuni`, `imputed` para ingresos), `categories` reales, `sources`, `method` y
  `limitations`, y el informe del IPF. `synthetic` identifica solo fixtures de demostración.
- `population_validation.json`: error de los márgenes frente a los objetivos y distribuciones resultantes.
- Limitaciones: `situacion_laboral` es del Censo 2021 sin reponderar; los ingresos son imputados y sin N.S./N.C.; el
  tamaño de municipio de más de 10.000 habitantes se estima a partir de la muestra del Censo.

## CIS (`etl-cis`)

Ficheros: `cis_studies.parquet`, `cis_questions.parquet`, `cis_respondents.parquet`, `cis_responses.parquet` y
`etl_cis_report.json` (avisos por estudio).

- `cis_studies`: `estudio_id` (número CIS), `topic` (carpeta de `raw/`), `fecha_publicacion` (catálogo),
  `reservado_examen`, `fieldwork_start`/`fieldwork_end` (ficha técnica; si no existe o discrepa se usan las marcas
  `IA_E1_*` / `FECHAINI` de los microdatos, que prevalecen en caso de discrepancia, con aviso), `peso_source`, avisos.
  `fecha_publicacion` del catálogo **no** es la fecha de campo.
- `cis_questions`: `question_uid` (= `evidence_id`, `CIS<estudio>_<variable>`), `estudio_id`, `question_text` (etiqueta
  de la variable; los CIS no incluyen el enunciado completo), `options` (etiquetas de valor, NS/NC explícitos),
  `question_type` (`single`/`multiple`), `topic`, `fieldwork_*`, `reservado_examen`, `n_valid`, `variables`.
  Las baterías binarias (Sí/No, Menciona/No menciona) con ≥ 2 variables `X_n` se agrupan en una pregunta múltiple.
- `cis_respondents`: `estudio_id`, `respondent_uid` (= `REGISTRO`), `peso` (`PESO`), `peso_ccaa` (`PESOCCAA`),
  `sexo`, `edad`, `grupo_edad`, `nivel_estudios`, `ccaa`, `tamuni`, `ingresos_hogar_tramo`. Sin `PESO` se usa peso
  uniforme (avisado). `peso_ccaa` solo se usa si se pide explícitamente.
- `cis_responses` (formato largo): `estudio_id`, `respondent_uid`, `question_uid`, `answer` (etiqueta; N.P. =
  «no procede» se descarta), `selected` (solo preguntas múltiples; una fila por persona y opción, incluidos los ceros).
  Un grupo de filas por estudio, de modo que se puede filtrar por `estudio_id`/`question_uid` sin cargar el fichero.
- La clave `(estudio_id, respondent_uid)` evita mezclar entrevistas de estudios distintos.

## Backtesting (`make-backtest-cases`, `backtest`)

Los casos se generan solo de estudios `reservado_examen = si`. Cada caso lleva `evidence_id`, `question`, `options` (sin
NS/NC ni «(NO LEER)»), `topic`, `fieldwork_start/end`, `fieldwork_date` (= inicio, corte estricto), `observed_distribution`
(con `peso`, renormalizada sin NS/NC), `observed_subgroups` (sexo, edad, estudios con n ≥ 100) y `allow_previous_waves`.

```json
{
  "case_id": "CIS3559_P8",
  "evidence_id": "CIS3559_P8",
  "estudio_id": "3559",
  "question": "Valoración retrospectiva de la situación económica de España (6 meses)",
  "options": ["Mejor", "Igual", "Peor"],
  "topic": "consumo",
  "fieldwork_date": "2026-04-17",
  "observed_distribution": {"Mejor": 0.19, "Igual": 0.15, "Peor": 0.66},
  "allow_previous_waves": true
}
```

La evidencia recuperada debe haber terminado su trabajo de campo **antes** del `fieldwork_date`, nunca puede ser un
estudio reservado y el backtest aborta si detecta fuga de identificadores o temporal. Los resultados observados se leen
tras simular y no entran en el RAG ni en los prompts.

## Evidencia (`ingest-surveys`)

`ingest-surveys --path data/processed/cis_questions.parquet` omite por defecto los estudios reservados
(`--include-reserved` solo para pruebas). Los `.sav` no se ingieren directamente: primero `etl-cis`.
