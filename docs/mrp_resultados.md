# MRP para subgrupos: resultado de la evaluación prerregistrada (resumen)

Protocolo en `docs/prereg_mrp.md` (commit `cdd4ec1`). Código de la evaluación: commit `1ac2c8c`, ejecutado una sola vez (`outputs/mrp/DONE`). Antes de la ejecución real, el paso `run` (que no estaba en el commit del preregistro) se probó con 12 pares FUERA del conjunto (t entre 2024-04 y 2024-06; `--smoke`): sirvió para comprobar que el código funciona y no se cambió nada después.

# MRP frente al método actual de subgrupos (evaluación prerregistrada)

279 pares (279 en el recuento), 71 series, 7542 subgrupos (con ≥ 100 encuestados en t). Error en puntos porcentuales por subgrupo; IC 95 % remuestreando series (10000 réplicas, semilla 0).

| Sistema | MAE (pp) |
|---|---|
| Sin diferencias (nacional para todos) | 4.45 |
| Método actual | 3.22 |
| MRP | 2.64 |

**MRP − método actual: -0.58 pp, IC 95 % [-0.93, -0.32], Wilcoxon p = <0,001; MRP mejor en 4126 y peor en 3416 subgrupos.**

## Criterio (mejora ≥ 0.3 pp con el IC entero por debajo de 0): **CUMPLE** (mejora de +0.58 pp)

## Por dimensión
| Dimensión | Subgrupos | Sin diferencias | Actual | MRP | MRP − actual [IC 95 %] | Wilcoxon p |
|---|---|---|---|---|---|---|
| sexo | 558 | 2.20 | 1.38 | 1.00 | -0.39 [-0.63, -0.21] | <0,001 |
| grupo_edad | 1924 | 7.02 | 4.41 | 2.49 | -1.91 [-3.22, -0.85] | <0,001 |
| ccaa | 2845 | 3.34 | 3.03 | 3.10 | +0.07 [-0.03, +0.16] | <0,001 |
| nivel_estudios | 1286 | 3.84 | 2.58 | 2.30 | -0.28 [-0.49, -0.11] | <0,001 |
| ingresos_hogar_tramo | 929 | 4.72 | 3.36 | 3.01 | -0.35 [-0.55, -0.20] | <0,001 |

## Descriptivo
- Dimensiones omitidas por oleada t−1 (menos del 50 % de valores válidos), en número de pares: {'sexo': 0, 'grupo_edad': 0, 'ccaa': 0, 'nivel_estudios': 0, 'ingresos_hogar_tramo': 123}
- Valores de C elegidos por validación cruzada (número de pares): {'0.01': 35, '0.03': 49, '0.1': 89, '0.3': 47, '1.0': 21, '3.0': 13, '10.0': 25}
- Subgrupos saltados por no existir en la población o en el método: 0

## Cruces de dos variables (validación posterior)

Script `python -m mavericks.mrp_cross_eval` (no cambia la aplicación ni `mrp.py`). Tarea: con los microdatos de la oleada t−1 y el reparto nacional real de t, predecir el reparto de cada celda de dos variables de t y compararlo con lo medido (ponderado) en las celdas con ≥ 50 encuestados con respuesta sustantiva. Solo preguntas repetidas del CIS en oleadas no reservadas (trabajo de campo de t entre 2024-07-01 y 2026-03-31). Error = diferencia absoluta media entre opciones, en puntos porcentuales; «Nacional» es usar el reparto nacional de t para todas las celdas. IC 95 % remuestreando estudios (10000 réplicas, semilla 0).

**Total: 27889 celdas, 549 pares, 68 estudios, 139 series.** MRP 3.65 pp frente a 7.00 pp del reparto nacional (diferencia -3.35 pp, IC 95 % [-4.23, -2.52]).

### Por cruce
| Cruce | Celdas | Pares | Estudios | Nacional (pp) | MRP (pp) | MRP − nacional [IC 95 %] | MRP mejor / peor |
|---|---|---|---|---|---|---|---|
| grupo_edad×ccaa | 9977 | 549 | 68 | 6.89 | 4.31 | -2.58 [-3.36, -1.83] | 6580 / 3397 |
| nivel_estudios×grupo_edad | 10315 | 549 | 68 | 6.91 | 3.35 | -3.56 [-4.39, -2.75] | 7663 / 2652 |
| sexo×grupo_edad | 7597 | 549 | 68 | 7.26 | 3.19 | -4.07 [-5.12, -3.11] | 5579 / 2018 |

### Por tamaño de la celda en t
| n encuestados | Celdas | Pares | Estudios | Nacional (pp) | MRP (pp) | MRP − nacional [IC 95 %] | MRP mejor / peor |
|---|---|---|---|---|---|---|---|
| 50–99 | 12685 | 549 | 68 | 7.91 | 4.27 | -3.65 [-4.53, -2.82] | 8868 / 3817 |
| 100–199 | 8494 | 549 | 68 | 6.40 | 3.55 | -2.85 [-3.73, -2.01] | 5899 / 2595 |
| 200–∞ | 6710 | 525 | 67 | 6.03 | 2.61 | -3.42 [-4.29, -2.53] | 5055 / 1655 |

### Límites
- El objetivo nacional es el **real** de t (como en la evaluación prerregistrada de subgrupos); con el nacional que estima el modelo para una pregunta nueva, el error de las celdas será mayor.
- «Lo medido» es una muestra: el MAE incluye el ruido muestral de cada celda (más alto cuanto menor es n); no se ha descontado.
- El IC remuestrea estudios (cubre la dependencia entre preguntas y celdas de un mismo estudio), no la dependencia entre oleadas consecutivas de la misma serie.
- Para cruces no existe el «método anterior»: solo se compara con usar el reparto nacional para todas las celdas.
