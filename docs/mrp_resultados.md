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