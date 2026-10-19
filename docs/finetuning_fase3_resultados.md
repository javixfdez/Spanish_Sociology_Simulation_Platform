# Fine-tuning, fase 3: resultados de a1 (resumen sin datos crudos)

Generado por `mavericks.ft.dev_stats`, `dev_leak` y `internal_val`. Evaluación única en dev_grande (marcador `outputs/ft/phase3/dev_grande_DONE`). Nada integrado en producción.

# Validación interna de a1 (estudios no vistos por a1; orden de opciones original, semilla 0)

1767 ejemplos de 2 estudios, 89 nacionales.

| sistema | ámbito | opciones | ejemplos | KL sin ponderar | KL ponderada | MAE (pp) | top-1 |
|---|---|---|---|---|---|---|---|
| a1 | global | todas | 1767 | 0.144 | 0.149 | 10.583 | 0.722 |
| a1 | global | 2 opciones | 620 | 0.169 | 0.191 | 18.622 | 0.642 |
| a1 | global | 3+ opciones | 1147 | 0.130 | 0.129 | 6.238 | 0.765 |
| a1 | nacional | todas | 89 | 0.133 | 0.134 | 9.610 | 0.742 |
| a1 | nacional | 2 opciones | 31 | 0.156 | 0.156 | 17.601 | 0.645 |
| a1 | nacional | 3+ opciones | 58 | 0.121 | 0.122 | 5.339 | 0.793 |
| Qwen base sin ajustar | global | todas | 1767 | 3.021 | 3.103 | 29.539 | 0.513 |
| Qwen base sin ajustar | global | 2 opciones | 620 | 2.018 | 1.965 | 38.229 | 0.624 |
| Qwen base sin ajustar | global | 3+ opciones | 1147 | 3.564 | 3.676 | 24.842 | 0.452 |
| Qwen base sin ajustar | nacional | todas | 89 | 2.973 | 2.989 | 29.775 | 0.483 |
| Qwen base sin ajustar | nacional | 2 opciones | 31 | 1.948 | 1.948 | 39.356 | 0.516 |
| Qwen base sin ajustar | nacional | 3+ opciones | 58 | 3.521 | 3.552 | 24.653 | 0.466 |
| uniforme | global | todas | 1767 | 0.328 | 0.320 | 20.510 | 0.301 |
| uniforme | global | 2 opciones | 620 | 0.223 | 0.213 | 27.096 | 0.160 |
| uniforme | global | 3+ opciones | 1147 | 0.384 | 0.373 | 16.950 | 0.377 |
| uniforme | nacional | todas | 89 | 0.300 | 0.299 | 19.662 | 0.303 |
| uniforme | nacional | 2 opciones | 31 | 0.207 | 0.207 | 26.186 | 0.129 |
| uniforme | nacional | 3+ opciones | 58 | 0.350 | 0.349 | 16.175 | 0.397 |
# Fase 3 — evaluación prerregistrada de a1 en dev_grande (preguntas nuevas, reparto nacional)

703 preguntas, 23 estudios. Prior de producción = media 50/50 de KNN actual y MLP por mes con su corte. IC por bootstrap de estudios enteros (10000 réplicas, semilla 0). Criterio: mejora ≥ 0.5 pp con el IC entero por debajo de 0.

### Todas las preguntas nuevas (703 preguntas, 23 estudios)
| Sistema | MAE (pp) | Δ frente al prior [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |
|---|---|---|---|---|
| Reparto uniforme | 24.22 | +5.60 [+3.85, +7.04] | 200 / 485 / 18 | <0,001 |
| KNN actual | 20.72 | +2.09 [+0.61, +3.24] | 288 / 375 / 40 | <0,001 |
| Prior de producción (50/50 KNN+MLP) | 18.63 | — | — | — |
| Qwen base sin ajustar | 34.19 | +15.56 [+13.76, +18.87] | 170 / 528 / 5 | <0,001 |
| a1 | 12.44 | -6.18 [-8.67, -2.37] | 485 / 185 / 33 | <0,001 |
| Media 50/50 a1 + prior | 14.78 | -3.85 [-5.00, -2.07] | 501 / 157 / 45 | <0,001 |

### 2 opciones (328 preguntas, 20 estudios)
| Sistema | MAE (pp) | Δ frente al prior [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |
|---|---|---|---|---|
| Reparto uniforme | 32.12 | +7.08 [+2.77, +9.66] | 98 / 225 / 5 | <0,001 |
| KNN actual | 27.80 | +2.76 [+0.73, +6.22] | 127 / 191 / 10 | <0,001 |
| Prior de producción (50/50 KNN+MLP) | 25.04 | — | — | — |
| Qwen base sin ajustar | 42.91 | +17.87 [+13.07, +25.38] | 111 / 216 / 1 | <0,001 |
| a1 | 16.56 | -8.48 [-11.01, -1.17] | 222 / 100 / 6 | <0,001 |
| Media 50/50 a1 + prior | 20.08 | -4.96 [-6.16, -1.49] | 230 / 89 / 9 | <0,001 |

### 3 o más opciones (375 preguntas, 23 estudios)
| Sistema | MAE (pp) | Δ frente al prior [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |
|---|---|---|---|---|
| Reparto uniforme | 17.32 | +4.30 [+2.74, +6.09] | 102 / 260 / 13 | <0,001 |
| KNN actual | 14.52 | +1.50 [+0.19, +2.90] | 161 / 184 / 30 | 0.002 |
| Prior de producción (50/50 KNN+MLP) | 13.02 | — | — | — |
| Qwen base sin ajustar | 26.56 | +13.55 [+10.17, +16.78] | 59 / 312 / 4 | <0,001 |
| a1 | 8.84 | -4.18 [-5.75, -2.39] | 263 / 85 / 27 | <0,001 |
| Media 50/50 a1 + prior | 10.14 | -2.88 [-3.59, -2.13] | 271 / 68 / 36 | <0,001 |

## Criterio prerregistrado

- **a1 frente al prior de producción: CUMPLE** (mejora +6.18 pp, IC 95 % por estudio [-8.67, -2.37])
- **Media 50/50 a1 + prior frente al prior de producción: CUMPLE** (mejora +3.85 pp, IC 95 % por estudio [-5.00, -2.07])

Segundos por pregunta (una a una, RTX 4060 portátil): a1 0.323; Qwen base 0.254

## Comprobaciones de fuga
- fecha máxima de entrenamiento de a1: 2025-06-30 (< 2025-07-01); fecha mínima de dev_grande: 2025-07-07
- ids de dev_grande dentro del entrenamiento de a1: 0; estudios compartidos: 0
- preguntas de dev_grande con texto idéntico (normalizado) a una de entrenamiento (CIS, Eurobarómetro o ESS): 36 de 703
- mejora > 4 pp sobre el prior: a1 (6.18 pp)
# Búsqueda de fuga (mejora de a1 > 4 pp)

| Subconjunto | preguntas | estudios | MAE prior | MAE a1 | Δ a1 − prior [IC 95 % por estudio] | MAE 50/50 | Δ 50/50 − prior [IC] |
|---|---|---|---|---|---|---|---|
| todas | 703 | 23 | 18.63 | 12.44 | -6.18 [-8.67, -2.37] | 14.78 | -3.85 [-5.00, -2.07] |
| sin texto idéntico a uno de entrenamiento | 667 | 23 | 18.78 | 12.86 | -5.93 [-8.56, -1.73] | 15.04 | -3.74 [-4.95, -1.79] |
| solo texto idéntico | 36 | 12 | 15.74 | 4.79 | -10.96 [-16.26, -6.63] | 9.80 | -5.94 [-8.45, -3.84] |
| sin variables de perfil | 680 | 23 | 18.89 | 12.72 | -6.17 [-8.62, -2.32] | 15.05 | -3.84 [-4.99, -2.04] |
| solo variables de perfil | 23 | 13 | 10.78 | 4.14 | -6.64 [-15.62, -0.25] | 6.77 | -4.01 [-8.12, -1.05] |
| sin ambos | 656 | 23 | 19.04 | 12.94 | -6.10 [-8.62, -1.99] | 15.23 | -3.82 [-4.99, -1.89] |

Diez preguntas con mayor mejora de a1 (id, MAE prior, MAE a1):
- CIS3554_P17: 71.4 → 7.1
- CIS3515_F2_5: 56.6 → 4.9
- CIS3515_F2_3: 53.8 → 6.3
- CIS3515_D5_99: 48.8 → 2.4
- CIS3515_B3_6: 54.6 → 8.5
- CIS3515_S1: 50.4 → 5.9
- CIS3515_F2_4: 52.9 → 8.8
- CIS3515_H16: 52.0 → 8.7
- CIS3508_P10: 64.0 → 21.2
- CIS3540_P2_96R: 45.3 → 2.6
# Descriptivo de subgrupos — a1 frente al método actual (preguntas nuevas de dev_grande)

12467 subgrupos del CIS (de 13481 muestreados; el resto sin análogas con microdatos o con categoría ausente de la población sintética). Método actual = diferencias empíricas de las análogas (KNN actual) contraídas a la mitad con raking sobre el prior de producción. Sin criterio de adopción.

| Ámbito | subgrupos | MAE a1 (pp) | MAE método actual (pp) | MAE «prior nacional sin diferencias» (pp) | Δ a1 − actual [IC 95 % por estudio] | a1 mejor / peor |
|---|---|---|---|---|---|---|
| todos | 12467 | 12.47 | 18.26 | 18.32 | -5.78 [-8.42, -2.43] | 8454 / 3396 |
| dim ccaa | 3233 | 12.80 | 18.46 | 18.48 | -5.66 [-8.43, -1.76] | 2174 / 884 |
| dim grupo_edad | 3390 | 12.74 | 18.92 | 19.04 | -6.18 [-8.94, -2.48] | 2298 / 927 |
| dim ingresos_hogar_tramo | 2033 | 10.89 | 15.38 | 15.43 | -4.49 [-6.39, -2.46] | 1358 / 567 |
| dim nivel_estudios | 2586 | 12.78 | 19.06 | 19.12 | -6.28 [-8.92, -2.34] | 1781 / 687 |
| dim sexo | 1225 | 12.85 | 18.94 | 18.99 | -6.10 [-8.76, -2.22] | 843 / 331 |
| 2 opciones | 5207 | 16.93 | 24.47 | 24.53 | -7.54 [-11.12, -0.36] | 3328 / 1742 |
| 3+ opciones | 7260 | 9.28 | 13.80 | 13.86 | -4.52 [-5.89, -2.89] | 5126 / 1654 |

Media 50/50 a1 + método actual: 14.51 pp.