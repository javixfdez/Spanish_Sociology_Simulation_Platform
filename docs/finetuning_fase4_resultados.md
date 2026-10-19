# Fine-tuning, fase 4: evaluación única del modo ajustado (modelo B) en los 88 oficiales

Resumen sin datos crudos; generado por `mavericks.final_eval_ft` (marcador `outputs/final_eval_ft/DONE`). Por defecto sigue el modo rápido.

# Evaluación única del modo ajustado (modelo B) — 88 casos oficiales (estudios reservados)

Adaptador B sha256 `f6a4dd9453f3233933b14a12af70d64ef0db197c37dedf24dbab0e832f339c6b`. Bootstrap por estudios (11 estudios; 10000 réplicas, semilla 0). Criterio: mejora ≥ 0.5 pp frente al modo rápido con el IC 95 % entero por debajo de 0.

### Total (comparación principal: modo ajustado frente a modo rápido) (88 casos, 11 estudios)
| Sistema | MAE (pp) | Δ frente al modo rápido [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |
|---|---|---|---|---|
| Reparto uniforme | 17.13 | +9.02 [+6.72, +11.11] | 15 / 70 / 3 | <0,001 |
| Sistema anterior (directo / KNN actual) | 8.76 | +0.65 [-0.19, +1.56] | 16 / 20 / 52 | 0.180 |
| Modo rápido (directo / KNN + MLP) | 8.11 | — | — | — |
| Modo con LLM (directo / 0,3·LLM + 0,7·prior) — referencia | 7.91 | -0.20 [-0.54, +0.13] | 22 / 14 / 52 | 0.341 |
| Modo AJUSTADO (directo / modelo B) | 6.84 | -1.28 [-2.62, +0.01] | 26 / 11 / 51 | 0.046 |

### Directos (estadística directa en ambos modos) (47 casos)
MAE de la estadística directa 2.49 pp (uniforme 14.80). Secundaria: B en los 43 con ≤ 7 opciones 2.15 pp.

### Sin ancla (pregunta nueva) (40 casos, 9 estudios)
| Sistema | MAE (pp) | Δ frente al modo rápido [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |
|---|---|---|---|---|
| Reparto uniforme | 19.89 | +5.33 [+2.01, +8.46] | 13 / 24 / 3 | 0.005 |
| Sistema anterior (directo / KNN actual) | 16.01 | +1.45 [-0.41, +3.67] | 15 / 20 / 5 | 0.166 |
| Modo rápido (directo / KNN + MLP) | 14.57 | — | — | — |
| Modo con LLM (directo / 0,3·LLM + 0,7·prior) — referencia | 14.09 | -0.47 [-1.38, +0.23] | 22 / 13 / 5 | 0.307 |
| Modo AJUSTADO (directo / modelo B) | 12.05 | -2.52 [-4.33, +0.07] | 25 / 11 / 4 | 0.067 |

### Resto (prior: pregunta parecida) (1 casos, 1 estudios)
| Sistema | MAE (pp) | Δ frente al modo rápido [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |
|---|---|---|---|---|
| Reparto uniforme | 16.26 | +1.61 [+1.61, +1.61] | 0 / 1 / 0 | — |
| Sistema anterior (directo / KNN actual) | 13.83 | -0.81 [-0.81, -0.81] | 1 / 0 / 0 | — |
| Modo rápido (directo / KNN + MLP) | 14.65 | — | — | — |
| Modo con LLM (directo / 0,3·LLM + 0,7·prior) — referencia | 15.87 | +1.23 [+1.23, +1.23] | 0 / 1 / 0 | — |
| Modo AJUSTADO (directo / modelo B) | 2.94 | -11.71 [-11.71, -11.71] | 1 / 0 / 0 | — |

### No directos (sin ancla + resto) (41 casos, 9 estudios)
| Sistema | MAE (pp) | Δ frente al modo rápido [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |
|---|---|---|---|---|
| Reparto uniforme | 19.80 | +5.23 [+2.00, +8.20] | 13 / 25 / 3 | 0.004 |
| Sistema anterior (directo / KNN actual) | 15.96 | +1.39 [-0.42, +3.65] | 16 / 20 / 5 | 0.184 |
| Modo rápido (directo / KNN + MLP) | 14.57 | — | — | — |
| Modo con LLM (directo / 0,3·LLM + 0,7·prior) — referencia | 14.14 | -0.43 [-1.37, +0.27] | 22 / 14 / 5 | 0.348 |
| Modo AJUSTADO (directo / modelo B) | 11.82 | -2.74 [-4.63, -0.05] | 26 / 11 / 4 | 0.046 |

Casos con más de 7 opciones (B no puede responder; el modo ajustado usa el modo rápido): 7 (3 no directos).

### Secundaria: B en todos los casos con ≤ 7 opciones, incluso con ancla directa (81 casos, 11 estudios)
| Sistema | MAE (pp) | Δ frente al modo rápido [IC 95 %] | mejoran / empeoran / empatan |
|---|---|---|---|
| Modo rápido (directo / KNN + MLP) | 8.19 | — | — |
| Modo AJUSTADO (directo / modelo B) | 6.80 | -1.39 [-2.85, +0.01] | 26 / 11 / 44 |
| B solo (incluso con ancla directa) | 6.55 | -1.65 [-2.99, -0.33] | 42 / 23 / 16 |

### Secundaria: media 50/50 B + prior rápido en los no directos con ≤ 7 opciones (38 casos, 9 estudios)
| Sistema | MAE (pp) | Δ frente al modo rápido [IC 95 %] | mejoran / empeoran / empatan |
|---|---|---|---|
| Modo rápido (directo / KNN + MLP) | 14.48 | — | — |
| Modo AJUSTADO (directo / modelo B) | 11.52 | -2.96 [-4.86, -0.06] | 26 / 11 / 1 |
| Media 50/50 B + prior rápido | 11.88 | -2.60 [-3.64, -0.94] | 25 / 9 / 4 |

### Por número de opciones (no directos)

### 2 opciones (13 casos, 6 estudios)
| Sistema | MAE (pp) | Δ frente al modo rápido [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |
|---|---|---|---|---|
| Reparto uniforme | 22.46 | +5.05 [-1.93, +12.47] | 4 / 9 / 0 | 0.376 |
| Sistema anterior (directo / KNN actual) | 19.01 | +1.60 [-6.18, +8.42] | 5 / 7 / 1 | 0.497 |
| Modo rápido (directo / KNN + MLP) | 17.41 | — | — | — |
| Modo con LLM (directo / 0,3·LLM + 0,7·prior) — referencia | 17.66 | +0.25 [-2.03, +2.47] | 6 / 6 / 1 | 0.839 |
| Modo AJUSTADO (directo / modelo B) | 13.91 | -3.50 [-13.26, +5.63] | 8 / 5 / 0 | 0.542 |

### 3 o más (28 casos, 9 estudios)
| Sistema | MAE (pp) | Δ frente al modo rápido [IC 95 % por estudio] | mejoran / empeoran / empatan | Wilcoxon p |
|---|---|---|---|---|
| Reparto uniforme | 18.57 | +5.32 [+2.09, +10.07] | 9 / 16 / 3 | 0.010 |
| Sistema anterior (directo / KNN actual) | 14.54 | +1.30 [+0.09, +2.66] | 11 / 13 / 4 | 0.274 |
| Modo rápido (directo / KNN + MLP) | 13.25 | — | — | — |
| Modo con LLM (directo / 0,3·LLM + 0,7·prior) — referencia | 12.50 | -0.75 [-1.32, -0.40] | 16 / 8 / 4 | 0.115 |
| Modo AJUSTADO (directo / modelo B) | 10.85 | -2.39 [-3.89, -0.14] | 18 / 6 / 4 | 0.025 |

## Criterio prerregistrado
- **Modo ajustado frente a modo rápido en los 88: NO CUMPLE** (mejora +1.28 pp, IC 95 % [-2.62, +0.01]; estudios remuestreados)

## Cobertura de los rangos (descriptiva; por opción)
| Casos | 50 % | 90 % |
|---|---|---|
| direct | 47.1 % | 88.0 % |
| no_direct | 46.5 % | 86.5 % |
| todos | 46.8 % | 87.3 % |

## Coste
- segundos por pregunta de B: servidor 0.312 s, ida y vuelta 0.321 s (88 consultas una a una)
- VRAM: pico total de la GPU durante la evaluación 7838 MiB de 8188 (con el servidor de B y el modelo de Ollama cargados); máximo asignado por el servidor 2.81 GB