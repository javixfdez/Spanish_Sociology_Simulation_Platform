# Modo ajustado: un LLM pequeño ajustado con encuestas reales

Resumen en una línea: **mejora con claridad en datos de desarrollo no vistos (dev_grande), pero en la evaluación final de los 88 casos oficiales no cumple el criterio
prerregistrado por muy poco (−1,28 pp, IC 95 % [−2,62; +0,01]). Por eso el modo rápido sigue siendo el modo por defecto y el modo ajustado es opcional.**

## Qué es

- **Modelo base:** `Qwen/Qwen3-4B-Instruct-2507` (revisión `cdbee75f17c01a7cc42f958dc650907174af0554`), cuantizado a 4 bits (nf4).
- **Adaptador:** QLoRA (LoRA r=8, alpha=32, dropout 0,05, en q, k, v, o, gate, up y down), entrenado con el método de Cao et al. (2025): el prompt presenta la pregunta con las
  opciones etiquetadas A, B, C… y la pérdida es la divergencia KL entre el reparto **real** y la distribución de probabilidad del **primer token** restringida a las letras válidas.
  En inferencia no se genera texto: se leen las probabilidades de las letras (≈0,3 s por pregunta en la RTX 4060).
- **Datos de entrenamiento:** solo repartos observados (ponderados, sin N.S./N.C.) del CIS (nacional y subgrupos con ≥100 entrevistados, como máximo 6 subgrupos por pregunta y dimensión),
  el Eurobarómetro y la ESS (filtrados: sin variables derivadas ni etiquetas de variable, en inglés, peso ×0,5; ESS11 excluida). Nada lo genera un LLM. Orden de opciones barajado al vuelo.
  Ningún estudio reservado (publicado desde 2026-04-01) entra en ningún conjunto (assert).
- **Dos modelos:**
  - **a1** (corte de trabajo de campo < 2025-07-01): 62.109 ejemplos disponibles, 53.029 de entrenamiento tras el presupuesto de tiempo (solo se recortan subgrupos), 1.767 de validación por estudios,
    3.315 pasos, bf16, 5,82 h en la RTX 4060. Pérdida de entrenamiento final 0,0159 y de validación 0,1506. Se evalúa en dev_grande.
  - **B** (corte < 2026-04-01): 94.949 ejemplos (92.657 de entrenamiento y 2.292 de validación), 5.792 pasos en Kaggle (2×T4, **fp16** porque la T4 no admite bf16), 6,76 h, sin submuestreo,
    4 pasos omitidos por desbordamiento de fp16 (todos al principio). Pérdida final de entrenamiento 0,0145 y de validación 0,1453. Es el que usa el modo ajustado y se evalúa en los 88 oficiales.
    Adaptador B: sha256 de `adapter_model.safetensors` = `f6a4dd9453f3233933b14a12af70d64ef0db197c37dedf24dbab0e832f339c6b`.
- **Modo ajustado:** si hay una pregunta casi idéntica en el histórico, se usa la estadística directa de siempre sin cambios. Si no, el reparto nacional y los de subgrupos salen del modelo
  (letras válidas, orden original, sin promediar permutaciones). Con más de 7 opciones (el modelo usa las letras A–G) usa el modo rápido y lo avisa.

## Resultados en dev_grande (a1; evaluación prerregistrada)

703 preguntas nuevas (sin pregunta parecida en el pasado) de 23 estudios, 2025-07 a 2026-03, solo reparto nacional. a1 se entrenó con trabajo de campo anterior a 2025-07-01. Error medio por opción en puntos
porcentuales; Δ frente al prior de producción (media 50/50 de KNN y MLP, el MLP entrenado por mes con su corte); IC 95 % remuestreando estudios enteros (10.000 réplicas, semilla 0).

| Sistema | MAE (pp) | Δ frente al prior [IC 95 %] | mejoran / empeoran | Wilcoxon p |
|---|---|---|---|---|
| Reparto uniforme | 24,22 | +5,60 [+3,85; +7,04] | 200 / 485 | <0,001 |
| KNN actual | 20,72 | +2,09 [+0,61; +3,24] | 288 / 375 | <0,001 |
| **Prior de producción** | **18,63** | — | — | — |
| Qwen base sin ajustar (misma lectura) | 34,19 | +15,56 [+13,76; +18,87] | 170 / 528 | <0,001 |
| **a1** | **12,44** | **−6,18 [−8,67; −2,37]** | 485 / 185 | <0,001 |
| Media 50/50 a1 + prior | 14,78 | −3,85 [−5,00; −2,07] | 501 / 157 | <0,001 |

- **Criterio** (mejora ≥ 0,5 pp con el IC entero por debajo de 0): a1 frente al prior **CUMPLE**; la media 50/50 a1 + prior frente al prior **CUMPLE**.
- **2 opciones (328 preguntas):** a1 −8,48 [−11,01; −1,17], media −4,96. **3 o más (375):** a1 −4,18 [−5,75; −2,39], media −2,88.
- **Comprobaciones de fuga** (la mejora supera los 4 pp, así que se buscó una fuga antes de informar): el entrenamiento de a1 llega a 2025-06-30 y dev_grande empieza el 2025-07-07; 0 identificadores y 0 estudios
  compartidos; 36 de las 703 preguntas tienen un texto idéntico a uno de entrenamiento (con otra escala, por lo que no cuentan como repetidas). **Sin esas 36 la mejora es de −5,93 pp [−8,56; −1,73]**, y sin ellas ni las
  variables de perfil sociodemográfico, −6,10. No se halló fuga. Las mayores ganancias vienen de ítems de sí/no o «Menciona / No menciona» en los que el prior se equivoca por 50–70 pp.
- **Subgrupos (descriptivo):** en 12.467 subgrupos del CIS de las mismas preguntas, el MAE de a1 es 12,47 pp frente a 18,26 del método actual de subgrupos (que apenas se separa del prior nacional sin diferencias, 18,32).
  La ganancia en subgrupos es sobre todo la del nivel nacional.
- **Validación interna** (1.767 ejemplos de 2 estudios no vistos por a1): KL 0,144 y MAE 10,6 pp frente a 3,02 y 29,5 del modelo base sin ajustar (el uniforme: 0,328 y 20,5).

## Resultados en los 88 casos oficiales (B; una sola ejecución)

Los 88 casos son de 11 estudios reservados (publicados desde 2026-04-01) que nunca se usaron para construir nada. Aquí la comparación principal es el modo ajustado frente al **modo rápido** (8,11 pp).
Criterio prerregistrado: mejora ≥ 0,5 pp con el IC 95 % por estudio entero por debajo de 0.

| Sistema | MAE (pp) | Δ frente al modo rápido [IC 95 %] | mejoran / empeoran | Wilcoxon p |
|---|---|---|---|---|
| Reparto uniforme | 17,13 | +9,02 [+6,72; +11,11] | 15 / 70 | <0,001 |
| Sistema anterior (directo / KNN) | 8,76 | +0,65 [−0,19; +1,56] | 16 / 20 | 0,180 |
| **Modo rápido** | **8,11** | — | — | — |
| Modo con LLM (referencia) | 7,91 | −0,20 [−0,54; +0,13] | 22 / 14 | 0,341 |
| **Modo ajustado (B)** | **6,84** | **−1,28 [−2,62; +0,01]** | 26 / 11 | 0,046 |

**Veredicto: NO CUMPLE.** La mejora media (1,28 pp) supera el umbral de 0,5 pp, pero el IC 95 % llega a +0,01 pp y no queda entero por debajo de 0. Es un resultado a las puertas del criterio: ni un fracaso claro ni una confirmación.

| Desglose | Casos | Modo rápido | Modo ajustado | Δ [IC 95 %] |
|---|---|---|---|---|
| Directos (estadística directa en ambos) | 47 | 2,49 | 2,49 | — |
| Sin ancla | 40 (9 estudios) | 14,57 | 12,05 | −2,52 [−4,33; +0,07] |
| Resto (prior) | 1 | 14,65 | 2,94 | un solo caso, no concluyente |
| No directos (sin ancla + resto) | 41 (9 estudios) | 14,57 | 11,82 | −2,74 [−4,63; −0,05] |
| 2 opciones (no directos) | 13 (6 estudios) | 17,41 | 13,91 | −3,50 [−13,26; +5,63] |
| 3 o más opciones (no directos) | 28 (9 estudios) | 13,25 | 10,85 | −2,39 [−3,89; −0,14] |

- **Cobertura de los rangos del modo ajustado** (por opción; calibración Mondrian de 2 y 3+ opciones con los residuos de a1 en dev_grande): 50 % → 46,5 % y 90 % → **86,5 %** en los no directos
  (todos los casos: 46,8 % y 87,3 %; los directos usan sus rangos actuales: 47,1 % y 88,0 %). Los rangos quedan algo cortos.
- **Regla de las 8 opciones** (fijada antes de ver resultados, commit `174cb82`): B responde a preguntas de 2 a 7 opciones; siete de los 88 casos tienen 8 (3 de ellos no directos) y en ellos el modo
  ajustado usa el modo rápido.
- **Secundarias (descriptivas, sin criterio):** B solo en los 81 casos con ≤ 7 opciones, incluso con ancla directa: 6,55 pp (Δ −1,65 [−2,99; −0,33] frente al rápido); media 50/50 B + prior en los 38 no directos
  con ≤ 7 opciones: 11,88 pp, frente a 11,52 del modo ajustado puro y 14,48 del rápido.

## Decisión

- **El modo rápido sigue por defecto**, según el criterio prerregistrado. El modo ajustado es **opcional** y es el de menor error medido en los 88 oficiales (6,84 pp frente a 8,11 del rápido y 7,91 del modo con LLM).
- La comparación «B también con ancla directa» (6,55 pp) es **secundaria** y no se adopta: queda como trabajo futuro (habría que prerregistrarla y confirmarla con datos nuevos).
- No se repite ninguna evaluación y no se ajusta nada tras ver los resultados.

## Limitaciones

- **Potencia baja en los 88:** solo 41 casos informativos (los no directos) de 9 estudios; los IC son anchos y el resultado principal roza el umbral.
- **Solo preguntas de 2 a 7 opciones** (letras A–G); con más, el modo ajustado usa el modo rápido.
- **a1 y B no son el mismo modelo:** los checkpoints intermedios de a1 no se conservaron (solo `adapter_final`, paso 3.315); a1 se entrenó en bf16 y B en fp16 (T4). La evidencia de dev_grande es de a1 y la de los 88, de B.
- **B vio dev_grande en el entrenamiento** (corte 2026-04-01), por lo que su error allí no es evaluable; sus rangos se calibraron con los residuos de a1.
- **VRAM justa:** con el modelo ajustado y qwen3:8b de Ollama cargados a la vez, el pico fue de 7.944 MiB de 8.188 (unos 245 MiB de margen). Si Ollama no tiene memoria para redactar la explicación, se usa la plantilla;
  el servidor libera la caché de CUDA tras cada lote.
- La mejora se concentra en ítems binarios o de tipo «Menciona / No menciona», donde el prior estadístico falla mucho; no es una prueba de que el modelo «entienda» la opinión pública.
- Qwen3-4B-Instruct-2507 se publicó en julio de 2025; dev_grande (desde 2025-07) es con alta probabilidad posterior a su preentrenamiento, pero la fecha de corte exacta no está confirmada (ver `docs/llm_fechas.md`).

## Cómo usarlo

0. Prepara el entorno del modelo (una vez). Es un entorno aparte del de `uv sync`; necesita una GPU NVIDIA con ~3 GB libres, probado en una RTX 4060 de 8 GB.

```sh
uv venv .venv_ft --python 3.13
uv pip install --python .venv_ft\Scripts\python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv_ft\Scripts\python.exe -r src/mavericks/ft/requirements.txt
# una vez, con internet: base fijada por revisión (el servidor trabaja sin conexión)
.venv_ft\Scripts\python.exe -c "from huggingface_hub import snapshot_download as d; d('Qwen/Qwen3-4B-Instruct-2507', revision='cdbee75f17c01a7cc42f958dc650907174af0554')"
```

1. Coloca el adaptador en `outputs/ft/runs/b1/adapter_final/` (con `adapter_config.json` y `adapter_model.safetensors`; sha256 `f6a4dd94…`). No está en git (los pesos nunca se suben).
2. Arranca el servidor (GPU, entorno `.venv_ft`; no para Ollama):
   ```
   powershell -NoProfile -ExecutionPolicy Bypass -File outputs\ft\run_server.ps1
   ```
   Escucha en `127.0.0.1:8765` (`-Port` lo cambia; el cliente usa `MAVERICKS_FT_URL`). `GET /health` devuelve `{"ok": true, ...}`; `POST /predict` y `POST /predict_batch` devuelven las probabilidades.
3. En la página **Simular** aparece el modo **Ajustado** solo si el servidor responde. Desde código: `predict(..., preset="ajustado")`.
4. **Fallback:** si el servidor no responde en 30 s, da error o la pregunta tiene más de 7 opciones, el servicio muestra la estimación del modo rápido, con aviso y `fallback=true` en la respuesta.
5. Los rangos del modo ajustado salen de `outputs/models/ft_intervals.json` (se genera con `python -m mavericks.ft.artifacts`); sin ese archivo no se muestran rangos.

## Cómo se hizo (reproducibilidad)

Todo el proceso está prerregistrado en [`docs/prereg_finetuning.md`](prereg_finetuning.md) (cada anexo se fijó en git antes de ejecutar lo que describe). Código en `src/mavericks/ft/`:
`data.py` (ejemplos), `core.py` (prompt y pérdida), `train.py` (a1, local), `train_kaggle.py` (B), `internal_val.py`, `dev_infer.py`, `dev_stats.py`, `dev_leak.py` (evaluación en dev_grande),
`server.py`, `client.py`, `artifacts.py` (modo ajustado) y `final_eval_ft.py` (evaluación en los 88). Resúmenes sin datos crudos: `docs/finetuning_fase3_resultados.md` y `docs/finetuning_fase4_resultados.md`.
