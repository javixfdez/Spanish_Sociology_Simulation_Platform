# Preregistro — MRP para repartos por subgrupo (fijado en git ANTES de evaluar)

**Pregunta.** ¿Un MRP rápido (sin MCMC) predice mejor los repartos por subgrupo que el método actual (desviaciones empíricas de la oleada anterior contraídas al 50 % + raking)?

## Datos (no se ha mirado ningún error)

- **Pares (t−1, t):** preguntas repetidas del CIS. Misma serie (`anchor.series_key`, nombre de variable estable; se excluyen P1, P2… sin nombre), similitud de redacción ≥ 0,85 (`anchor.question_similarity`),
  mismo ítem de batería (`anchor.same_item`), opciones equivalentes (`anchor.option_mapping`) y 2 a 7 opciones sustantivas (sin N.S./N.C.), pregunta de respuesta única.
  Oleada **t** con trabajo de campo entre **2024-07-01 y 2025-06-30**; oleada **t−1** = la anterior más reciente de la misma serie, con trabajo de campo desde 2024-01-01 y de otro estudio.
- **Se excluyen** las variables de perfil sociodemográfico (`prior_v2.bigdev_eval.PROFILE`: sexo, edad, estudios, CCAA…), porque predecir un subgrupo de «sexo» con la pregunta «sexo» no mide nada.
  Nada reservado (publicado desde 2026-04-01), nada de dev_grande (desde 2025-07-01) ni los 88 oficiales: lo comprueba `assert_no_reserved` y las fechas.
- **Recuento fijado antes de calcular errores** (`python -m mavericks.mrp_eval counts`): **279 pares**, **71 series**, 40 estudios en t y 43 en t−1 (t de 2024-07-04 a 2025-06-30; t−1 de 2024-04-04 a 2025-06-07).
  **7.542 subgrupos** con ≥ 100 encuestados con respuesta sustantiva en t: sexo 558, edad 1.924, CCAA 2.845, estudios 1.286, ingresos 929. Se usan todos (sin muestreo).
- Categorías no válidas (N.S., N.C., «nc», «otros») no son subgrupos. La población sintética (50.000 adultos, calibrada con el INE) es el marco de postestratificación y de pesos del raking de los dos métodos.

## Tarea

Con los **microdatos de t−1** y el **reparto nacional REAL de t** (el que daría el modo directo, el prior o el modelo ajustado si acertaran), predecir el reparto de cada subgrupo de t. La verdad es el reparto ponderado (pesos del CIS)
de las respuestas sustantivas del subgrupo en t.

## Sistemas (cada uno recibe exactamente las mismas entradas)

1. **Método actual:** `analog_segments.group_estimates` con la oleada t−1 como única análoga (peso 1): desviación del subgrupo respecto al nacional de t−1 (con la reducción jerárquica del `AnchorStore`), aplicada al nacional de t
   con contracción 0,5 y raking dentro de cada dimensión con las cuotas de la población.
2. **MRP** (`mrp.py`; **una única variante**, definida aquí y no se ajusta tras ver resultados):
   - Regresión logística multinomial con penalización L2, variables indicadoras de sexo, tramo de edad (CIS), CCAA, estudios e ingresos más cuatro variables por CCAA estandarizadas derivadas de la población sintética calibrada con el INE
     (% con estudios superiores, % en municipios de más de 100.000 habitantes, renta del hogar por punto medio de tramo y edad media). Pesos del CIS normalizados a media 1 (sin recorte).
   - `C` ∈ {0,01; 0,03; 0,1; 0,3; 1; 3; 10} elegido por validación cruzada de 5 pliegues (semilla 0, pérdida logarítmica ponderada) **dentro de los microdatos de t−1** de cada pregunta.
   - Una dimensión con menos del 50 % de valores válidos en t−1 se omite para esa oleada (se registra); los valores no válidos de una dimensión usada quedan con todas sus indicadoras a cero (sin imputar).
   - Postestratificación sobre la población sintética y recalibración por **inclinación exponencial** de los logits para que el nacional coincida con el nacional real de t.
3. **Sin diferencias:** el nacional de t para todos (referencia).

## Métrica y análisis

- **MAE por subgrupo** en puntos porcentuales (media de |predicho − observado| sobre las opciones).
- IC 95 % por **bootstrap remuestreando series de preguntas** (las 71; 10.000 réplicas, semilla 0) de la diferencia media MRP − método actual; Wilcoxon (pareado, por subgrupo); desglose por dimensión (edad, sexo, CCAA, estudios, ingresos) con el mismo bootstrap.
- Descriptivo (sin criterio): cuántas dimensiones se omitieron, valores de `C` elegidos y subgrupos en los que gana cada método.

## Criterio

**Se adopta MRP** si mejora al método actual en **≥ 0,3 pp** (MAE medio de todos los subgrupos) con el IC 95 % **entero por debajo de 0**. Si no, el método actual se mantiene. No se prueban variantes del MRP ni se elige la mejor;
si se quisiera otra variante, requeriría un nuevo preregistro y datos no usados aquí.

## Marcador

`outputs/mrp/DONE`: tras ejecutar `python -m mavericks.mrp_eval run` una sola vez no se repite ni se cambia nada.
