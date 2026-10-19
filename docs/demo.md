# Guion y respaldo de la demo

Prueba hecha el 2026-10-19 con el código de la rama `javier-picazo` (RTX 4060 Laptop 8 GB, Ollama 0.40.6, KV q8_0, atención flash, paralelo 2, `qwen3:8b`; servidor del modelo ajustado B en el puerto 8765).
Las llamadas son las mismas que hace la interfaz (`service.predict` y la explicación con comprobación de cifras); se repiten con `python -m mavericks.demo_run rapido|llm|ajustado` y `python -m mavericks.demo_run report`.
Condiciones: **rápido** sin Ollama ni servidor (Ollama parado, 0 MiB de VRAM); **con LLM** con Ollama; **ajustado** con `scripts\run_server.ps1` y Ollama a la vez.

## Resumen de tiempos

| Modo | Segundos por pregunta (cifras) | Explicación | Notas |
|---|---|---|---|
| Rápido | 3.1 de media (la primera, 5.0) | plantilla, instantánea | carga de e5 al arrancar: 18 s (una vez) |
| Con LLM | 108 de media (94–116) | 2,5–3 s con qwen3:8b | 10 segmentos de población × modelo de razonamiento; no sirve para una demo en directo si no está precalculado |
| Ajustado | 5.6 de media (la primera, 10.1) | 3 s con Ollama (0 s si está en caché) | lectura del modelo ajustado ≈ 0,3 s por petición; el resto es el prior rápido que se calcula igualmente para decidir el modo |

La pregunta fuera de alcance se rechaza en milisegundos en los tres modos con la tarjeta «Fuera de alcance» (no se simula nada).

## Las preguntas, en los tres modos

### 1. Vivienda: «¿Cree que el precio del alquiler debería limitarse por ley?»
Opciones: Muy de acuerdo / Bastante de acuerdo / Regular / Poco de acuerdo / Nada de acuerdo

| Modo | Cómo se calculó | Primera opción | Confianza | Segundos (cifras + explicación) | Explicación |
|---|---|---|---|---|---|
| Rápido | Rápido · none | Regular (39 %) | baja | 5.0 + 0.0 | plantilla |
| Con LLM | Con LLM · none | Regular (35 %) | baja | 116.4 + 2.8 | llm |
| Ajustado | Ajustado (experimental) · none | Muy de acuerdo (42 %) | baja | 10.1 + 0.0 | llm |

### 2. Soledad: «¿Con qué frecuencia se siente solo o sola?»
Opciones: Nunca / Casi nunca / A veces / A menudo / Siempre

| Modo | Cómo se calculó | Primera opción | Confianza | Segundos (cifras + explicación) | Explicación |
|---|---|---|---|---|---|
| Rápido | Rápido · none | Casi nunca (33 %) | baja | 2.9 + 0.0 | plantilla |
| Con LLM | Con LLM · none | Casi nunca (30 %) | baja | 111.7 + 3.0 | llm |
| Ajustado | Ajustado (experimental) · none | Nunca (28 %) | baja | 4.0 + 0.0 | llm |

### 3. Ia: «¿Le preocupa que la inteligencia artificial le quite su empleo?»
Opciones: Mucho / Bastante / Poco / Nada

| Modo | Cómo se calculó | Primera opción | Confianza | Segundos (cifras + explicación) | Explicación |
|---|---|---|---|---|---|
| Rápido | Rápido · none | Nada (42 %) | baja | 1.8 + 0.0 | plantilla |
| Con LLM | Con LLM · none | Nada (34 %) | baja | 110.4 + 2.7 | llm |
| Ajustado | Ajustado (experimental) · none | Mucho (39 %) | baja | 4.0 + 0.0 | llm |

### 4. Medio ambiente: «¿Está a favor de prohibir la venta de coches de combustión a partir de 2035?»
Opciones: Muy de acuerdo / Bastante de acuerdo / Regular / Poco de acuerdo / Nada de acuerdo

| Modo | Cómo se calculó | Primera opción | Confianza | Segundos (cifras + explicación) | Explicación |
|---|---|---|---|---|---|
| Rápido | Rápido · none | Muy de acuerdo (33 %) | baja | 3.1 + 0.0 | plantilla |
| Con LLM | Con LLM · none | Muy de acuerdo (27 %) | baja | 106.1 + 2.6 | llm |
| Ajustado | Ajustado (experimental) · none | Muy de acuerdo (39 %) | baja | 5.1 + 0.0 | llm |

### 5. Salud: «¿Cómo valora el funcionamiento de la sanidad pública?»
Opciones: Muy bueno / Bueno / Regular / Malo / Muy malo

| Modo | Cómo se calculó | Primera opción | Confianza | Segundos (cifras + explicación) | Explicación |
|---|---|---|---|---|---|
| Rápido | Rápido · none | Bueno (41 %) | baja | 3.4 + 0.0 | plantilla |
| Con LLM | Con LLM · none | Bueno (43 %) | baja | 112.0 + 2.7 | llm |
| Ajustado | Ajustado (experimental) · none | Bueno (45 %) | baja | 4.3 + 8.8 | llm |

### 6. Consumo: «¿Piensa comprar menos productos en los próximos meses por la subida de precios?»
Opciones: Sí / No

| Modo | Cómo se calculó | Primera opción | Confianza | Segundos (cifras + explicación) | Explicación |
|---|---|---|---|---|---|
| Rápido | Rápido · none | No (79 %) | baja | 2.7 + 0.0 | plantilla |
| Con LLM | Con LLM · none | No (72 %) | baja | 93.8 + 2.5 | llm |
| Ajustado | Ajustado (experimental) · none | Sí (65 %) | baja | 6.1 + 0.0 | llm |

### 7. Fuera de alcance: «¿Cuál es la capital de Francia?»
Opciones: París / Londres

| Modo | Cómo se calculó | Primera opción | Confianza | Segundos (cifras + explicación) | Explicación |
|---|---|---|---|---|---|
| Rápido | fuera de alcance (Fuera de alcance) | — | — | 0.0 | — |
| Con LLM | fuera de alcance (Fuera de alcance) | — | — | 0.0 | — |
| Ajustado | fuera de alcance (Fuera de alcance) | — | — | 0.0 | — |

## Prueba de caída del servidor del modelo ajustado

- «¿Cree que el precio del alquiler debería limitarse por ley?» con el servidor apagado: modo que produjo el resultado **Rápido**, fallback **True**, motivo: el servidor del modelo ajustado no respondió; avisos: ['Las preguntas más parecidas del CIS están poco relacionadas con la tuya: el resultado es poco informativo.', 'El modelo ajustado no respondió; se muestra la estimación del modo rápido.']; 12.1 s

(«Segundos» = cifras + explicación. Una explicación con 0,0 s sale de la caché de una ejecución anterior; sin caché tarda unos 3 s. «none» = pregunta nueva, sin equivalente en el CIS.)

## Distribuciones completas (porcentajes)

- **Vivienda** — rápido: Muy de acuerdo 20 %, Bastante de acuerdo 13 %, Regular 39 %, Poco de acuerdo 11 %, Nada de acuerdo 17 % | con LLM: Muy de acuerdo 18 %, Bastante de acuerdo 16 %, Regular 35 %, Poco de acuerdo 14 %, Nada de acuerdo 18 % | ajustado: Muy de acuerdo 42 %, Bastante de acuerdo 33 %, Regular 1 %, Poco de acuerdo 12 %, Nada de acuerdo 12 %
- **Soledad** — rápido: Nunca 30 %, Casi nunca 34 %, A veces 12 %, A menudo 16 %, Siempre 9 % | con LLM: Nunca 24 %, Casi nunca 30 %, A veces 18 %, A menudo 17 %, Siempre 10 % | ajustado: Nunca 28 %, Casi nunca 28 %, A veces 28 %, A menudo 10 %, Siempre 7 %
- **Ia** — rápido: Mucho 26 %, Bastante 19 %, Poco 12 %, Nada 42 % | con LLM: Mucho 24 %, Bastante 23 %, Poco 19 %, Nada 34 % | ajustado: Mucho 39 %, Bastante 39 %, Poco 16 %, Nada 7 %
- **Medio ambiente** — rápido: Muy de acuerdo 33 %, Bastante de acuerdo 22 %, Regular 18 %, Poco de acuerdo 12 %, Nada de acuerdo 15 % | con LLM: Muy de acuerdo 27 %, Bastante de acuerdo 22 %, Regular 21 %, Poco de acuerdo 15 %, Nada de acuerdo 15 % | ajustado: Muy de acuerdo 39 %, Bastante de acuerdo 34 %, Regular 1 %, Poco de acuerdo 14 %, Nada de acuerdo 11 %
- **Salud** — rápido: Muy bueno 19 %, Bueno 41 %, Regular 26 %, Malo 10 %, Muy malo 4 % | con LLM: Muy bueno 20 %, Bueno 43 %, Regular 24 %, Malo 8 %, Muy malo 4 % | ajustado: Muy bueno 24 %, Bueno 45 %, Regular 15 %, Malo 10 %, Muy malo 6 %
- **Consumo** — rápido: Sí 21 %, No 79 % | con LLM: Sí 28 %, No 72 % | ajustado: Sí 65 %, No 35 %

## Qué se ve en la interfaz

- Encima de cada resultado: **«Modo que ha producido este resultado: …»** (Rápido, Con LLM o Ajustado) y el modo solicitado. En el ajustado se añade siempre la nota
  *«Experimental: menor error medido, no cumplió el criterio prerregistrado en el examen.»*
- Si hubo que volver al modo rápido (servidor apagado, Ollama sin responder, más de 7 opciones, fallo del modelo de lenguaje) sale un **aviso amarillo** con el motivo y la frase «Las cifras son las del modo rápido».
- En preguntas repetidas del CIS las cifras son estadística directa en los tres modos, y una nota lo dice.

## Prueba de caída del servidor del modelo ajustado

Con el servidor apagado (proceso terminado) se repitió la pregunta 1 en modo ajustado: el resultado salió igualmente, **producido por el modo Rápido**, `fallback = True`, motivo «el servidor del modelo ajustado no respondió», con el aviso
«El modelo ajustado no respondió; se muestra la estimación del modo rápido.» (12.1 s). Durante el guion, además, una petición real recibió un corte de conexión esporádico (`ConnectionResetError`): ese resultado también cayó al modo rápido con aviso, y por eso el cliente
reintenta ahora una vez antes de rendirse (el servidor tiene además una cola de 64 conexiones y registra los errores).

## Lo que se ve raro (honesto)

1. **Todas las preguntas del guion son «sin ancla» y de confianza baja**: son preguntas nuevas, sin equivalente en el CIS; es lo esperado y lo que más cuesta (error típico medido en preguntas nuevas: ≈ 13–25 puntos en sí/no y ≈ 9–13 con 3 o más opciones, según el modo). Para enseñar el modo directo hay que usar una pregunta que sí exista en el CIS (los ejemplos precalculados de la barra lateral).
2. **Los tres modos no coinciden en el signo en algunas preguntas**: IA (rápido «Nada» 42 %, con LLM «Nada» 34 %, ajustado «Mucho» 39 % y «Bastante» 39 %) y consumo (rápido «No» 79 %, ajustado «Sí» 65 %). No hay una verdad contra la que comparar en directo: el
   examen oficial solo mide el error medio (ajustado 6,84 pp frente a 8,11 del rápido y 7,91 del modo con LLM), sin garantías sobre cada pregunta.
3. **Empates exactos en el modo ajustado** (soledad: Nunca / Casi nunca / A veces, 27,7 % cada una; IA: Mucho y Bastante, 38,6 %): el modelo devuelve sus logits en bf16 y, con valores de unas decenas, bf16 solo distingue saltos de ≈ 0,1; letras con logits casi iguales
   salen idénticas. Es una limitación de resolución, no un error de cálculo. No se corrigió para no cambiar el modelo ni los resultados evaluados; calcular la última capa en fp32 sería una mejora futura (habría que reevaluarla).
4. **Poca masa en la opción intermedia** del modo ajustado en escalas de acuerdo («Regular» 0,6 % en vivienda y 1,0 % en medio ambiente): en las escalas de acuerdo del CIS la opción central suele recibir bastante más; no se ha investigado la causa y conviene vigilarlo.
5. **El modo con LLM tarda unos 100 s por pregunta**, y en el registro aparecen «Event loop is closed» al primer intento de la narración de cada pregunta (se repite solo y sale bien): ruido conocido del cliente asíncrono, sin efecto en las cifras. Sin el modelo de lenguaje el resto del sistema no se ve afectado.
6. **Las explicaciones del LLM** son correctas en cifras (se comprueban) pero genéricas; en el modo ajustado el modelo de lenguaje llegó a escribir «se apoya en estudios similares», lo cual no es exacto: las cifras vienen del modelo ajustado. La plantilla ya lo dice bien; el texto libre del LLM no se puede garantizar.
