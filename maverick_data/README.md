# maverick_data

Microdatos publicos para el reto Maverick (poblacion sintetica de la sociedad espanola).

- raw/general/        Lo leen siempre todos los agentes.
  - poblacion/        Censo 2021 y EPA: base demografica de la poblacion sintetica.
  - cis/              Barometros mensuales y estudios de tema mixto.
- raw/vivienda/       ECV 2022-2025 (tenencia, renta, gastos). CIS 3532 cuando se publique.
- raw/salud/          Encuesta de Salud de Espana 2023 y estudios CIS de salud.
- raw/ia_tecnologia/  TIC-H 2022-2025, FECYT 2022 y 2024, estudios CIS de tecnologia.
- raw/consumo/        EPF 2022-2025 y CIS Indice de Confianza del Consumidor.
- raw/medio_ambiente/ Estudios CIS de clima y medio ambiente.
- raw/soledad/        Pendiente: CIS 3535, 3486 y 3508.
- raw/_excluidos/     Pre y postelectorales autonomicos (muestra de una sola CCAA).
- catalogo.csv        Un fichero por fila: ruta, fuente, titulo, anio, tema, tamano.

Los ficheros de raw/ no se modifican. Los datos procesados irian en interim/ y processed/.
En GitHub va solo el codigo y catalogo.csv (limite de 100 MB por fichero).
Los datos se suben como dataset privado de Hugging Face.
