from mavericks.etl.questionnaire import compose_text, extract_stems, stem_for

TEXT = """Estudio:  Barómetro de marzo 2026
Clave: ECIS3546
1 de 11
Filtros:
P.2 si P.1 distinto de 8
P1. Ante los recientes bombardeos, ¿está Ud. muy preocupado/a,
bastante, algo, poco o nada preocupado/a? (ENTREVISTADOR/A: LEER)
[P1] (CATEGORÍA NOMINAL)
Muy preocupado/a ................................ 1
Poco ............ 2
P.20 A continuación, voy a mencionarle los nombres de algunos líderes
y nos gustaría que nos dijera si los conoce
[LIDERESCONOCE] (NUMÉRICA ORDINA.)
 Conoce No conoce N.C.
Pedro Sánchez 1 7 9
P.6 ¿Su coche es de gasolina, diésel o eléctrico? (ENTREVISTADOR/A: RESPUESTA MÚLTIPLE)
[P6] (MÚLTIPLE)
Gasolina ........ 1
"""


def test_extract_stems_cleans_codes_pages_and_directives():
    stems = extract_stems(TEXT)
    assert stems["P1"] == "Ante los recientes bombardeos, ¿está Ud. muy preocupado/a, bastante, algo, poco o nada preocupado/a?"
    assert "Estudio" not in stems["P1"] and "ENTREVISTADOR" not in stems["P6"]
    assert stems["P6"] == "¿Su coche es de gasolina, diésel o eléctrico?"


def test_battery_items_use_stem_without_suffix_and_append_item():
    stems = extract_stems(TEXT)
    assert stem_for("LIDERESCONOCE_3", stems) == stems["LIDERESCONOCE"]
    shared = {stems["LIDERESCONOCE"]: 4}
    text = compose_text("Pedro Sánchez", ["LIDERESCONOCE_1"], stems, shared)
    assert text.endswith("[Pedro Sánchez]") and text.startswith("A continuación")
    assert compose_text("Grado de preocupación", ["P1"], stems, {}) == stems["P1"]
    assert compose_text("x", ["P99"], stems, {}) is None
