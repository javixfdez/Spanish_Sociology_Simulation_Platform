"""La interfaz sigue mostrando de dónde sale cada respuesta y cuánta confianza tiene (rango habitual, avisos, modo)."""

import re

from streamlit.testing.v1 import AppTest

from mavericks import ui_charts, ui_common


def _texts(at):
    return [e.value for e in (*at.markdown, *at.caption, *at.warning, *at.info, *at.subheader)]


def test_usual_range_band_is_drawn_and_labels_do_not_sit_on_it():
    distribution = {"Sí": 0.6, "No": 0.4}
    core = {"Sí": (0.5, 0.7), "No": (0.3, 0.5)}
    fig = ui_charts.distribution_bars(distribution, None, core=core)
    band = [t for t in fig.data if t.type == "scatter" and t.mode == "markers"]
    labels = [t for t in fig.data if t.type == "scatter" and t.mode == "text"]
    assert len(band) == 1 and band[0].error_x.color == ui_charts.CORE_BAND
    assert len(labels) == 1 and set(labels[0].text) == {"60.0 %", "40.0 %"}
    assert dict(zip(labels[0].y, labels[0].x, strict=True)) == {"Sí": 70.0, "No": 50.0}  # a la derecha del extremo de la banda


def test_chart_without_band_or_interval_still_labels_the_bars():
    fig = ui_charts.distribution_bars({"A": 0.25, "B": 0.75})
    assert [list(t.text) for t in fig.data if t.type == "scatter"] == [["75.0 %", "25.0 %"]]


def _produced_by_app():
    import streamlit as st

    from mavericks.ui_result import render_produced_by

    st.session_state.setdefault("n", 0)
    render_produced_by(
        {"producido_por": {"etiqueta": "Rápido", "solicitado": "ajustado", "fallback": True, "motivo": "el servidor del modelo ajustado no respondió", "experimental": "Experimental: no cumplió el criterio prerregistrado.", "nota": "Nota de prueba"}}
    )


def test_mode_label_fallback_and_experimental_note_are_visible():
    at = AppTest.from_function(_produced_by_app).run()
    assert not at.exception
    text = " ".join(_texts(at))
    assert "Rápido" in text and "solicitado: Ajustado" in text
    assert "Se ha vuelto al modo rápido" in text and "Experimental" in text and "Nota de prueba" in text


def _measured_app():
    from mavericks.ui_result import render_measured

    options = ["Sí", "No"]
    national = {"n": 2598, "n_eff": 1113.0, "distribution": {"Sí": 0.6, "No": 0.4}, "margin": {"Sí": 0.029, "No": 0.029}, "interval": {"Sí": (0.57, 0.63), "No": (0.37, 0.43)}}
    small = {"label": "Sexo: mujer · edad: 18-24", "n": 12, "small": True, "distribution": {"Sí": 0.5, "No": 0.5}, "margin": {"Sí": 0.2, "No": 0.2}}
    assert list(national["distribution"]) == options
    render_measured({"measured": {"study_id": "3400", "evidence_id": "CIS3400_X", "fieldwork_end": "2024-10-21", "national": national, "crosses": [small], "age_warning": "Dato de octubre de 2024: la opinión puede haber cambiado desde entonces."}})


def test_measured_block_shows_study_date_n_margin_old_data_and_small_group_warnings():
    at = AppTest.from_function(_measured_app).run()
    assert not at.exception
    text = " ".join(_texts(at))
    assert "Lo que midió el CIS" in text and "Estudio 3400" in text and "2024-10-21" in text and "2598 encuestados" in text and "margen de error" in text
    assert "Dato de octubre de 2024: la opinión puede haber cambiado desde entonces." in text
    assert "12 encuestados" in text and "Muestra pequeña" in text


def _luminance(color):
    channels = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    r, g, b = (c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def test_confidence_pills_have_readable_contrast_with_white_text():
    pills = dict(re.findall(r"\.pill\.(\w+) \{background:(#[0-9A-Fa-f]{6});\}", ui_common.CSS))
    assert set(pills) == {"alta", "media", "baja"}
    for level, color in pills.items():
        assert 1.05 / (_luminance(color) + 0.05) >= 4.5, level


def test_fixed_footer_leaves_room_for_the_sidebar_and_the_page_has_bottom_padding():
    assert 'data-collapsed="false"' in ui_common.CSS and ".brand-footer" in ui_common.CSS
    assert re.search(r"padding-bottom:\s*3\.5rem", ui_common.CSS)
