"""Validación de cruces de dos variables del MRP: la lógica del script (celdas, resumen y documento), sin datos reales."""

import numpy as np
import pandas as pd
import pytest

from mavericks import mrp_cross_eval as cross

PAIR = {"options": ["Sí", "No"]}


def frame(rows):
    """rows: (sexo, grupo_edad, respuesta, peso, repeticiones)."""
    data, uid = [], 0
    for sexo, edad, answer, weight, times in rows:
        for _ in range(times):
            uid += 1
            data.append({"respondent_uid": uid, "answer": answer, "peso": weight, "sexo": sexo, "grupo_edad": edad})
    return pd.DataFrame(data)


def test_cells_need_at_least_fifty_respondents_and_use_weighted_shares():
    f = frame([("mujer", "18-24", "Sí", 2.0, 30), ("mujer", "18-24", "No", 1.0, 30), ("hombre", "18-24", "Sí", 1.0, 49), ("hombre", "65+", "No", 1.0, 80)])
    truth = cross.cell_truth(f, PAIR, crosses=(("sexo", "grupo_edad"),))
    assert set(truth) == {(("sexo", "mujer"), ("grupo_edad", "18-24")), (("sexo", "hombre"), ("grupo_edad", "65+"))}  # la celda de 49 queda fuera
    n, share = truth[(("sexo", "mujer"), ("grupo_edad", "18-24"))]
    assert n == 60 and share == pytest.approx([2 / 3, 1 / 3])  # 30·2 frente a 30·1


def test_invalid_categories_and_non_substantive_answers_are_ignored():
    f = frame([("mujer", "NC", "Sí", 1.0, 60), ("mujer", "18-24", "No sabe", 1.0, 60), ("mujer", "18-24", "Sí", 1.0, 55)])
    truth = cross.cell_truth(f, PAIR, crosses=(("sexo", "grupo_edad"),))
    assert list(truth) == [(("sexo", "mujer"), ("grupo_edad", "18-24"))] and truth[next(iter(truth))][0] == 55


def test_summary_compares_mrp_with_the_national_baseline_and_bootstraps_studies():
    rng = np.random.default_rng(1)
    rows = []
    for study in "ABCDEF":
        for cell in range(20):
            national = 6 + rng.normal(0, 1)
            rows.append({"series": "s" + study, "t_id": "id" + study, "t_study": study, "cross": "sexo×grupo_edad", "cell": str(cell), "n": 50 + 10 * cell, "n_options": 3,
                         "err_mrp": national - 2 + rng.normal(0, 0.3), "err_national": national})
    res = cross.summarize(pd.DataFrame(rows))
    total = res["total"]
    assert total["celdas"] == 120 and total["estudios"] == 6 and total["delta"] == pytest.approx(total["mae_mrp"] - total["mae_nacional"])
    assert total["ic"][0] < total["delta"] < total["ic"][1] < 0 and total["mrp_mejor"] > total["mrp_peor"]
    assert set(res["por_n"]) == {"50–99", "100–199", "200–∞"} and sum(b["celdas"] for b in res["por_n"].values()) == 120
    text = cross.report_text(res)
    assert text.startswith(cross.MARKER) and "120 celdas" in text and "Por cruce" in text and "sexo×grupo_edad" in text


def test_the_document_section_is_appended_once_and_replaced_without_touching_the_rest(tmp_path):
    doc = tmp_path / "mrp_resultados.md"
    doc.write_text("# Resultados\r\n\r\nTexto previo que no se toca.\r\n", encoding="utf-8", newline="")
    cross.write_doc(cross.MARKER + "\n\nprimera versión\n", doc)
    cross.write_doc(cross.MARKER + "\n\nsegunda versión\n", doc)
    text = doc.read_bytes().decode("utf-8")
    assert text.startswith("# Resultados\r\n\r\nTexto previo que no se toca.\r\n") and text.count(cross.MARKER) == 1
    assert "segunda versión" in text and "primera versión" not in text and "\r\n" in text


def test_the_window_never_reaches_the_reserved_studies():
    assert cross.T_WINDOW[1] < "2026-04-01" and cross.MIN_CELL == 50
