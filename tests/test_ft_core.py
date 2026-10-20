import numpy as np
import pytest

from mavericks.ft import core
from mavericks.ft.data import MAX_SUBGROUPS, profile_text, real_question, sample_subgroups


def row(**kw):
    base = {"id": "Q1|nacional|", "profile": "Población: adultos residentes en España", "source": "CIS", "year": "2025", "question": "¿Cree que sube?",
            "options": ["Sí", "No", "Depende"], "props": [0.5, 0.3, 0.2], "language": "es"}
    return {**base, **kw}


def test_shuffle_on_the_fly_reorders_options_and_distribution_together_and_is_deterministic():
    r = row()
    prompt, props = core.build_item(r, epoch=0, seed=0)
    again = core.build_item(r, epoch=0, seed=0)
    assert (prompt, props) == again and sorted(props) == [0.2, 0.3, 0.5]
    options = [line[3:] for line in prompt.splitlines() if line[:2] in ("A)", "B)", "C)")]
    for option, p in zip(options, props, strict=True):  # cada opción conserva su proporción
        assert {"Sí": 0.5, "No": 0.3, "Depende": 0.2}[option] == p
    assert any(core.build_item(r, epoch=e, seed=0)[1] != [0.5, 0.3, 0.2] for e in range(6))  # el orden cambia entre épocas
    assert core.build_item(r, shuffle=False)[1] == [0.5, 0.3, 0.2]


def test_prompt_states_source_language_and_profile():
    es = core.build_item(row(), shuffle=False)[0]
    en = core.build_item(row(source="Eurobarómetro", language="en", question="Do you trust the EU?"), shuffle=False)[0]
    assert "Fuente: CIS, 2025 (pregunta en español)" in es and "Fuente: Eurobarómetro, 2025 (pregunta en inglés)" in en
    assert profile_text("sexo", "mujer") == "Población: mujeres adultas residentes en España"
    assert profile_text("grupo_edad", "75+") == "Población: personas de 75 años o más residentes en España"
    assert es.splitlines()[-1] == "Respuesta:" and es.count("\n") >= 5


@pytest.mark.parametrize(
    "text,keep",
    [
        ("POLITICAL INTEREST INDEX (D71 SUMMARIZED) - STRONG", False),
        ("TRUST IN INSTITUTIONS: REG/LOCAL PUBLIC AUTHORITIES", False),
        ("Do you think that mental health patients are judged differently than other patients?", True),
        ("Q4a Which of the following sources do you use for getting information on current affairs?", True),
        ("MEDIA USE INDEX: HIGH", False),
        ("Yes or no", False),
    ],
)
def test_external_question_filter(text, keep):
    assert (real_question(text) is not None) == keep
    if keep and text.startswith("Q4a"):
        assert real_question(text).startswith("Which of the following")  # sin el código de variable


def test_subgroup_sampling_keeps_all_national_and_caps_each_dimension():
    examples = [{"id": f"Q|nacional|{i}", "evidence_id": "Q", "scope": "nacional", "dim": None} for i in range(2)]
    examples += [{"id": f"Q|ccaa|{i}", "evidence_id": "Q", "scope": "subgrupo", "dim": "ccaa"} for i in range(19)]
    examples += [{"id": f"Q|sexo|{i}", "evidence_id": "Q", "scope": "subgrupo", "dim": "sexo"} for i in range(2)]
    kept = sample_subgroups(examples)
    assert sum(e["scope"] == "nacional" for e in kept) == 2
    assert sum(e["dim"] == "ccaa" for e in kept) == MAX_SUBGROUPS and sum(e["dim"] == "sexo" for e in kept) == 2
    assert [e["id"] for e in sample_subgroups(examples)] == [e["id"] for e in kept]  # semilla fija


def test_kl_loss_ignores_invalid_letters_and_weights_by_n():
    torch = pytest.importorskip("torch")
    logits = torch.tensor([[2.0, 0.0, -1.0, 5.0, 5.0, 5.0, 5.0]])
    ids = list(range(7))
    valid = torch.arange(7)[None, :] < torch.tensor([3])[:, None]
    log_q = torch.log_softmax(logits.masked_fill(~valid, float("-inf")), dim=-1)
    loss, kl = core.kl_loss(log_q, valid, [[0.7, 0.2, 0.1]], torch.tensor([1.0]))
    expected = sum(p * np.log(p / q) for p, q in zip([0.7, 0.2, 0.1], torch.softmax(logits[0, :3], 0).tolist(), strict=True))
    assert float(loss) == pytest.approx(expected, rel=1e-5) and not torch.isnan(loss) and ids
