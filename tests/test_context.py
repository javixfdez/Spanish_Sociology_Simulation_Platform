import asyncio
import json
from datetime import date

import pandas as pd

from mavericks.anchor import AnchorStore
from mavericks.context import ContextStore, build_context_index, chunk, infer_date
from mavericks.evidence import EvidenceRepository
from mavericks.graph import simulate
from mavericks.llm import LLMService, MockLLMProvider
from mavericks.population import PopulationLoader, PopulationRepository, create_demo
from mavericks.schemas import Prediction, Question, Settings

NOTE = """Fuente: INE, nota de prensa IPC0526
Nota de Prensa: IPC. Mayo 2026.
12 de junio de 2026
Principales resultados
La tasa de variación anual del IPC del mes de mayo se situó en el 3,2%, la misma que la registrada en abril.
La inflación subyacente aumentó dos décimas, hasta el 3,0%, y los precios de los alimentos siguieron subiendo."""


def test_infer_date_uses_text_then_filename_year():
    assert infer_date("INE_nota_IPC0526", NOTE) == date(2026, 6, 12)
    assert infer_date("Informe_soledad_2024", "sin fecha") == date(2024, 12, 31)
    assert infer_date("sin_ano", "sin fecha") is None


def test_build_index_skips_documents_published_after_cutoff(tmp_path):
    folder = tmp_path / "consumo" / "contexto"
    folder.mkdir(parents=True)
    (folder / "INE_nota_IPC0526.txt").write_text(NOTE, encoding="utf-8")
    (folder / "INE_nota_IPC0826.txt").write_text(NOTE.replace("12 de junio de 2026", "12 de septiembre de 2026"), encoding="utf-8")
    index = build_context_index(tmp_path)
    assert {e["doc"] for e in index["entries"]} == {"INE_nota_IPC0526.txt"}
    assert index["skipped"] == ["INE_nota_IPC0826.txt"]
    assert all(e["published"] == "2026-06-12" and e["topic"] == "consumo" for e in index["entries"])
    assert all(len(c) >= 40 for c in chunk(NOTE))


def test_snippets_respect_strict_temporal_cutoff_and_topic():
    entries = [
        {"doc": "viejo", "topic": "consumo", "published": "2026-01-10", "text": "La inflación de los precios de los alimentos subió en enero."},
        {"doc": "nuevo", "topic": "consumo", "published": "2026-06-12", "text": "La inflación de los precios de los alimentos subió en mayo."},
        {"doc": "salud", "topic": "salud", "published": "2026-01-10", "text": "La inflación de los precios de los alimentos y la salud."},
    ]
    store = ContextStore(entries)
    got = store.snippets("inflación precios alimentos", "consumo", "2026-03-01")
    assert [g["source"] for g in got] == ["viejo"]  # ni lo posterior al corte ni otro tema
    assert store.snippets("inflación precios", "consumo", "2026-01-10") == []  # estricto: el mismo día no vale


def test_snapshot_uses_latest_nonreserved_barometer_before_cutoff(tmp_path):
    studies = pd.DataFrame(
        [
            {"estudio_id": "1", "titulo": "CIS_1_barometro_enero", "fieldwork_start": "2026-01-05", "fieldwork_end": "2026-01-10", "reservado_examen": False},
            {"estudio_id": "2", "titulo": "CIS_2_barometro_febrero", "fieldwork_start": "2026-02-02", "fieldwork_end": "2026-02-07", "reservado_examen": False},
            {"estudio_id": "3", "titulo": "CIS_3_barometro_abril", "fieldwork_start": "2026-04-06", "fieldwork_end": "2026-04-10", "reservado_examen": True},
        ]
    )
    studies.to_parquet(tmp_path / "cis_studies.parquet")
    people = pd.DataFrame([{"estudio_id": "2", "respondent_uid": str(i), "peso": 1.0} for i in range(400)])
    people.to_parquet(tmp_path / "cis_respondents.parquet")
    answers = pd.DataFrame(
        [
            {"estudio_id": "2", "respondent_uid": str(i), "question_uid": "CIS2_PESPANNA1",
             "answer": "El paro" if i < 100 else "La inmigración" if i < 300 else "N.S.", "selected": None}
            for i in range(400)
        ]
    ).assign(selected=lambda d: d.selected.astype("boolean"))
    answers.to_parquet(tmp_path / "cis_responses.parquet")
    store = ContextStore([], tmp_path)
    snap = store.snapshot(AnchorStore(tmp_path), "2026-04-06")
    assert snap["source"] == "Barómetro CIS 2" and "La inmigración 67%" in snap["text"] and "El paro 33%" in snap["text"]
    assert store.snapshot(AnchorStore(tmp_path), "2026-01-10") is None  # ninguno terminó antes
    assert "3" not in snap["source"]  # el barómetro reservado nunca


def test_graph_passes_context_only_before_the_case_cutoff(tmp_path):
    path = create_demo(tmp_path / "fixtures")
    frame, meta = PopulationLoader().load(path)
    evidence = EvidenceRepository(tmp_path / "e.sqlite")
    evidence.ingest(path.parent / "evidence.json")
    entries = [
        {"doc": "antes", "topic": "ia_tecnologia", "published": "2025-01-01", "text": "Uso de inteligencia artificial en aplicaciones de pago."},
        {"doc": "despues", "topic": "ia_tecnologia", "published": "2026-05-01", "text": "Uso de inteligencia artificial en aplicaciones de pago nuevo."},
    ]
    seen = []

    class Spy(MockLLMProvider):
        async def generate(self, schema, payload, temperature=None):
            if schema is Prediction:
                seen.append(payload)
            return await super().generate(schema, payload)

    llm = LLMService(Spy(), Settings(llm_provider="mock", llm_retries=0), tmp_path / "c")
    asyncio.run(
        simulate(Question(question="¿Usas aplicaciones de inteligencia artificial de pago?", options=["Sí", "No"]),
                 PopulationRepository(frame, meta), evidence, llm, 3,
                 retrieval_filters={"cutoff": "2026-03-01"}, context_store=ContextStore(entries))
    )
    assert seen and all([c["source"] for c in p["context"]] == ["antes"] for p in seen)
    seen.clear()
    asyncio.run(simulate(Question(question="¿Usas aplicaciones de inteligencia artificial de pago?", options=["Sí", "No"]),
                         PopulationRepository(frame, meta), evidence, LLMService(Spy(), Settings(llm_provider="mock", llm_retries=0), tmp_path / "c2"), 3))
    assert seen and all("context" not in p for p in seen)  # sin flag: comportamiento por defecto
    json.dumps(seen[0])
