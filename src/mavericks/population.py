import json
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

REQUIRED = ["person_id", "peso", "sexo", "edad", "grupo_edad", "nivel_estudios", "version_poblacion"]
DIMENSIONS = ["grupo_edad", "sexo", "nivel_estudios"]


class PopulationMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")
    population_version: str
    synthetic: bool = False
    variable_provenance: dict[str, str] = {}
    categories: dict[str, list] = {}
    coverage: str = "Adultos en hogares privados; excluye establecimientos colectivos"


class PopulationValidator:
    def validate(self, frame, metadata):
        missing = set(REQUIRED) - set(frame.columns)
        if missing:
            raise ValueError(f"Columnas obligatorias ausentes: {sorted(missing)}")
        if frame.empty or frame[REQUIRED].isna().any().any():
            raise ValueError("Población vacía o valores obligatorios ausentes")
        if frame.person_id.duplicated().any():
            raise ValueError("person_id duplicados")
        age = pd.to_numeric(frame.edad, errors="raise")
        weights = pd.to_numeric(frame.peso, errors="raise")
        if (age < 18).any() or not np.isfinite(age).all():
            raise ValueError("Edad inválida")
        if (weights <= 0).any() or not np.isfinite(weights).all():
            raise ValueError("Peso inválido")
        if set(frame.version_poblacion.astype(str)) != {metadata.population_version}:
            raise ValueError("Versión de población inconsistente")
        for column, categories in metadata.categories.items():
            if column in frame and not set(frame[column].dropna()) <= set(categories):
                raise ValueError(f"Categorías inválidas: {column}")
        return frame.assign(peso=weights, edad=age)


class PopulationLoader:
    def load(self, path, metadata_path=None):
        path = Path(path)
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
        meta_path = Path(metadata_path) if metadata_path else path.parent / "population_metadata.json"
        if not meta_path.exists():
            raise ValueError("Se requiere population_metadata.json; no se puede inferir procedencia")
        meta = PopulationMetadata.model_validate_json(meta_path.read_text(encoding="utf-8-sig"))
        return PopulationValidator().validate(frame, meta), meta


class PopulationRepository:
    def __init__(self, frame, metadata):
        self.frame = PopulationValidator().validate(frame, metadata)
        self.metadata = metadata


def build_segments(repository, max_segments=24, min_size=1):
    if max_segments < 1 or min_size < 1:
        raise ValueError("Límites de segmentación inválidos")
    frame = repository.frame
    # Reduce resolution by removing whole dimensions, preserving interpretable subgroups.
    dimensions = DIMENSIONS.copy()
    while dimensions:
        groups = list(frame.groupby(dimensions, sort=True, dropna=False))
        if len(groups) <= max_segments and min(len(g) for _, g in groups) >= min_size:
            break
        dimensions.pop()
    if not dimensions:
        groups = [((), frame)]
    segments = []
    for i, (keys, group) in enumerate(groups):
        keys = keys if isinstance(keys, tuple) else (keys,)
        distributions = {}
        for column in frame.columns:
            if column in {"person_id", "peso", "version_poblacion"}:
                continue
            if frame[column].nunique() <= 40:
                frequencies = group.groupby(column, dropna=False).peso.sum() / group.peso.sum()
                distributions[column] = {str(k): float(v) for k, v in frequencies.items()}
        segments.append(
            {
                "segment_id": f"S{i + 1:03d}",
                "population_weight": float(group.peso.sum()),
                "n_synthetic": len(group),
                "demographic_profile": dict(zip(dimensions, map(str, keys))),
                "socioeconomic_profile": distributions,
                "representative_profiles": group.drop(columns="person_id").head(3).to_dict("records"),
                "imputation_warnings": [
                    f"{k}: {v}" for k, v in repository.metadata.variable_provenance.items() if v != "observed"
                ],
            }
        )
    if not np.isclose(sum(s["population_weight"] for s in segments), frame.peso.sum()):
        raise ValueError("Pérdida de peso al segmentar")
    return segments


def create_demo(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(120):
        rows.append(
            dict(
                person_id=f"DEMO_{i}",
                peso=1 + i % 5,
                edad=20 + i % 60,
                grupo_edad="18-34" if 20 + i % 60 < 35 else ("35-54" if 20 + i % 60 < 55 else "55+"),
                sexo=["mujer", "hombre"][i % 2],
                nivel_estudios=["basicos", "superiores"][i % 2],
                ingresos_hogar_tramo=["bajo", "medio", "alto"][i % 3],
                version_poblacion="DEMO-SYNTHETIC-v1",
            )
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(path / "population.csv", index=False)
    meta = PopulationMetadata(
        population_version="DEMO-SYNTHETIC-v1",
        synthetic=True,
        variable_provenance={k: "synthetic" for k in frame.columns},
    )
    (path / "population_metadata.json").write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    evidence = [
        dict(
            evidence_id="DEMO_AI",
            source="DEMO / SYNTHETIC",
            study_id="DEMO",
            question_text="¿Utiliza aplicaciones de inteligencia artificial?",
            options=["Sí", "No"],
            topic="ia_tecnologia",
            metadata={"synthetic": True},
        )
    ]
    (path / "evidence.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return path / "population.csv"
