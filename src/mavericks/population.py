import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

REQUIRED = ["person_id", "peso", "sexo", "edad", "grupo_edad", "nivel_estudios", "version_poblacion"]


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


DEFAULT_DIMENSIONS = ["grupo_edad", "sexo", "nivel_estudios", "ccaa", "tamuni", "ingresos_hogar_tramo"]
_ALIASES = {
    "edad": "grupo_edad",
    "estudios": "nivel_estudios",
    "educacion": "nivel_estudios",
    "nivel_educativo": "nivel_estudios",
    "comunidad": "ccaa",
    "comunidad_autonoma": "ccaa",
    "region": "ccaa",
    "territorio": "ccaa",
    "habitat": "tamuni",
    "tamano_municipio": "tamuni",
    "municipio": "tamuni",
    "ingresos": "ingresos_hogar_tramo",
    "renta": "ingresos_hogar_tramo",
    "ingresos_hogar": "ingresos_hogar_tramo",
}
_EXCLUDED_COLUMNS = {"person_id", "peso", "version_poblacion"}


def resolve_dimensions(requested, frame):
    """Orden de segmentación: primero lo pedido (con alias), luego las dimensiones por defecto.

    Solo se usan columnas presentes con al menos dos categorías.
    """
    ordered = []
    for name in list(requested or []) + DEFAULT_DIMENSIONS:
        key = "".join(c for c in unicodedata.normalize("NFKD", str(name)) if not unicodedata.combining(c))
        key = key.strip().lower().replace(" ", "_")
        key = _ALIASES.get(key, key)
        if key in frame.columns and key not in ordered and frame[key].nunique() > 1:
            ordered.append(key)
    return ordered


def _split(frame, node, dimension, budget, min_size):
    """Divide un nodo por una dimensión. Devuelve la lista de hijos o None si no hay división útil.

    Los hijos pequeños se agrupan en un hijo «resto»; ``budget`` limita el número de hijos.
    """
    rows = frame.loc[node["rows"]]
    groups = {
        category: part.index.to_numpy()
        for category, part in rows.groupby(dimension, sort=True, dropna=False)
        if len(part)
    }
    if len(groups) < 2:
        return None
    weight = rows.peso.groupby(rows[dimension], dropna=False).sum()
    ranked = sorted(groups, key=lambda c: (-weight[c], str(c)))
    kept = [c for c in ranked if len(groups[c]) >= min_size]
    rest = [c for c in ranked if c not in kept]
    if len(kept) > budget - (1 if rest else 0):
        keep_n = max(0, budget - 1)
        rest = kept[keep_n:] + rest
        kept = kept[:keep_n]
    if not kept:
        return None
    if rest and sum(len(groups[c]) for c in rest) < min_size:
        # Un resto demasiado pequeño se funde con el hijo más pequeño en vez de crear un segmento ínfimo.
        smallest = kept.pop()
        rest = [smallest] + rest
    if len(kept) + (1 if rest else 0) < 2:
        return None
    children = []
    for category in kept:
        children.append({"rows": groups[category], "constraints": {**node["constraints"], dimension: [category]}})
    if rest:
        merged = np.concatenate([groups[c] for c in rest])
        children.append({"rows": merged, "constraints": {**node["constraints"], dimension: sorted(rest, key=str)}})
    return children


def build_segments(repository, max_segments=24, min_size=10, dimensions=None):
    """Segmentación jerárquica por pesos.

    Parte de un único segmento y divide siempre el de mayor peso por la siguiente dimensión de la lista
    ``dimensions`` (después las de DEFAULT_DIMENSIONS). Las celdas pequeñas se agrupan en un segmento
    «resto» en lugar de eliminar dimensiones enteras, de modo que se obtienen ~``max_segments`` segmentos.
    """
    if max_segments < 1 or min_size < 1:
        raise ValueError("Límites de segmentación inválidos")
    frame = repository.frame.reset_index(drop=True)
    order = resolve_dimensions(dimensions, frame)
    nodes = [{"rows": np.arange(len(frame)), "constraints": {}, "done": False}]
    while len(nodes) < max_segments:
        pending = [n for n in nodes if not n["done"]]
        if not pending:
            break
        node = max(pending, key=lambda n: frame.peso.to_numpy()[n["rows"]].sum())
        children = None
        for dimension in order:
            if dimension in node["constraints"]:
                continue
            children = _split(frame, node, dimension, max_segments - len(nodes) + 1, min_size)
            if children:
                break
        if not children:
            node["done"] = True
            continue
        nodes = [n for n in nodes if n is not node]
        nodes.extend({**c, "done": False} for c in children)
    nodes.sort(key=lambda n: (-frame.peso.to_numpy()[n["rows"]].sum(), str(sorted(n["constraints"].items()))))
    columns = [c for c in frame.columns if c not in _EXCLUDED_COLUMNS and frame[c].nunique() <= 40]
    provenance = [f"{k}: {v}" for k, v in repository.metadata.variable_provenance.items() if v != "observed"]
    segments = []
    for i, node in enumerate(nodes):
        group = frame.loc[node["rows"]]
        total = group.peso.sum()
        profile, homogeneous, merged = {}, [], {}
        for dimension in order:
            categories = node["constraints"].get(dimension)
            if not categories:
                continue
            if len(categories) == 1:
                profile[dimension] = str(categories[0])
                homogeneous.append(dimension)
            else:
                profile[dimension] = f"Otros ({len(categories)} categorías)"
                merged[dimension] = [str(c) for c in categories]
        distributions = {
            c: {str(k): float(v) for k, v in (group.groupby(c, dropna=False).peso.sum() / total).items()}
            for c in columns
        }
        rng = np.random.default_rng(int(node["rows"][0]))
        picks = rng.choice(len(group), size=min(3, len(group)), replace=False, p=(group.peso / total).to_numpy())
        segments.append(
            {
                "segment_id": f"S{i + 1:03d}",
                "population_weight": float(total),
                "n_synthetic": len(group),
                "demographic_profile": profile,
                "constraints": {d: [str(c) for c in cats] for d, cats in node["constraints"].items()},
                "homogeneous_dimensions": homogeneous,
                "merged_categories": merged,
                "socioeconomic_profile": distributions,
                "representative_profiles": group.iloc[picks].drop(columns="person_id").to_dict("records"),
                "imputation_warnings": provenance,
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
