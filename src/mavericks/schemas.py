from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .topics import Topic


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_provider: Literal["ollama", "mock", "external"] = "ollama"
    ollama_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = Field(8192, ge=1024)
    anchor_max_shift: float = Field(0.15, gt=0, le=1)
    anchor_shrinkage: float = Field(25.0, ge=0)
    anchor_direct_similarity: float = Field(0.9, gt=0, le=1)
    anchor_prior_similarity: float = Field(0.55, gt=0, le=1)
    cis_dir: str = "data/processed"
    stability_runs: int = Field(1, ge=1, le=10)
    llm_assessment: bool = True
    few_shot: int = Field(0, ge=0, le=8)
    sources: str = ""
    context: bool = False
    sharpen: bool = False
    context_path: str = "data/processed_v2/context.json"
    retriever: Literal["tfidf", "embed", "hybrid"] = "tfidf"
    embedding_model: str = "intfloat/multilingual-e5-base"
    embedding_device: str = "cpu"
    max_evidence_hits: int = Field(8, ge=1, le=20)
    stability_temperature: float = Field(0.7, ge=0, le=2)
    calibration_path: str = "data/processed/calibration.json"
    interval_alpha: float = Field(0.1, gt=0, lt=1)
    min_calibration_cases: int = Field(20, ge=1)
    llm_temperature: float = Field(0, ge=0, le=2)
    llm_timeout: float = Field(120, gt=0)
    llm_max_concurrency: int = Field(1, ge=1, le=32)
    llm_retries: int = Field(2, ge=0, le=5)
    simulation_timeout: float = Field(900, gt=0)
    embedding_model_path: str = ""
    chroma_path: str = "data/processed/chroma"
    external_model: str = ""
    external_base_url: str = ""
    external_api_key: str = ""


class Question(BaseModel):
    question: str = Field(min_length=5)
    options: list[str] = Field(min_length=2)
    question_type: Literal["single", "multiple"] = "single"
    target_population: str = "Adultos residentes en hogares privados de España"
    optional_context: str = ""

    @model_validator(mode="after")
    def unique(self):
        if any(not o.strip() for o in self.options) or len(set(self.options)) != len(self.options):
            raise ValueError("Las opciones deben ser únicas y no vacías")
        return self


class Methodology(BaseModel):
    topic: Topic
    question_type: Literal["single", "multiple"]
    relevant_demographics: list[str]
    retrieval_queries: list[str]
    warnings: list[str] = []


class AssessmentItem(BaseModel):
    evidence_id: str
    relevance: Literal["direct", "indirect", "incompatible"]
    differences: list[str] = []
    supported_variables: list[str] = []
    extrapolations: list[str] = []


class Assessment(BaseModel):
    items: list[AssessmentItem]
    missing_information: list[str] = []


class Prediction(BaseModel):
    segment_id: str
    probabilities: dict[str, float]
    explanation: str
    evidence_ids: list[str] = []
    limitations: list[str] = []

    @model_validator(mode="after")
    def bounded(self):
        import math

        if not self.probabilities or any(
            not math.isfinite(p) or not 0 <= p <= 1 for p in self.probabilities.values()
        ):
            raise ValueError("Probabilidades inválidas")
        return self

    def renormalized(self, tolerance=0.15):
        """Corrige errores de redondeo del LLM en respuestas únicas: si la suma está dentro de la
        tolerancia se reescala a 1; fuera de ella se deja intacta para que la validación falle."""
        total = sum(self.probabilities.values())
        if total > 0 and abs(total - 1) <= tolerance and abs(total - 1) > 1e-9:
            return self.model_copy(update={"probabilities": {k: v / total for k, v in self.probabilities.items()}})
        return self

    def validate_context(self, segment_id, options, evidence_ids, question_type="single"):
        if self.segment_id != segment_id or set(self.probabilities) != set(options):
            raise ValueError("Segmento u opciones incorrectos")
        if not set(self.evidence_ids) <= set(evidence_ids):
            raise ValueError("evidence_ids inexistentes")
        if question_type == "single" and abs(sum(self.probabilities.values()) - 1) > 1e-6:
            raise ValueError("Las probabilidades no suman 1")
        return self


class Narrative(BaseModel):
    summary: str
    hypotheses: list[str] = []
    limitations: list[str] = []


class Evidence(BaseModel):
    evidence_id: str
    source: str
    study_id: str | None = None
    question_id: str | None = None
    question_text: str | None = None
    question_summary: str | None = None
    options: list[str] = []
    topic: str | None = None
    fieldwork_date: str | None = None
    target_population: str | None = None
    survey_mode: str | None = None
    sample_size: int | None = None
    available_demographics: list[str] = []
    original_file: str | None = None
    metadata: dict = {}
