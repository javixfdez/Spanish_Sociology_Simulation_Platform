from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .topics import Topic


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_provider: Literal["ollama", "mock", "external"] = "ollama"
    ollama_model: str = "qwen3:8b"
    ollama_base_url: str = "http://localhost:11434"
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
