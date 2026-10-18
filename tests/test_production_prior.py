from pathlib import Path

import numpy as np
import pytest

from mavericks import service as svc
from mavericks.prior_v2 import learned, production

ROOT = Path(__file__).resolve().parent.parent / "src" / "mavericks"


class FakeEmb:
    """Embeddings deterministas por texto (8 dimensiones) con la misma interfaz que la caché y que el embedder e5."""

    def vec(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        v = rng.normal(size=8)
        return v / np.linalg.norm(v)

    def question(self, evidence_id):
        return self.vec("q:" + evidence_id)

    def option_query(self, label):
        return self.vec("o:" + label)

    def encode_passages(self, texts):
        return np.stack([self.vec("q:" + t) for t in texts])

    def encode_queries(self, texts):
        return np.stack([self.vec("o:" + t) for t in texts])


def questions(n=60):
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        options = ["Sí", "No"] if i % 2 else ["Mucho", "Bastante", "Poco"]
        props = rng.dirichlet(np.ones(len(options)))
        out.append(learned.TrainQuestion(f"CIS{i}_P1", str(i), "2025-01-01", options, props.tolist(), [False] * len(options)))
    return out


def test_fit_predict_and_save_roundtrip(tmp_path):
    emb = FakeEmb()
    featurizer, model = production.fit(questions(), emb, dims=4, hidden=(8,), seed=0)
    prior = production.ProductionPrior(featurizer, model, {"max_options": 7})
    p = prior.predict("¿Le preocupa el alquiler?", ["Mucho", "Bastante", "Poco"], emb)
    assert p.shape == (3,) and p.sum() == pytest.approx(1)
    assert prior.supports(["a", "b"]) and not prior.supports(["a"] * 8) and not prior.supports(["a"])


def stub_resources(model=None, error=None):
    return svc.Resources(None, None, None, {}, None, prior_model=model, prior_model_error=error)


class StubModel:
    meta = {"max_options": 7, "trained_at": "x", "cutoff_date": "y", "n_questions": 1}

    def __init__(self, probabilities=None, boom=False):
        self.probabilities, self.boom = probabilities, boom

    def supports(self, options):
        return 2 <= len(options) <= 7

    def predict(self, question, options, embedder):
        if self.boom:
            raise AssertionError("no debe llamarse")
        return np.array(self.probabilities)


def test_blend_is_a_50_50_mean_and_reports_the_method(monkeypatch):
    monkeypatch.setattr(svc, "get_embedder", lambda: object())
    knn = {"a": 0.8, "b": 0.2}
    mixed, used = svc._blend_prior(stub_resources(StubModel([0.2, 0.8])), "q", ["a", "b"], knn, True)
    assert mixed == pytest.approx({"a": 0.5, "b": 0.5}) and used["metodo"] == "media_knn_mlp" and used["pesos"] == {"knn": 0.5, "mlp": 0.5}


@pytest.mark.parametrize(
    "resources,options,embedder,fragment",
    [
        (stub_resources(None, "No hay modelo del prior aprendido"), ["a", "b"], object(), "No hay modelo"),
        (stub_resources(StubModel([0.5, 0.5])), ["a"] * 9, object(), "cubre de 2 a 7"),
        (stub_resources(StubModel([0.5, 0.5])), ["a", "b"], None, "e5 no disponible"),
    ],
)
def test_falls_back_to_knn_and_says_why(monkeypatch, resources, options, embedder, fragment):
    monkeypatch.setattr(svc, "get_embedder", lambda: embedder)
    knn = {o: 1 / len(set(options)) for o in options}
    names = list(dict.fromkeys(options))
    mixed, used = svc._blend_prior(resources, "q", names, {o: 1 / len(names) for o in names}, True)
    assert used["metodo"] == "knn" and fragment in used["motivo"]
    assert mixed == pytest.approx({o: 1 / len(names) for o in names}) and knn


def test_validation_and_backtest_never_use_the_production_model():
    # con filtros (validación) el modelo de producción ni se consulta
    mixed, used = svc._blend_prior(stub_resources(StubModel(boom=True)), "q", ["a", "b"], {"a": 0.7, "b": 0.3}, enabled=False)
    assert used["metodo"] == "knn" and mixed == {"a": 0.7, "b": 0.3}
    # ningún módulo del backtest ni de las validaciones referencia el modelo de producción ni su carpeta
    forbidden = ("ProductionPrior", "prior_model", "outputs/models", "production")
    guarded = [
        "evaluation.py", "graph.py", "backtest_cases.py", "fastcal.py", "ablation.py",
        "prior_v2/bigdev.py", "prior_v2/bigdev_eval.py", "prior_v2/confirm.py", "prior_v2/confirm2.py",
        "prior_v2/task1.py", "prior_v2/task2.py", "prior_v2/task3.py", "prior_v2/knn.py", "prior_v2/data.py",
    ]
    for name in guarded:
        source = (ROOT / name).read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
        for word in forbidden:
            assert word not in code.replace('"""', ""), f"{name} no puede referirse al modelo de producción ({word})"
