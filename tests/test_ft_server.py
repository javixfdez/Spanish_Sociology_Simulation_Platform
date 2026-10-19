import sys
import types

import pytest


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def exp(self):
        return self

    def __getitem__(self, key):
        return FakeTensor(self.values[key]) if isinstance(key, slice) else self

    def float(self):
        return self

    def cpu(self):
        return self

    def __iter__(self):
        return iter(self.values)


@pytest.fixture
def server(monkeypatch):
    """El servidor real con torch y train simulados (el entorno principal no los tiene)."""
    calls = {"empty_cache": 0, "batches": []}
    torch = types.ModuleType("torch")
    torch.no_grad = lambda f=None: (f if f else (lambda g: g))
    torch.cuda = types.SimpleNamespace(empty_cache=lambda: calls.__setitem__("empty_cache", calls["empty_cache"] + 1), max_memory_allocated=lambda: 0)
    train = types.ModuleType("mavericks.ft.train")

    def make_batch(frame, epoch, tok, shuffle=True):
        calls["batches"].append(len(frame))
        return None, [list(p) for p in frame.props], None

    def letter_log_probs(model, enc, ids, n_options):
        if calls.get("fail"):
            raise RuntimeError("CUDA out of memory")
        return [FakeTensor([0.5] * n) for n in n_options], None

    train.load_model = lambda adapter=None, trainable=True: (object(), types.SimpleNamespace(eval=lambda: None))
    train.letter_ids = lambda tok: list(range(7))
    train.make_batch, train.letter_log_probs = make_batch, letter_log_probs
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "mavericks.ft.train", train)
    monkeypatch.delitem(sys.modules, "mavericks.ft.server", raising=False)
    import mavericks.ft.server as module

    yield module, calls
    monkeypatch.delitem(sys.modules, "mavericks.ft.server", raising=False)


def test_cuda_cache_released_after_every_batch_even_on_error(server):
    module, calls = server
    model = module.Model("adaptador")
    items = [{"question": "¿Pregunta de prueba?", "options": ["Sí", "No"]} for _ in range(module.BATCH + 3)]
    out = model.predict(items)
    assert len(out) == len(items) and calls["empty_cache"] == 2 and calls["batches"] == [module.BATCH, 3]
    calls["fail"] = True
    before = calls["empty_cache"]
    with pytest.raises(RuntimeError):
        model.predict(items[:2])
    assert calls["empty_cache"] == before + 1


def test_rejects_more_than_seven_options(server):
    module, _ = server
    with pytest.raises(ValueError):
        module.Model("a").predict([{"question": "¿Pregunta de prueba?", "options": [str(i) for i in range(8)]}])
