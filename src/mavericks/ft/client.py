"""Cliente (solo biblioteca estándar) del servidor del modelo ajustado (``mavericks.ft.server``)."""

import json
import os
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8765"
TIMEOUT = 30.0


class FtError(RuntimeError):
    pass


class FtClient:
    def __init__(self, url=None, timeout=TIMEOUT):
        self.url = (url or os.environ.get("MAVERICKS_FT_URL") or DEFAULT_URL).rstrip("/")
        self.timeout = timeout

    def _call(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.load(response)
        except Exception as exc:  # red, tiempo agotado o respuesta mal formada
            raise FtError(f"el servidor del modelo ajustado no respondió ({type(exc).__name__}: {exc})") from exc

    def healthy(self):
        try:
            return bool(self._call("/health").get("ok"))
        except FtError:
            return False

    def predict(self, question, options, profile=None, year=None):
        out = self._call("/predict", {"question": question, "options": list(options), "profile": profile, "year": year})
        return _checked(out["probs"], len(options)), out["seconds"]

    def predict_batch(self, items):
        out = self._call("/predict_batch", {"items": items})
        return [_checked(p, len(it["options"])) for p, it in zip(out["probs"], items, strict=True)], out["seconds"]


def _checked(probs, n):
    if len(probs) != n or any(p != p or p < 0 for p in probs) or abs(sum(probs) - 1) > 1e-3:
        raise FtError("respuesta no válida del servidor del modelo ajustado")
    return probs
