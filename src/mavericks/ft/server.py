r"""Servidor local del modelo ajustado (QLoRA de Qwen3-4B): carga base nf4 + adaptador UNA vez y devuelve repartos leyendo el primer token. Se ejecuta en .venv_ft.

    set PYTHONPATH=src
    .venv_ft\Scripts\python -m mavericks.ft.server --adapter outputs/ft/runs/b1/adapter_final --port 8765

    GET  /health
    POST /predict        {"question", "options", "profile"?, "source"?, "year"?, "language"?}      -> {"probs": [...], "seconds": s}
    POST /predict_batch  {"items": [{...}, ...]}                                                  -> {"probs": [[...], ...], "seconds": s}

El prompt y la lectura de letras son los de entrenamiento (``core.prompt_text``, ``train.make_batch`` y ``train.letter_log_probs``): orden de opciones
original, sin barajar y sin promediar permutaciones. Las opciones llegan ya con su etiqueta de usuario (``assist.display_label``).
"""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd
import torch

from . import train

DEFAULT_PROFILE = "Población: adultos residentes en España"
MAX_OPTIONS = 7
BATCH = 16


class Model:
    def __init__(self, adapter):
        self.tok, self.model = train.load_model(adapter=adapter, trainable=False)
        self.model.eval()
        self.ids = train.letter_ids(self.tok)
        self.lock = threading.Lock()  # una sola GPU: las peticiones se atienden de una en una
        self.year = time.strftime("%Y")

    def frame(self, items):
        rows = []
        for i, it in enumerate(items):
            options = [str(o) for o in it["options"]]
            if not 2 <= len(options) <= MAX_OPTIONS:
                raise ValueError(f"de 2 a {MAX_OPTIONS} opciones")
            rows.append(
                {"id": str(i), "question": " ".join(str(it["question"]).split()), "options": options, "props": [1 / len(options)] * len(options),
                 "profile": it.get("profile") or DEFAULT_PROFILE, "source": it.get("source") or "CIS", "year": str(it.get("year") or self.year),
                 "language": it.get("language") or "es", "n": 1000, "weight_source": 1.0}
            )
        return pd.DataFrame(rows)

    @torch.no_grad()
    def predict(self, items):
        frame = self.frame(items)
        out = []
        with self.lock:
            for start in range(0, len(frame), BATCH):
                enc, props, _ = train.make_batch(frame.iloc[start : start + BATCH], 0, self.tok, shuffle=False)
                log_q, _ = train.letter_log_probs(self.model, enc, self.ids, [len(p) for p in props])
                out += [[float(x) for x in log_q[i].exp()[: len(p)].float().cpu()] for i, p in enumerate(props)]
        return out


def make_handler(model, adapter):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, payload):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send(200, {"ok": True, "adapter": str(adapter), "vram_gb": torch.cuda.max_memory_allocated() / 2**30})
            else:
                self._send(404, {"error": "no existe"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                started = time.time()
                if self.path == "/predict":
                    probs = model.predict([body])[0]
                elif self.path == "/predict_batch":
                    probs = model.predict(body["items"])
                else:
                    return self._send(404, {"error": "no existe"})
                self._send(200, {"probs": probs, "seconds": time.time() - started})
            except (ValueError, KeyError, TypeError) as exc:
                self._send(400, {"error": str(exc)})
            except Exception as exc:  # no tumbar el servidor por una petición
                self._send(500, {"error": f"{type(exc).__name__}: {exc}"})

        def log_message(self, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="outputs/ft/runs/b1/adapter_final")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    model = Model(args.adapter)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(model, args.adapter))
    print(f"servidor listo en http://127.0.0.1:{args.port} ({args.adapter})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
