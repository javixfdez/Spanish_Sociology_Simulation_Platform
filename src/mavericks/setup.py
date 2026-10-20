"""``mavericks setup``: deja la app lista con un solo comando.

Pasos: Python y uv, ``uv sync --extra embed --extra hub``, datos (``fetch-assets``), adaptador del modo ajustado (solo con GPU NVIDIA de 8 GB o más), modelo e5 y una pregunta de prueba en modo rápido.
Cada error se explica en castellano y con la solución. Es idempotente: lo que ya está descargado y verificado no se vuelve a bajar.
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import assets

MIN_PYTHON = (3, 11)
MIN_GPU_MIB = 7900  # una tarjeta de 8 GB anuncia ≈ 8.188 MiB
E5 = "intfloat/multilingual-e5-base"
TEST_QUESTION = ("¿Le preocupa que la inteligencia artificial le quite su empleo?", ["Mucho", "Bastante", "Poco", "Nada"])
TOTAL = 7


class SetupError(Exception):
    def __init__(self, problem, fix):
        super().__init__(problem)
        self.problem, self.fix = problem, fix


def say(n, text):
    print(f"[{n}/{TOTAL}] {text}", flush=True)


def uv_command():
    """Cómo invocar uv: en el PATH, la variable UV que define `uv run`, o `python -m uv` (con el Python del PATH o el actual). None si no se encuentra."""
    if os.environ.get("UV") and Path(os.environ["UV"]).exists():
        return [os.environ["UV"]]
    found = shutil.which("uv")
    if found:
        return [found]
    for python in (shutil.which("python"), sys.executable):
        if python:
            probe = subprocess.run([python, "-m", "uv", "--version"], capture_output=True, text=True)
            if probe.returncode == 0:
                return [python, "-m", "uv"]
    return None


def gpu_memory_mib():
    """Memoria total de la primera GPU NVIDIA en MiB, o None si no hay GPU NVIDIA (o no se puede consultar)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "--query-gpu=memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def data_ready():
    missing = [path for path, *_ in assets.ASSETS if not Path(path).exists()]
    return not missing, missing


def adapter_ready():
    path = Path(assets.ADAPTER_DIR) / "adapter_model.safetensors"
    return path.exists() and assets.sha256(path) == assets.ADAPTER_SHA256


def e5_ready():
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(E5, local_files_only=True)
        return True
    except Exception:
        return False


def run(skip_sync=False, no_adapter=False, local_zip=None, url=None):
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    started = time.time()
    notes = []
    try:
        say(1, f"Python {sys.version_info.major}.{sys.version_info.minor}")
        if sys.version_info < MIN_PYTHON:
            raise SetupError(f"Hace falta Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} o superior (tienes {sys.version.split()[0]}).", "Instala una versión reciente desde https://www.python.org/downloads/ y vuelve a ejecutar el comando.")

        say(2, "uv")
        uv = uv_command()
        if uv is None and skip_sync:
            print("    (no hace falta: la sincronización ya la hizo scripts/setup.ps1)")
        elif uv is None:
            raise SetupError("No se encuentra uv.", "Instálalo con `pip install uv` (o `python -m pip install --user uv`) y vuelve a ejecutar el comando.")

        say(3, "Dependencias (uv sync --extra embed --extra hub)")
        if skip_sync:
            print("    (omitido: ya sincronizado)")
        else:
            env = {**os.environ, "UV_SYSTEM_CERTS": os.environ.get("UV_SYSTEM_CERTS", "1")}
            sync = subprocess.run([*uv, "sync", "--extra", "embed", "--extra", "hub"], env=env, capture_output=True, text=True)
            if sync.returncode != 0:
                tail = (sync.stderr or sync.stdout).strip().splitlines()[-3:]
                raise SetupError("`uv sync` ha fallado: " + " | ".join(tail), "Comprueba tu conexión; si ves errores de certificados TLS, ejecuta con `UV_SYSTEM_CERTS=1` (en PowerShell: `$env:UV_SYSTEM_CERTS = \"1\"`).")
        for module in ("sentence_transformers", "huggingface_hub", "requests", "streamlit"):
            try:
                __import__(module)
            except ImportError as exc:
                raise SetupError(f"Falta la dependencia {module}.", "Ejecuta `uv sync --extra embed --extra hub` y vuelve a lanzar `uv run mavericks setup`.") from exc

        say(4, "Datos de la app (fetch-assets)")
        ready, missing = data_ready()
        if ready:
            print("    ya están todos los archivos")
        else:
            try:
                assets.fetch_assets(local_zip, url=url)
            except SystemExit as exc:
                raise SetupError("No se pudieron descargar los datos.", str(exc)) from exc
            print("    descargados y verificados (sha256)")

        say(5, "Adaptador del modo ajustado")
        memory = gpu_memory_mib()
        if no_adapter:
            notes.append("Modo ajustado: omitido (--no-adapter).")
            print("    omitido (--no-adapter)")
        elif memory is None:
            notes.append("Modo ajustado NO disponible: no hay GPU NVIDIA. El modo rápido funciona igual.")
            print("    sin GPU NVIDIA: el modo ajustado no estará disponible (el modo rápido funciona igual)")
        elif memory < MIN_GPU_MIB:
            notes.append(f"Modo ajustado NO disponible: la GPU tiene {memory / 1024:.1f} GB y hacen falta 8 GB.")
            print(f"    GPU de {memory / 1024:.1f} GB (< 8 GB): el modo ajustado no estará disponible")
        elif adapter_ready():
            print("    adaptador ya descargado y verificado")
        else:
            try:
                assets.fetch_adapter()
            except SystemExit as exc:
                raise SetupError("No se pudo descargar el adaptador.", f"{exc}. Repositorio público JaviPica/mavericks-qwen3-4b-cis: comprueba tu conexión y repite `mavericks fetch-adapter`.") from exc
            print("    adaptador descargado y verificado (sha256)")
            notes.append("Para usar el modo ajustado falta además el entorno .venv_ft y la base Qwen (docs/finetuning.md, «Cómo usarlo») y arrancar scripts/run_server.ps1.")

        say(6, "Modelo e5 (embeddings del prior)")
        if e5_ready():
            print("    ya está en la caché")
        else:
            try:
                from .embeddings import SentenceTransformerEmbedder

                SentenceTransformerEmbedder(E5, device="cpu")
            except Exception as exc:
                raise SetupError(f"No se pudo descargar {E5}: {type(exc).__name__}.", "Hace falta internet esta vez (≈ 1 GB). Si hay un proxy o antivirus con inspección TLS, `UV_SYSTEM_CERTS=1` suele bastar; repite el comando.") from exc
            print("    descargado")

        say(7, "Pregunta de prueba en modo rápido")
        try:
            from . import service

            resources = service.load_resources()
            service.get_embedder()
            t = time.time()
            out = service.predict(TEST_QUESTION[0], TEST_QUESTION[1], preset="rapido_sin_llm", resources=resources)
            top = max(out["national"]["distribution"], key=out["national"]["distribution"].get)
            print(f"    «{TEST_QUESTION[0]}» → «{top}» {100 * out['national']['distribution'][top]:.0f} % (modo {out['mode_label']}, {time.time() - t:.1f} s)")
        except Exception as exc:
            raise SetupError(f"La pregunta de prueba falló ({type(exc).__name__}: {exc}).", "Revisa los pasos anteriores; si el error menciona un archivo, repite `mavericks fetch-assets`.") from exc
    except SetupError as exc:
        print(f"\nERROR: {exc.problem}\nSolución: {exc.fix}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"\nListo ({time.time() - started:.0f} s): ejecuta  uv run streamlit run app.py")
    for note in notes:
        print(f"  · {note}")
    return {"ok": True, "notas": notes}
