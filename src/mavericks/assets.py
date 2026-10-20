"""Paquete de ejecución de la app (archivos fuera de git) y su descarga desde Hugging Face.

    mavericks pack-assets                 # crea dist/mavericks_assets_v1.zip + manifest.json (sin el adaptador)
    mavericks fetch-assets [--from zip]   # descarga (o usa un zip local), descomprime y verifica los sha256
    mavericks fetch-adapter               # descarga el adaptador B y verifica su sha256

Solo se empaquetan agregados y datos procesados de una lista cerrada (``ASSETS``); nunca ``maverick_data/`` ni ficheros originales del Eurobarómetro o la ESS. Los estudios
reservados (publicados desde 2026-04-01) se quitan de los microdatos y de la tabla nacional del CIS.
"""

import hashlib
import html
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from pathlib import Path

ASSETS_REPO = "JaviPica/mavericks-assets"
ASSETS_FILE = "mavericks_assets_v1.zip"
ADAPTER_REPO = "JaviPica/mavericks-qwen3-4b-cis"
ADAPTER_DIR = "outputs/ft/runs/b1/adapter_final"
ADAPTER_SHA256 = "f6a4dd9453f3233933b14a12af70d64ef0db197c37dedf24dbab0e832f339c6b"
ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors")
FORBIDDEN_SUFFIXES = {".sav", ".dta", ".por", ".zip", ".sas7bdat", ".rar", ".7z"}  # formatos de los ficheros originales
ALL, FAST_AND_LLM, ADJUSTED = ("rapido", "llm", "ajustado"), ("rapido", "llm"), ("ajustado",)

# (ruta, modos que la necesitan, para qué sirve, fuente). La lista es cerrada: lo que no está aquí no entra en el paquete.
ASSETS = [
    ("data/processed/synthetic_population.parquet", ALL, "Población sintética de 50.000 adultos (segmentos y cuotas de los subgrupos)", "Generada con etl-population: Censo 2021, Cifras de Población 1/1/2026, EPA 2T26 y ECV 2025 (INE)"),
    ("data/processed/population_metadata.json", ALL, "Metadatos y procedencia de la población sintética", "etl-population"),
    ("data/processed/population_targets.json", ALL, "Objetivos de la ECP para la página de población", "population-targets (INE)"),
    ("data/processed/population_validation.json", ALL, "Validación de márgenes de la población sintética", "etl-population (INE)"),
    ("data/processed/cis_respondents.parquet", ALL, "Entrevistados del CIS (variables sociodemográficas y peso) para las diferencias por grupo", "CIS, microdatos de estudios NO reservados, procesados con etl-cis"),
    ("data/processed/cis_responses.parquet", ALL, "Respuestas del CIS por entrevistado y pregunta (modo directo y subgrupos)", "CIS, microdatos de estudios NO reservados, procesados con etl-cis"),
    ("data/processed_v2/evidence_v2.sqlite", ALL, "Almacén de evidencia (preguntas, opciones y distribuciones del CIS, Eurobarómetro y ESS de España)", "ingest-surveys sobre CIS y etl-external (agregados), sin reservados"),
    ("data/processed_v2/cis_national.json", ALL, "Distribución nacional de cada pregunta del CIS (prior por analogía)", "build-national (CIS no reservado)"),
    ("outputs/models/prior_mlp.pkl", ALL, "Modelo aprendido del prior (MLP sobre embeddings e5)", "train-prior (CIS no reservado)"),
    ("outputs/models/prior_mlp.json", ALL, "Metadatos del modelo del prior (corte, preguntas, hash del almacén)", "train-prior"),
    ("outputs/models/prior_intervals.json", ALL, "Cuantiles de los rangos del prior por número de opciones", "calibrate-prior (dev_grande y confirmaciones)"),
    ("outputs/models/prior_validation.json", ALL, "Tabla «Validación con datos no vistos» y error típico del prior", "calibrate-prior"),
    ("outputs/fast_validation.json", ALL, "Validación del modo rápido (153 casos) y rangos por modo", "fast-validation"),
    ("outputs/baseline_v1/calibration.json", ALL, "Calibración de la confianza por modo (errores históricos del backtest)", "backtest sobre los casos oficiales y de desarrollo"),
    ("outputs/baseline_v1/backtest.json", ALL, "Resultados del backtest con LLM (página Validación)", "backtest (agregados por caso)"),
    ("outputs/case_floors.json", ALL, "Suelo de ruido por caso (página Validación)", "noise-floor"),
    ("outputs/final_eval/report.json", ALL, "Resumen de la evaluación final oficial (página Validación)", "final_eval (una sola ejecución)"),
    ("outputs/models/ft_intervals.json", ADJUSTED, "Rangos del modo ajustado (calibración Mondrian con los residuos de a1 en dev_grande)", "python -m mavericks.ft.artifacts"),
    ("outputs/models/ft_validation.json", ADJUSTED, "Error típico del modo ajustado en dev_grande", "python -m mavericks.ft.artifacts"),
]


def sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def reserved_studies(cis_dir="data/processed"):
    import pandas as pd

    studies = pd.read_parquet(Path(cis_dir) / "cis_studies.parquet", columns=["estudio_id", "reservado_examen"])
    return {str(s) for s in studies.loc[studies.reservado_examen.astype(bool), "estudio_id"]}


def filter_parquet(src, dst, reserved):
    """Copia el parquet sin las filas de estudios reservados (por lotes: cis_responses tiene 31 millones de filas)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    source = pq.ParquetFile(src)
    writer = pq.ParquetWriter(dst, source.schema_arrow, compression="zstd")
    dropped = kept = 0
    for batch in source.iter_batches(batch_size=1_000_000):
        table = pa.Table.from_batches([batch])
        mask = pc.invert(pc.is_in(pc.cast(table["estudio_id"], pa.string()), value_set=pa.array(sorted(reserved), pa.string())))
        filtered = table.filter(mask)
        dropped += table.num_rows - filtered.num_rows
        kept += filtered.num_rows
        writer.write_table(filtered)
    writer.close()
    return kept, dropped


def filter_national(src, dst, reserved_ids):
    table = json.loads(Path(src).read_text(encoding="utf-8"))
    kept = {k: v for k, v in table.items() if k not in reserved_ids}
    Path(dst).write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    return len(kept), len(table) - len(kept)


def check_allowed(path):
    p = Path(path)
    assert not p.is_absolute() and ".." not in p.parts, f"ruta no relativa: {path}"
    assert p.parts[0] != "maverick_data", f"nunca se empaqueta maverick_data: {path}"
    assert p.suffix.lower() not in FORBIDDEN_SUFFIXES, f"formato de fichero original no permitido: {path}"
    assert any(path == a[0] for a in ASSETS), f"fuera de la lista cerrada: {path}"


def pack(out="dist/mavericks_assets_v1.zip", root="."):
    from .prior_v2.data import reserved_ids

    root = Path(root)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    reserved = reserved_studies(root / "data/processed")
    reserved_evidence = reserved_ids(root / "data/processed")
    from .evidence import EvidenceRepository

    stored = {e.evidence_id for e in EvidenceRepository(root / "data/processed_v2/evidence_v2.sqlite").all()}
    assert not stored & reserved_evidence, "el almacén de evidencia contiene estudios reservados"
    files, notes = [], {}
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, modes, purpose, source in ASSETS:
            check_allowed(path)
            src = root / path
            assert src.exists(), f"falta {path}: genera los archivos con los comandos del README antes de empaquetar"
            use = src
            if path in ("data/processed/cis_respondents.parquet", "data/processed/cis_responses.parquet"):
                use = Path(tmp) / Path(path).name
                kept, dropped = filter_parquet(src, use, reserved)
                notes[path] = f"{dropped} filas de {len(reserved)} estudios reservados eliminadas ({kept} conservadas)"
            elif path == "data/processed_v2/cis_national.json":
                use = Path(tmp) / "cis_national.json"
                kept, dropped = filter_national(src, use, reserved_evidence)
                notes[path] = f"{dropped} preguntas de estudios reservados eliminadas ({kept} conservadas)"
            archive.write(use, path)
            files.append({"path": path, "size": use.stat().st_size, "sha256": sha256(use), "modes": list(modes), "purpose": purpose, "source": source})
        assert not any(f["path"].startswith("maverick_data") or Path(f["path"]).suffix.lower() in FORBIDDEN_SUFFIXES for f in files)
        manifest = {
            "version": 1, "created": time.strftime("%Y-%m-%d %H:%M"), "files": files, "filters": notes,
            "excludes": ["maverick_data/ (microdatos y ficheros originales)", "ficheros originales del Eurobarómetro (GESIS) y de la ESS", "estudios reservados (publicados desde 2026-04-01)", "el adaptador del modelo (repo aparte)"],
        }
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=1))
    (out.parent / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"zip": str(out), "zip_bytes": out.stat().st_size, "zip_sha256": sha256(out), "files": len(files), "filters": notes}


def _hub():
    try:
        import huggingface_hub
    except ImportError as exc:
        raise SystemExit("Falta huggingface_hub: instala el extra con `uv sync --extra hub`.") from exc
    try:  # redes con inspección TLS (antivirus, proxy): certificados del sistema
        import truststore

        truststore.inject_into_ssl()
    except ImportError:
        pass
    return huggingface_hub


URL_VARIABLE = "MAVERICKS_ASSETS_URL"
WAYS = (
    "Hay tres vías para conseguir los datos:\n"
    "  1) un enlace de descarga (p. ej. de Google Drive, «cualquiera con el enlace»): `mavericks fetch-assets --from-url <enlace>` o `MAVERICKS_ASSETS_URL=<enlace>` en el archivo .env;\n"
    "  2) Hugging Face (repositorio privado, hay que pedir acceso a JaviPica): `uv run hf auth login` y después `mavericks fetch-assets`;\n"
    "  3) un zip local: `mavericks fetch-assets --from ruta/mavericks_assets_v1.zip`."
)


def env_value(name, env_file=".env"):
    """Valor de una variable del entorno o, si no está, del archivo .env (formato CLAVE=valor)."""
    if os.environ.get(name):
        return os.environ[name].strip()
    path = Path(env_file)
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            key, _, value = line.partition("=")
            if key.strip() == name and value.strip():
                return value.strip().strip("\"'")
    return None


def drive_file_id(url):
    """Identificador de un archivo de Google Drive a partir de cualquiera de sus enlaces habituales, o None si no es de Drive."""
    if not re.search(r"(drive|docs)\.google\.com|drive\.usercontent\.google\.com", url):
        return None
    match = re.search(r"/d/([\w-]{15,})", url) or re.search(r"[?&]id=([\w-]{15,})", url)
    return match.group(1) if match else None


def _stream_to_file(response, target, label):
    total = int(response.headers.get("Content-Length") or 0)
    done, last = 0, 0.0
    with open(target, "wb") as f:
        for block in response.iter_content(1 << 20):
            f.write(block)
            done += len(block)
            if time.time() - last > 2 and total:
                print(f"  {label}: {done / 2**20:.0f} de {total / 2**20:.0f} MB", file=sys.stderr, flush=True)
                last = time.time()


def download_url(url, target):
    """Descarga ``url`` a ``target`` con requests. Acepta enlaces de Google Drive «cualquiera con el enlace», incluida la página de aviso de virus de los archivos grandes."""
    import requests

    session = requests.Session()
    file_id = drive_file_id(url)
    try:
        if file_id:
            response = session.get("https://drive.usercontent.google.com/download", params={"id": file_id, "export": "download", "confirm": "t"}, stream=True, timeout=60)
            if "text/html" in response.headers.get("Content-Type", ""):  # aviso de virus / confirmación: el formulario lleva los parámetros que hay que reenviar
                page = response.text
                form = re.search(r'<form[^>]+id="download-form"[^>]*action="([^"]+)"', page) or re.search(r'<form[^>]+action="([^"]+)"[^>]*id="download-form"', page)
                if not form:
                    raise RuntimeError("Google Drive no devolvió el archivo: comprueba que el enlace es de «cualquiera con el enlace» y que no ha superado su cuota de descargas")
                params = dict(re.findall(r'<input[^>]+name="([^"]+)"[^>]+value="([^"]*)"', page))
                response = session.get(html.unescape(form.group(1)), params=params, stream=True, timeout=60)
        else:
            response = session.get(url, stream=True, timeout=60)
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise RuntimeError("el enlace devolvió una página web y no un zip (¿falta el permiso «cualquiera con el enlace»?)")
        _stream_to_file(response, target, "descargando")
    except requests.RequestException as exc:
        raise RuntimeError(f"no se pudo descargar ({type(exc).__name__}: {exc})") from exc
    if not zipfile.is_zipfile(target):
        raise RuntimeError("lo descargado no es un zip válido")
    return target


def get_zip(local_zip=None, url=None, repo=ASSETS_REPO, filename=ASSETS_FILE, tmp_dir=None):
    """Ruta del zip de datos. Orden: zip local; enlace (argumento o MAVERICKS_ASSETS_URL en el entorno o en .env); Hugging Face (privado). Si nada funciona, error con las tres vías."""
    if local_zip:
        if not Path(local_zip).exists():
            raise SystemExit(f"No existe el zip local {local_zip}.\n{WAYS}")
        return Path(local_zip)
    errors = []
    link = url or env_value(URL_VARIABLE)
    if link:
        try:
            target = Path(tmp_dir or tempfile.mkdtemp(prefix="mavericks_")) / ASSETS_FILE
            return download_url(link, target)
        except Exception as exc:
            errors.append(f"enlace ({link[:60]}…): {exc}")
    try:
        return Path(_hub().hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset"))
    except SystemExit:
        raise
    except Exception as exc:
        errors.append(f"Hugging Face ({repo}): {type(exc).__name__}: {str(exc)[:160]} (el repositorio es privado: hay que pedir acceso y hacer `hf auth login`)")
    raise SystemExit("No se pudieron descargar los datos.\n" + "\n".join(f"- {e}" for e in errors) + "\n" + WAYS)


def fetch_assets(local_zip=None, repo=ASSETS_REPO, filename=ASSETS_FILE, dest=".", url=None):
    """Consigue el zip (ver ``get_zip``), lo descomprime en ``dest`` y verifica los sha256 del manifest."""
    zip_path = get_zip(local_zip, url, repo, filename)
    dest = Path(dest)
    with zipfile.ZipFile(zip_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        for entry in manifest["files"]:
            check_allowed(entry["path"])
            target = dest / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry["path"]) as src, open(target, "wb") as dst:
                while block := src.read(1 << 20):
                    dst.write(block)
    bad = [e["path"] for e in manifest["files"] if sha256(dest / e["path"]) != e["sha256"]]
    if bad:
        raise SystemExit(f"sha256 distinto al del manifest: {bad}")
    return {"zip": str(zip_path), "files": len(manifest["files"]), "verified": True, "dest": str(dest)}


def fetch_adapter(repo=ADAPTER_REPO, dest=ADAPTER_DIR):
    """Descarga el adaptador B de Hugging Face y comprueba su sha256."""
    _hub().snapshot_download(repo_id=repo, repo_type="model", local_dir=dest, allow_patterns=list(ADAPTER_FILES))
    actual = sha256(Path(dest) / "adapter_model.safetensors")
    if actual != ADAPTER_SHA256:
        raise SystemExit(f"El adaptador descargado no es el esperado (sha256 {actual}, esperado {ADAPTER_SHA256}).")
    return {"dest": dest, "sha256": actual, "verified": True}
