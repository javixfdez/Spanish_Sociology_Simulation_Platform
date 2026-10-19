"""Paquete para entrenar el modelo B (corte < 2026-04-01) en Kaggle. Solo CPU: no usa la GPU.

    python -m mavericks.ft.kaggle_pack

Crea outputs/ft/kaggle/dataset/ (examples_b.parquet, core.py, train.py, train_kaggle.py, manifest.json) y el cuaderno outputs/ft/kaggle/kaggle_train_b.ipynb.
"""

import hashlib
import json
import shutil
from pathlib import Path

from ..prior_v2.data import assert_no_reserved
from . import train

OUT = Path("outputs/ft/kaggle")
COLUMNS = ["id", "evidence_id", "study", "date", "scope", "dim", "source", "year", "language", "question", "options", "props", "profile", "n", "weight_source", "in_b"]


def cell(source):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": source.strip().splitlines(keepends=True)}


NOTEBOOK = [
    """
# Modelo B (corte < 2026-04-01): QLoRA de Qwen3-4B-Instruct-2507, misma configuración que a1, cómputo fp16 (T4)
import subprocess, sys
r = subprocess.run("pip install -q transformers==5.20.1 peft==0.21.2 bitsandbytes==0.50.2 accelerate==1.15.0 pyarrow", shell=True)
if r.returncode:  # si las versiones exactas no encajan con el torch de Kaggle, se usan las más recientes y se anota
    subprocess.run("pip install -q -U transformers peft bitsandbytes accelerate pyarrow", shell=True, check=True)
    print("AVISO: no se pudieron fijar las versiones de a1; se instalaron las más recientes")
import torch, transformers, peft, bitsandbytes
print(torch.__version__, transformers.__version__, peft.__version__, bitsandbytes.__version__, "GPUs:", torch.cuda.device_count())
""",
    """
# Localiza el dataset privado y copia los scripts a /kaggle/working
import glob, os, shutil, json
found = glob.glob("/kaggle/input/**/examples_b.parquet", recursive=True)
assert found, "No encuentro el dataset: añádelo con Add Input (ver instrucciones)"
DATA = os.path.dirname(found[0])
for f in ("core.py", "train.py", "train_kaggle.py"):
    shutil.copy(os.path.join(DATA, f), "/kaggle/working/" + f)
print(DATA, json.load(open(os.path.join(DATA, "manifest.json"))))
""",
    """
# Descarga la MISMA revisión del modelo base que a1
from huggingface_hub import snapshot_download
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
snapshot_download("Qwen/Qwen3-4B-Instruct-2507", revision=REVISION)
""",
    """
# Entrenamiento: 2 procesos (una T4 cada uno) si hay 2 GPUs; si no, uno. Sale el progreso en vivo y se escribe /kaggle/working/status.log
import subprocess, torch
n = max(1, min(2, torch.cuda.device_count()))
cmd = f"cd /kaggle/working && torchrun --standalone --nproc_per_node={n} train_kaggle.py --cut B --name b1 --data {DATA} --out /kaggle/working --target-hours 9.5"
print(cmd)
proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    print(line, end="")
proc.wait()
assert proc.returncode == 0, f"el entrenamiento terminó con código {proc.returncode}"
""",
    """
# Resultado: última parte del registro y el adaptador final comprimido (descárgalo desde la pestaña Output)
import shutil
print(open("/kaggle/working/status.log", encoding="utf-8").read()[-1500:])
shutil.make_archive("/kaggle/working/adapter_b_final", "zip", "/kaggle/working/runs/b1/adapter_final")
""",
]


def write_notebook():
    nb = {
        "cells": [cell(c) for c in NOTEBOOK],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (OUT / "kaggle_train_b.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")


def main():
    frame = train.load_frame("B")  # asserts date < 2026-04-01
    frame = frame[COLUMNS].reset_index(drop=True)
    assert (frame.date < "2026-04-01").all()
    assert_no_reserved(sorted(frame.evidence_id.unique()), "ejemplos del paquete de Kaggle")
    assert not frame.study.astype(str).str.contains("ESS11", case=False).any(), "ESS11 excluido"
    folder = OUT / "dataset"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(folder / "examples_b.parquet", index=False)
    shutil.copy(Path("src/mavericks/ft/core.py"), folder / "core.py")
    # train.py importa core de forma relativa; en Kaggle los tres ficheros están juntos, sin paquete
    code = Path("src/mavericks/ft/train.py").read_text(encoding="utf-8").replace("from . import core", "import core")
    (folder / "train.py").write_text(code, encoding="utf-8")
    shutil.copy(Path("src/mavericks/ft/train_kaggle.py"), folder / "train_kaggle.py")
    manifest = {
        "examples": len(frame),
        "studies": int(frame.study.nunique()),
        "max_date": str(frame.date.max()),
        "national": int((frame.scope == "nacional").sum()),
        "sha256_examples": hashlib.sha256((folder / "examples_b.parquet").read_bytes()).hexdigest(),
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    write_notebook()
    size = sum(f.stat().st_size for f in folder.iterdir()) / 2**20
    print(json.dumps(manifest, indent=1), f"\ntamaño total del dataset: {size:.1f} MB")


if __name__ == "__main__":
    main()
