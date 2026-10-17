"""ETL de microdatos del CIS (zips con .sav) al contrato de la Fase 1.

Genera en ``out_dir``: cis_studies, cis_questions, cis_respondents (parquet) y cis_responses
(formato largo, un grupo de filas por estudio) más etl_cis_report.json.
"""

import csv
import html
import io
import json
import logging
import os
import re
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

from .. import categories as cat

logger = logging.getLogger(__name__)

CUTOFF = date(2026, 6, 30)
DEFAULT_UNIVERSE = "Adultos residentes en hogares privados de España"
SKIP_FOLDERS = {"_excluidos"}
# Variables de identificación, geografía, demografía de fondo, pesos y paradatos.
_EXCLUDED = re.compile(
    r"^(ESTUDIO|ENTREV.*|REGISTRO|CUES|TIPO_TEL.*|CCAA|PROV|MUN|TAMUNI|CAPITAL|SEXO|EDAD.*|NACIONALIDAD|"
    r"ADQUINAC|PAISNAC.*|FECHA.*|HORA.*|DURACION|FIJO|PESO.*|IA_.*|V2_.*|V3|SINCERIDAD|P0|ESTUDIOS|INGRESHOG)$"
)
_POSITIVE = {"si", "menciona", "mencionado", "citado", "cita", "marcado", "seleccionado"}
_NEGATIVE = {"no", "no menciona", "no mencionado", "no citado", "no cita", "no marcado", "no seleccionado"}
_MONTHS = {
    m: i + 1
    for i, m in enumerate(
        "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
    )
}
_MONTHS["setiembre"] = 9
MAX_CATEGORIES = 30
MIN_VALID = 30

RESPONSE_SCHEMA = pa.schema(
    [
        ("estudio_id", pa.string()),
        ("respondent_uid", pa.string()),
        ("question_uid", pa.string()),
        ("answer", pa.string()),
        ("selected", pa.bool_()),
    ]
)


def clean(text):
    return re.sub(r"\s+", " ", html.unescape(str(text or ""))).strip()


def _is_np(label):
    return cat.norm(label) in ("n.p.", "np")


def parse_fieldwork(text):
    """Extrae (inicio, fin) de «Fecha de realización: Del 2 al 6 de marzo de 2026». None si no hay."""
    m = re.search(r"(?is)fecha\s+de\s+realizaci.n\s*:?\s*(.{0,160}?\b(?:19|20)\d{2})", text or "")
    if not m:
        return None
    tokens = re.findall(
        r"(\d{1,2})(?:\s+(?:de\s+)?([a-záéíóú]+))?(?:\s+de\s+((?:19|20)\d{2}))?", m.group(1), flags=re.I
    )
    parsed = []
    for day, month, year in tokens:
        month = _MONTHS.get(month.lower()) if month else None
        parsed.append([int(day), month, int(year) if year else None])
    if not parsed:
        return None
    # Los tokens iniciales heredan mes y año del siguiente.
    for i in range(len(parsed) - 2, -1, -1):
        if parsed[i][1] is None:
            parsed[i][1] = parsed[i + 1][1]
        if parsed[i][2] is None:
            parsed[i][2] = parsed[i + 1][2]
    try:
        dates = [date(y, mo, d) for d, mo, y in parsed]
    except (ValueError, TypeError):
        return None
    if len(dates) > 1 and dates[0] > dates[-1]:  # «Del 30 de diciembre al 5 de enero de 2026»
        dates[0] = date(dates[0].year - 1, dates[0].month, dates[0].day)
    return min(dates), max(dates)


def micro_dates(frame):
    """Rango de fechas de campo desde IA_E1_* o FECHAINI/FECHAFIN."""
    try:
        if {"IA_E1_ANYO", "IA_E1_MES", "IA_E1_DIA"} <= set(frame.columns):
            d = frame[["IA_E1_ANYO", "IA_E1_MES", "IA_E1_DIA"]].dropna()
            year = d.IA_E1_ANYO.astype(int).where(d.IA_E1_ANYO >= 100, d.IA_E1_ANYO + 2000)
            ts = pd.to_datetime(
                pd.DataFrame({"year": year, "month": d.IA_E1_MES.astype(int), "day": d.IA_E1_DIA.astype(int)}),
                errors="coerce",
            ).dropna()
        elif {"FECHAINI", "FECHAFIN"} <= set(frame.columns):
            ts = pd.concat(
                [pd.to_datetime(frame[c], dayfirst=True, errors="coerce") for c in ("FECHAINI", "FECHAFIN")]
            ).dropna()
        else:
            return None
        if ts.empty:
            return None
        return ts.min().date(), ts.max().date()
    except (ValueError, TypeError):
        return None


def read_catalog(path):
    rows = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rows[Path(row["archivo"].replace("\\", "/")).name] = row
    return rows


def read_zip(path):
    """Devuelve (frame, meta, texto de la ficha técnica) de un zip del CIS."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        sav = next(n for n in names if n.lower().endswith(".sav"))
        with tempfile.TemporaryDirectory() as tmp:
            z.extract(sav, tmp)
            frame, meta = pyreadstat.read_sav(os.path.join(tmp, sav))
        ft = next((n for n in names if n.lower().startswith("ft") and n.lower().endswith(".pdf")), None)
        text = ""
        if ft:
            try:
                from pypdf import PdfReader

                text = "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(z.read(ft))).pages)
            except Exception as exc:  # PDF ilegible: se usan las fechas del microdato
                logger.warning("%s: ficha técnica ilegible (%s)", path, exc)
    return frame, meta, text


def _find(frame, meta, names, label_prefix):
    for name in names:
        if name in frame.columns:
            return name
    labels = meta.column_names_to_labels
    for name in frame.columns:
        if clean(labels.get(name)).lower().startswith(label_prefix):
            return name
    return None


def _labelled(series, labels, mapper):
    """Aplica un mapeo etiqueta->categoría común sobre códigos numéricos."""
    cache = {code: mapper(label) for code, label in labels.items()}
    return series.map(cache)


def build_respondents(frame, meta, estudio_id, warnings):
    V = meta.variable_value_labels
    out = pd.DataFrame(index=frame.index)
    out["estudio_id"] = estudio_id
    if "REGISTRO" in frame and frame.REGISTRO.is_unique:
        out["respondent_uid"] = frame.REGISTRO.astype(int).astype(str)
    else:
        warnings.append("REGISTRO ausente o duplicado: se usa el índice de fila")
        out["respondent_uid"] = np.arange(len(frame)).astype(str)
    if "PESO" in frame:
        out["peso"] = pd.to_numeric(frame.PESO, errors="coerce")
        peso_source = "PESO"
    else:
        out["peso"] = 1.0
        peso_source = "uniforme"
        warnings.append("Sin variable PESO: peso uniforme")
    out["peso_ccaa"] = pd.to_numeric(frame.PESOCCAA, errors="coerce") if "PESOCCAA" in frame else np.nan
    if (out.peso.isna() | (out.peso <= 0)).any():
        warnings.append("PESO no positivo en algunas filas: se sustituye por 1")
        out["peso"] = out.peso.where(out.peso > 0, 1.0)
    sex = _find(frame, meta, ["SEXO"], "sexo de la persona")
    out["sexo"] = _labelled(frame[sex], V.get(sex, {}), cat.map_sexo) if sex else None
    age = _find(frame, meta, ["EDAD", "EDADEXACTA"], "edad de la persona")
    if age:
        edad = pd.to_numeric(frame[age], errors="coerce")
        edad = edad.where((edad >= 18) & (edad < 120))
        out["edad"] = edad
    else:
        out["edad"] = np.nan
        warnings.append("Sin variable de edad")
    out["grupo_edad"] = cat.age_group(out.edad)
    edu = _find(frame, meta, ["ESTUDIOS"], "estudios de la persona")
    out["nivel_estudios"] = _labelled(frame[edu], V.get(edu, {}), cat.map_estudios) if edu else None
    out["ccaa"] = _labelled(frame.CCAA, V.get("CCAA", {}), cat.map_ccaa) if "CCAA" in frame else None
    out["tamuni"] = _labelled(frame.TAMUNI, V.get("TAMUNI", {}), cat.map_tamuni) if "TAMUNI" in frame else None
    if "INGRESHOG" in frame:
        out["ingresos_hogar_tramo"] = _labelled(frame.INGRESHOG, V.get("INGRESHOG", {}), cat.map_ingresos)
    else:
        out["ingresos_hogar_tramo"] = None
    for column in ["sexo", "nivel_estudios", "ccaa", "tamuni", "ingresos_hogar_tramo"]:
        out[column] = out[column].astype("object")
        source_present = {"tamuni": "TAMUNI", "ingresos_hogar_tramo": "INGRESHOG"}.get(column)
        if out[column].isna().all() and (source_present is None or source_present in frame):
            warnings.append(f"{column}: sin datos comparables con el esquema común")
    out["edad"] = out.edad.astype("float64")
    return out, peso_source


def _binary_codes(labels):
    """(código positivo, código negativo) si las etiquetas (sin N.P.) son un par Sí/No o Menciona/No menciona."""
    real = {code: cat.norm(label) for code, label in labels.items() if not _is_np(label)}
    if len(real) != 2:
        return None
    pos = [c for c, n in real.items() if n in _POSITIVE]
    neg = [c for c, n in real.items() if n in _NEGATIVE]
    if len(pos) == 1 and len(neg) == 1:
        return pos[0], neg[0]
    return None


def _candidate_variables(frame, meta, used):
    labels = meta.column_names_to_labels
    for name in frame.columns:
        if name in used or _EXCLUDED.match(name) or "recodificad" in clean(labels.get(name)).lower():
            continue
        if frame[name].dtype == object:
            continue
        yield name


def _answer_label(code, labels):
    return clean(labels.get(code)) if code in labels else str(int(code)) if float(code).is_integer() else str(code)


def build_questions(frame, meta, estudio_id, respondent_uids, used, topic, common):
    """Itera (registro de pregunta, DataFrame de respuestas)."""
    V, L = meta.variable_value_labels, meta.column_names_to_labels
    candidates = list(_candidate_variables(frame, meta, used))
    families, singles = {}, []
    for name in candidates:
        m = re.match(r"^(.*)_(\d+)$", name)
        if m and _binary_codes(V.get(name, {})):
            families.setdefault(m.group(1), []).append(name)
        else:
            singles.append(name)
    for prefix, members in list(families.items()):
        if len(members) < 2:
            singles.extend(members)
            del families[prefix]

    def record(uid, text, options, qtype, n_valid, variables):
        return {
            "question_uid": uid,
            "evidence_id": uid,
            "estudio_id": estudio_id,
            "study_id": estudio_id,
            "question_text": text,
            "question_summary": None,
            "options": options,
            "question_type": qtype,
            "topic": topic,
            "n_valid": int(n_valid),
            "variables": variables,
            **common,
        }

    for name in singles:
        labels = V.get(name, {})
        column = frame[name]
        valid = column.notna() & ~column.map(lambda c: _is_np(labels.get(c, ""))).fillna(False).astype(bool)
        values = column[valid]
        if len(values) < MIN_VALID or values.nunique() < 2:
            continue
        if values.nunique() > MAX_CATEGORIES or (not labels and values.nunique() > 12):
            continue  # numérica/abierta: no es una pregunta categórica
        answers = values.map(lambda c: _answer_label(c, labels))
        ordered = [clean(v) for k, v in sorted(labels.items()) if not _is_np(v)]
        extras = sorted({a for a in answers.unique() if a not in ordered})
        options = list(dict.fromkeys(ordered + extras))
        uid = f"CIS{estudio_id}_{name}"
        yield (
            record(uid, clean(L.get(name)) or name, options, "single", len(values), [name]),
            pd.DataFrame(
                {
                    "estudio_id": estudio_id,
                    "respondent_uid": respondent_uids[values.index].to_numpy(),
                    "question_uid": uid,
                    "answer": answers.to_numpy(),
                    "selected": pd.array([None] * len(values), dtype="boolean"),
                }
            ),
        )

    for prefix, members in families.items():
        codes = {name: _binary_codes(V[name]) for name in members}
        block = frame[members]
        valid = pd.Series(False, index=frame.index)
        for name, (pos, neg) in codes.items():
            valid |= block[name].isin([pos, neg])
        if valid.sum() < MIN_VALID:
            continue
        options, parts = [], []
        for name in members:
            option = clean(L.get(name)) or name
            if option in options:
                option = f"{option} ({name})"
            options.append(option)
            pos = codes[name][0]
            idx = frame.index[valid]
            parts.append(
                pd.DataFrame(
                    {
                        "estudio_id": estudio_id,
                        "respondent_uid": respondent_uids[idx].to_numpy(),
                        "question_uid": f"CIS{estudio_id}_{prefix}",
                        "answer": option,
                        "selected": (block.loc[idx, name] == pos).to_numpy(),
                    }
                )
            )
        text = f"Respuesta múltiple {prefix}: " + "; ".join(options[:8]) + ("…" if len(options) > 8 else "")
        yield (
            record(f"CIS{estudio_id}_{prefix}", text, options, "multiple", valid.sum(), members),
            pd.concat(parts, ignore_index=True),
        )


def process_study(path, folder, catalog_row, include_reserved=True):
    frame, meta, ft_text = read_zip(path)
    warnings = []
    estudio_id = re.search(r"CIS_(\d+)", Path(path).name).group(1)
    ft = parse_fieldwork(ft_text)
    micro = micro_dates(frame)
    if ft and micro and (abs((ft[0] - micro[0]).days) > 1 or abs((ft[1] - micro[1]).days) > 1):
        warnings.append(f"Fechas de campo discrepantes: ficha {ft[0]}..{ft[1]} vs microdatos {micro[0]}..{micro[1]}")
    discrepant = bool(warnings)  # sólo contiene el aviso de fechas hasta este punto
    if discrepant:
        chosen, source = micro, "microdatos"  # las marcas por entrevista prevalecen sobre erratas de la ficha
    else:
        chosen, source = (ft, "ficha_tecnica") if ft else (micro, "microdatos") if micro else (None, None)
    if chosen is None:
        warnings.append("Sin fecha de campo")
    respondents, peso_source = build_respondents(frame, meta, estudio_id, warnings)
    uids = respondents.respondent_uid
    used = set()
    lower = ft_text.lower()
    nationwide = "nacional" in lower.split("ámbito")[-1][:60] if "ámbito" in lower else False
    reserved = (catalog_row or {}).get("reservado_examen", "").strip().lower() == "si"
    common = {
        "source": f"CIS {estudio_id}",
        "fieldwork_start": chosen[0].isoformat() if chosen else None,
        "fieldwork_end": chosen[1].isoformat() if chosen else None,
        "fieldwork_date": chosen[1].isoformat() if chosen else None,
        "fecha_publicacion": (catalog_row or {}).get("fecha_publicacion") or None,
        "reservado_examen": reserved,
        "target_population": DEFAULT_UNIVERSE if nationwide else None,
        "survey_mode": "telefónica" if "telef" in lower else ("presencial" if "domicilio" in lower else None),
        "sample_size": int(len(frame)),
        "weight_column": "peso",
        "original_file": Path(path).name,
        "available_demographics": [
            c for c in ["sexo", "edad", "grupo_edad", "nivel_estudios", "ccaa", "tamuni", "ingresos_hogar_tramo"]
            if respondents[c].notna().any()
        ],
    }
    questions, chunks = [], []
    for q, responses in build_questions(frame, meta, estudio_id, uids, used, folder, common):
        questions.append(q)
        chunks.append(responses)
    study = {
        "estudio_id": estudio_id,
        "titulo": clean(Path(path).stem),
        "topic": folder,
        "fecha_publicacion": common["fecha_publicacion"],
        "reservado_examen": reserved,
        "fieldwork_start": common["fieldwork_start"],
        "fieldwork_end": common["fieldwork_end"],
        "fieldwork_source": source,
        "n_respondents": int(len(frame)),
        "n_questions": len(questions),
        "peso_source": peso_source,
        "survey_mode": common["survey_mode"],
        "warnings": warnings,
        "original_file": Path(path).name,
    }
    responses = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=RESPONSE_SCHEMA.names)
    return study, questions, respondents, responses


def run_etl(raw_dir, catalog_path, out_dir, only=None):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog = read_catalog(catalog_path) if catalog_path and Path(catalog_path).exists() else {}
    studies, questions, respondents, skipped = [], [], [], []
    seen = set()
    writer = pq.ParquetWriter(out_dir / "cis_responses.parquet", RESPONSE_SCHEMA)
    try:
        for zip_path in sorted(raw_dir.glob("*/cis/*.zip")):
            folder = zip_path.parent.parent.name
            number = re.search(r"CIS_(\d+)", zip_path.name)
            if folder in SKIP_FOLDERS:
                skipped.append({"file": zip_path.name, "reason": "carpeta excluida"})
                continue
            if only and (not number or number.group(1) not in only):
                continue
            row = catalog.get(zip_path.name)
            published = (row or {}).get("fecha_publicacion")
            if published and date.fromisoformat(published) > CUTOFF:
                skipped.append({"file": zip_path.name, "reason": "publicado después del 30/06/2026"})
                continue
            if number and number.group(1) in seen:
                skipped.append({"file": zip_path.name, "reason": "estudio duplicado en otra carpeta"})
                continue
            logger.info("Procesando %s (%s)", zip_path.name, folder)
            study, qs, resp, answers = process_study(zip_path, folder, row)
            seen.add(study["estudio_id"])
            studies.append(study)
            questions.extend(qs)
            respondents.append(resp)
            if len(answers):
                writer.write_table(pa.Table.from_pandas(answers, schema=RESPONSE_SCHEMA, preserve_index=False))
    finally:
        writer.close()
    pd.DataFrame(questions).to_parquet(out_dir / "cis_questions.parquet", index=False)
    pd.concat(respondents, ignore_index=True).to_parquet(out_dir / "cis_respondents.parquet", index=False)
    pd.DataFrame(studies).to_parquet(out_dir / "cis_studies.parquet", index=False)
    report = {
        "n_studies": len(studies),
        "n_questions": len(questions),
        "n_reserved_studies": sum(s["reservado_examen"] for s in studies),
        "skipped": skipped,
        "study_warnings": {s["estudio_id"]: s["warnings"] for s in studies if s["warnings"]},
    }
    (out_dir / "etl_cis_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
