"""ETL de encuestas europeas (Eurobarómetro y ESS) restringidas a España, como evidencia adicional.

No son del CIS ni casos de backtest: solo alimentan el RAG, con las mismas reglas de corte temporal.
Salida: ``external_evidence.json`` (compatible con ``ingest-surveys``), ``external_studies.parquet`` e informe.
"""

import html
import json
import logging
import re
import tempfile
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

from .. import categories as cat

logger = logging.getLogger(__name__)

CUTOFF = date(2026, 6, 30)
MIN_VALID = 300
MAX_OPTIONS = 15
EB_NOTE = "Encuesta europea (Eurobarómetro), muestra española de n≈1.000; no es del CIS"
ESS_NOTE = "Encuesta Social Europea (ESS), muestra española; no es del CIS"
_NON_SUBSTANTIVE = re.compile(
    r"\b(dk|don'?t know|refus\w*|not applicable|inap\w*|no answer|not available|missing|not asked|nap|na)\b", re.I
)
_EB_TECHNICAL = re.compile(
    r"^(studyno\d*|doi|version|edition|survey|respondent_serial|serialid|uniqid|tnscntry|country|isocntry|eu27b|"
    r"d\d+\w*|p\d+\w*|w\d+\w*|wex|nuts\w*|v\d+|q1\.\d+)$",
    re.I,
)
_ESS_TECHNICAL = re.compile(
    r"^(name|essround|edition|proddate|idno|cntry|dweight|pspwght|pweight|anweight|inw\w*|region\w*|regunit|"
    r"agea|gndr|eisced\w*|edulvl\w*|hinctnt\w*|domicil|prtv\w*|prtc\w*|prtd\w*|prtl\w*|prtm\w*|pt\w*|"
    r"mode|vdcond\w*|supqad\w*|supqmm|supqyr|supqdd|sample|stratum|psu|\w*lng|ctzcntr\w*|brnc\w*|"
    r"rlgdnm\w*|cntbrth\w*|ccnthum|ccrdprs|lnghom\w*|blgetmg|facntr|mocntr|edagepad|\w*cntbr\w*|\w*ctrb\w*)$",
    re.I,
)
_OTHER_COUNTRIES = (
    "BELGIUM DENMARK GERMANY GREECE FRANCE IRELAND ITALY LUXEMBOURG NETHERLANDS PORTUGAL AUSTRIA FINLAND SWEDEN "
    "BULGARIA CROATIA CYPRUS CZECH ESTONIA HUNGARY LATVIA LITHUANIA MALTA POLAND ROMANIA SLOVAKIA SLOVENIA "
    "BRITAIN NORTHERN"
).split()


_FIELD = re.compile(r"\(campo\s+([^)]*)\)", re.I)
_RANGE = re.compile(r"^(\d{1,2})(?:/(\d{1,2}))?(?:/(\d{4}))?\s*-\s*(\d{1,2})/(\d{1,2})/(\d{4})$")


def parse_field_dates(title):
    """Fechas de campo del título del catálogo: «(campo 26/03-10/05/2025)», «(campo 23-30/11/2022)».

    Devuelve (inicio, fin) en ISO o None si el título no trae un rango completo (p. ej. «campo 2022»).
    """
    match = _FIELD.search(title or "")
    range_match = _RANGE.match(match.group(1).strip()) if match else None
    if not range_match:
        return None
    d1, m1, y1, d2, m2, y2 = range_match.groups()
    end = date(int(y2), int(m2), int(d2))
    start = date(int(y1 or y2), int(m1 or m2), int(d1))
    if start > end:  # el campo cruza el año: 28/12-10/01/2025
        start = date(start.year - 1, start.month, start.day)
    return start.isoformat(), end.isoformat()


def read_external_catalog(path):
    """{nombre de fichero: {titulo, publicado}} de las filas encuesta_eurobarometro / encuesta_ess del catálogo."""
    import csv

    rows = {}
    if not path or not Path(path).exists():
        return rows
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("tipo") in {"encuesta_eurobarometro", "encuesta_ess"}:
                published = row.get("fecha_publicacion") or None
                rows[Path(row["archivo"].replace("\\", "/")).name] = {
                    "title": row.get("titulo", ""),
                    "published": date.fromisoformat(published) if published else None,
                }
    return rows


def _clean(text):
    return re.sub(r"\s+", " ", html.unescape(str(text or ""))).strip()


def _is_non_substantive(label):
    return bool(_NON_SUBSTANTIVE.search(label))


def _num(frame, column):
    """Columna numérica o NaN si el fichero no la trae (no todos los estudios incluyen todas las variables)."""
    if column in frame:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index)


def _grupo(age):
    return cat.age_group(age)


def eb_education(age_end):
    """Equivalencia aproximada: el Eurobarómetro pregunta la edad de fin de estudios (d8)."""
    out = pd.Series(np.nan, index=age_end.index, dtype="object")
    out[age_end == 97] = "sin_estudios"
    valid = age_end.between(1, 90)
    out[valid & (age_end <= 14)] = "primaria"
    out[valid & age_end.between(15, 16)] = "secundaria_1"
    out[valid & age_end.between(17, 19)] = "secundaria_2"
    out[valid & (age_end >= 20)] = "superiores"
    return out


def ess_education(eisced):
    mapping = {1: "primaria", 2: "secundaria_1", 3: "secundaria_2", 4: "secundaria_2", 5: "fp", 6: "superiores", 7: "superiores"}
    return eisced.map(mapping)


def parse_codebook(html_text, variables):
    """{variable: enunciado completo} desde el codebook HTML del ESS (texto entre la etiqueta y «Value»)."""
    text = re.sub(r"<[^>]+>", "\n", html_text)
    lines = [_clean(x) for x in text.splitlines()]
    lines = [x for x in lines if x]
    wanted = set(variables)
    wording = {}
    for i, line in enumerate(lines[:-2]):
        if line in wanted and line not in wording:
            end = next((j for j in range(i + 2, min(i + 14, len(lines))) if lines[j] == "Value"), None)
            if end is not None and end > i + 2:
                wording[line] = " ".join(lines[i + 2 : end])
    return wording


def _weighted_distribution(values, weights, options):
    sums = pd.Series(weights.to_numpy(), index=values.to_numpy()).groupby(level=0).sum()
    chosen = sums.reindex(options).fillna(0.0)
    mass = chosen.sum()
    if mass <= 0:
        return None
    n_eff = float(mass**2 / (weights**2).sum())
    return {o: float(v / mass) for o, v in chosen.items()}, n_eff


def _questions(
    frame, meta, spain, weight_col, prefix, study_id, topic, wording, technical, note, source, dates, published=None
):
    """Registros de evidencia para las variables categóricas con datos españoles suficientes."""
    L, V = meta.column_names_to_labels, meta.variable_value_labels
    w = pd.to_numeric(spain[weight_col], errors="coerce").fillna(0.0)
    sexo, grupo = spain["_sexo"], spain["_grupo"]
    records, families = [], {}
    columns = [c for c in frame.columns if not technical.match(c) and c in V]
    # Baterías de menciones (Mentioned / Not mentioned) -> una pregunta múltiple
    for c in columns:
        labels = {k: str(v).lower() for k, v in V[c].items()}
        if set(labels.values()) == {"mentioned", "not mentioned"}:
            families.setdefault(re.sub(r"[._]\d+$", "", c), []).append(c)
    used = set()
    for base, members in families.items():
        if len(members) < 2:
            continue
        used.update(members)
        text = _clean(L.get(members[0]) or base)
        options = [_clean((L.get(m) or m).split(":")[-1]) for m in members]
        sel = pd.DataFrame({m: pd.to_numeric(spain[m], errors="coerce") for m in members})
        answered = sel.notna().any(axis=1) & (w > 0)
        if answered.sum() < MIN_VALID:
            continue
        shares = {}
        for m, o in zip(members, options):
            positive = [k for k, v in V[m].items() if str(v).lower() == "mentioned"][0]
            shares[o] = float(np.average((sel.loc[answered, m] == positive).astype(float), weights=w[answered]))
        records.append(
            _record(prefix, base, study_id, topic, text.split(":")[0] + " (multiple response)", text, options, "multiple",
                    {"distribution": shares, "n_respondents": int(answered.sum()), "n_eff": float((w[answered].sum() ** 2) / (w[answered] ** 2).sum())},
                    {}, note, source, dates, weight_col, published)
        )
    for c in columns:
        if c in used or not 2 <= len(V[c]) <= 40:
            continue
        label = _clean(L.get(c) or c)
        if re.search(r"(?i)(?:^|[:\s])(" + "|".join(_OTHER_COUNTRIES) + r")\b", label.split(":")[-1]) and ":" in label:
            continue
        values = pd.to_numeric(spain[c], errors="coerce")
        options_map = {k: _clean(v) for k, v in V[c].items()}
        substantive = {k: v for k, v in options_map.items() if not _is_non_substantive(v)}
        if not 2 <= len(substantive) <= MAX_OPTIONS:
            continue
        valid_all = values.notna() & (w > 0)
        valid = valid_all & values.isin(list(substantive))
        if valid.sum() < MIN_VALID:
            continue
        answers = values[valid].map(substantive)
        options = list(dict.fromkeys(substantive.values()))
        result = _weighted_distribution(answers, w[valid], options)
        if result is None or max(result[0].values()) > 0.995:
            continue
        stats = {"distribution": result[0], "n_respondents": int(valid.sum()), "n_eff": result[1],
                 "ns_nc_share": float(1 - w[valid].sum() / w[valid_all].sum())}
        subgroups = {}
        for dim, series in (("sexo", sexo), ("grupo_edad", grupo)):
            groups = {}
            for category, idx in answers.groupby(series[valid]).groups.items():
                if len(idx) >= 100:
                    got = _weighted_distribution(answers.loc[idx], w[valid].loc[idx], options)
                    if got:
                        groups[str(category)] = {"distribution": got[0], "n_respondents": len(idx)}
            if groups:
                subgroups[dim] = groups
        text = wording.get(c) or label
        records.append(_record(prefix, c, study_id, topic, label, text, options, "single", stats, subgroups, note, source, dates, weight_col, published))
    return records


def _record(
    prefix, variable, study_id, topic, summary, text, options, qtype, national, subgroups, note, source, dates, weight_col,
    published=None,
):
    uid = f"{prefix}_{variable}".replace(" ", "")
    return {
        "evidence_id": uid,
        "question_uid": uid,
        "source": source,
        "study_id": study_id,
        "estudio_id": study_id,
        "question_text": _clean(text)[:700],
        "question_summary": _clean(summary)[:200],
        "options": options,
        "question_type": qtype,
        "topic": topic,
        "idioma": "en",
        "fieldwork_start": dates[0],
        "fecha_publicacion": published.isoformat() if published else None,
        "fieldwork_end": dates[1],
        "fieldwork_date": dates[1],
        "reservado_examen": False,
        "target_population": None,
        "sample_size": national["n_respondents"],
        "weight_column": weight_col,
        "survey_mode": None,
        "available_demographics": sorted(subgroups),
        "original_file": study_id,
        "metadata": {
            "survey_note": note,
            "empirical_statistics": {"national": national, **subgroups},
            "language": "en",
            "n_spain": national["n_respondents"],
        },
    }


def _read(path):
    if str(path).lower().endswith(".dta"):
        return pyreadstat.read_dta(str(path))
    return pyreadstat.read_sav(str(path))


def process_eurobarometer(path, topic, catalog_row=None):
    frame, meta = _read(path)
    frame.columns = [c.lower() for c in frame.columns]
    meta.column_names_to_labels = {k.lower(): v for k, v in meta.column_names_to_labels.items()}
    meta.variable_value_labels = {k.lower(): v for k, v in meta.variable_value_labels.items()}
    country = "isocntry" if "isocntry" in frame else "country"
    spain = frame[frame[country].astype(str).str.upper().str.startswith("ES")].copy()
    study = Path(path).name.split("_")[0]
    report = {"study_id": study, "topic": topic, "n_spain": int(len(spain)), "format": "eurobarometro"}
    if spain.empty or "w1" not in spain:
        report["skipped"] = "sin España o sin peso w1"
        return [], report
    spain["_sexo"] = _num(spain, "d10").map({1: "hombre", 2: "mujer"})
    spain["_grupo"] = _grupo(_num(spain, "d11").where(lambda x: x < 97))
    spain["_estudios"] = eb_education(_num(spain, "d8"))
    # La fecha de campo es la del catálogo (título) o, si falta, la de las entrevistas. La fecha de publicación
    # (catálogo o variable version) solo sirve para la regla del 30/06/2026: GESIS publica 1-2 años después del campo.
    catalog_dates = parse_field_dates((catalog_row or {}).get("title"))
    micro_dates = _eb_dates(spain, meta)
    dates = catalog_dates or micro_dates
    published = (catalog_row or {}).get("published") or _eb_published(spain)
    report["published"] = published.isoformat() if published else None
    report["fieldwork_source"] = "catalogo" if catalog_dates else ("entrevistas" if micro_dates[1] else None)
    if catalog_dates and micro_dates[1] and micro_dates != catalog_dates:
        report["date_mismatch"] = {"catalogo": catalog_dates, "entrevistas": micro_dates}
    if published and published > CUTOFF:
        report["skipped"] = f"archivo publicado el {published.isoformat()}, posterior al 30/06/2026"
        return [], report
    report["fieldwork"] = dates
    if dates[1] is None:
        report["warning"] = "sin fecha de trabajo de campo: la evidencia quedará fuera de cualquier corte temporal"
    elif date.fromisoformat(dates[1]) > CUTOFF:
        report["skipped"] = "trabajo de campo posterior al 30/06/2026"
        return [], report
    records = _questions(frame, meta, spain, "w1", f"EB{study}", f"EB-{study}", topic, {}, _EB_TECHNICAL,
                         EB_NOTE, "eurobarometro", dates, published)
    report["n_questions"] = len(records)
    return records, report


def _eb_published(spain):
    """Fecha de publicación del archivo GESIS: «1.0.0 (2025-02-26)» en la variable version."""
    if "version" not in spain or spain["version"].dropna().empty:
        return None
    match = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(spain["version"].dropna().iloc[0]))
    return date(int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None


def _eb_dates(spain, meta):
    labels = meta.variable_value_labels.get("p1", {})
    stamps = []
    for code in _num(spain, "p1").dropna().unique():
        text = labels.get(code) or labels.get(float(code))
        try:
            stamps.append(datetime.strptime(str(text), "%d-%b-%Y").date())
        except ValueError:
            continue
    return (min(stamps).isoformat(), max(stamps).isoformat()) if stamps else (None, None)


def process_ess(path, codebook_html, topic, catalog_row=None):
    frame, meta = _read(path)
    spain = frame[frame["cntry"].astype(str).str.upper() == "ES"].copy()
    rnd = int(_num(spain, "essround").dropna().iloc[0])
    name = f"ESS{rnd}" + ("SC" if "SC" in Path(path).name else "")
    report = {"study_id": name, "topic": topic, "n_spain": int(len(spain)), "format": "ess"}
    spain["_sexo"] = _num(spain, "gndr").map({1: "hombre", 2: "mujer"})
    spain["_grupo"] = _grupo(_num(spain, "agea").where(lambda x: x < 999))
    spain["_estudios"] = ess_education(_num(spain, "eisced"))
    produced = _ess_production_date(spain)
    report["produced"] = produced.isoformat() if produced else None
    # Publicación según el catálogo si existe (puede documentar que los datos de España ya salieron en una
    # edición anterior); si no, la fecha de producción del fichero. Regla del reto: nada posterior al 30/06/2026.
    published = (catalog_row or {}).get("published") or produced
    report["published"] = published.isoformat() if published else None
    if (catalog_row or {}).get("published") and produced and produced > CUTOFF:
        report["edition_note"] = (
            f"fichero producido el {produced.isoformat()}; se usa la publicación del catálogo "
            f"({published.isoformat()}), que indica datos de España idénticos a la edición anterior"
        )
    if published and published > CUTOFF:
        report["skipped"] = f"edición publicada el {published.isoformat()}, posterior al 30/06/2026"
        return [], report
    dates = _ess_dates(spain)
    catalog_dates = parse_field_dates((catalog_row or {}).get("title"))
    if catalog_dates:
        dates = catalog_dates
    report["fieldwork"] = dates
    wording = parse_codebook(codebook_html, frame.columns) if codebook_html else {}
    report["wording_from_codebook"] = len(wording)
    records = _questions(frame, meta, spain, "anweight", name, name, topic, wording, _ESS_TECHNICAL, ESS_NOTE, "ess", dates, published)
    report["n_questions"] = len(records)
    return records, report


def _ess_production_date(spain):
    if "proddate" not in spain or spain["proddate"].dropna().empty:
        return None
    try:
        return datetime.strptime(str(spain["proddate"].dropna().iloc[0]), "%d.%m.%Y").date()
    except ValueError:
        return None


def _ess_dates(spain):
    stamps = []
    for y, m, d in (("inwyys", "inwmms", "inwdds"), ("inwyye", "inwmme", "inwdde")):
        if {y, m, d} <= set(spain.columns):
            part = spain[[y, m, d]].apply(pd.to_numeric, errors="coerce").dropna()
            part = part[(part[y] > 2000) & part[m].between(1, 12) & part[d].between(1, 31)]
            if not part.empty:
                ts = pd.to_datetime(pd.DataFrame({"year": part[y], "month": part[m], "day": part[d]}), errors="coerce").dropna()
                stamps += [ts.min().date(), ts.max().date()]
    for column in ("inwds", "scwsds"):
        if not stamps and column in spain:
            ts = pd.to_datetime(spain[column], errors="coerce").dropna()
            if not ts.empty:
                stamps = [ts.min().date(), ts.max().date()]
    return (min(stamps).isoformat(), max(stamps).isoformat()) if stamps else (None, None)


def run_external_etl(raw_dir, out_dir, only=None, catalog=None):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records, reports = [], []
    rows = read_external_catalog(catalog or raw_dir.parent / "catalogo.csv")
    for path in sorted(raw_dir.glob("*/eurobarometro/*.sav")):
        if only and path.name.split("_")[0] not in only:
            continue
        logger.info("Eurobarómetro %s", path.name)
        recs, report = process_eurobarometer(path, path.parent.parent.name, rows.get(path.name))
        records += recs
        reports.append(report)
    for zip_path in sorted((raw_dir / "general" / "ess").glob("*.zip")):
        if "csv" in zip_path.name.lower() or (only and not any(o.lower() in zip_path.name.lower() for o in only)):
            continue  # el CSV duplica al Stata
        with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as tmp:
            data = next(n for n in z.namelist() if n.lower().endswith((".sav", ".dta")))
            book = next((n for n in z.namelist() if n.lower().endswith(".html")), None)
            z.extract(data, tmp)
            html_text = z.read(book).decode("utf-8", errors="ignore") if book else ""
            logger.info("ESS %s", zip_path.name)
            recs, report = process_ess(Path(tmp) / data, html_text, "general", rows.get(zip_path.name))
        records += recs
        reports.append(report)
    (out_dir / "external_evidence.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(reports).to_json(out_dir / "external_studies.json", orient="records", force_ascii=False, indent=2)
    summary = {
        "studies": len(reports),
        "questions": len(records),
        "n_spain": {r["study_id"]: r["n_spain"] for r in reports},
        "skipped": {r["study_id"]: r["skipped"] for r in reports if r.get("skipped")},
        "warnings": {r["study_id"]: r["warning"] for r in reports if r.get("warning")},
        "date_mismatches": {r["study_id"]: r["date_mismatch"] for r in reports if r.get("date_mismatch")},
        "edition_notes": {r["study_id"]: r["edition_note"] for r in reports if r.get("edition_note")},
    }
    (out_dir / "etl_external_report.json").write_text(json.dumps({**summary, "detail": reports}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
