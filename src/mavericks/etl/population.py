"""ETL de la población sintética (Fase 1).

Semilla: microdatos del Censo 2021 (adultos). Se reponderan con IPF (raking) a:
  * Cifras de Población (ECP) a una fecha de referencia anterior al 30/06/2026:
    CCAA x sexo x grupo de edad, CCAA x nacionalidad y nacionalidad x sexo x grupo de edad.
  * EPA 2T26: nivel de estudios por sexo x grupo de edad y por CCAA (25 años o más).
Después se extrae una muestra de tamaño n (semilla fija) con probabilidad proporcional al peso
y se imputan los ingresos del hogar a partir de la ECV 2025. Todas las categorías siguen
el esquema común del CIS (``mavericks.categories``).
"""

import io
import json
import logging
import re
import tempfile
import zipfile
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import categories as cat

logger = logging.getLogger(__name__)

CUTOFF = date(2026, 6, 30)
CENSO_2021_POBLACION = 47_385_107  # población residente del Censo 2021 (para estimar tamaño de municipio)
POPULATION_VERSION_PREFIX = "SP2026"

# CCAA (índice de cat.CCAA) de cada provincia (código INE de dos dígitos).
_PROV_CCAA = {
    0: "04 11 14 18 21 23 29 41",
    1: "22 44 50",
    2: "33",
    3: "07",
    4: "35 38",
    5: "39",
    6: "05 09 24 34 37 40 42 47 49",
    7: "02 13 16 19 45",
    8: "08 17 25 43",
    9: "03 12 46",
    10: "06 10",
    11: "15 27 32 36",
    12: "28",
    13: "30",
    14: "31",
    15: "01 20 48",
    16: "26",
    17: "51",
    18: "52",
}
PROV_TO_CCAA = {p: i for i, provs in _PROV_CCAA.items() for p in provs.split()}
_ECV_REGION = {
    "ES11": 11, "ES12": 2, "ES13": 5, "ES21": 15, "ES22": 14, "ES23": 16, "ES24": 1, "ES30": 12,
    "ES41": 6, "ES42": 7, "ES43": 10, "ES51": 8, "ES52": 9, "ES53": 3, "ES61": 0, "ES62": 13,
    "ES63": 17, "ES64": 18, "ES70": 4,
}  # fmt: skip
_EPA_CCAA = {f"{i + 1:02d}": i for i in range(17)} | {"51": 17, "52": 18}
_CNEDA = {  # Censo: ESREAL_CNEDA -> nivel común (índice de cat.NIVEL_ESTUDIOS)
    "01": 0, "02": 0, "03": 1, "04": 2, "05": 3, "06": 4, "07": 4,
    "08": 5, "09": 5, "10": 5, "11": 5, "12": 5,
}  # fmt: skip
_EPA_EDU = {"AN": 0, "P1": 0, "P2": 1, "S1": 2, "SG": 3, "SP": 4, "SU": 5}
_ECV_EDU = {"000": 0, "100": 1, "200": 2, "340": 3, "344": 3, "350": 4, "353": 4, "354": 4, "450": 4, "500": 5}
_GROUP_LOWER = np.array([18, 25, 35, 45, 55, 65, 75])
_COARSE = np.array([-1, 0, 1, 2, 3, 4, 4])  # grupos CIS -> grupos de la EPA (65 o más agrupado)
_LABOR = {"1": "ocupado", "2": "parado", "4": "pensionista", "5": "pensionista", "6": "otra_inactividad", "7": "estudiante"}
_NATIONALITY = ["española", "extranjera"]
LABOR_CATEGORIES = ["ocupado", "parado", "pensionista", "estudiante", "otra_inactividad"]
HOUSEHOLD_SIZE = [1, 2, 3, 4]  # 4 = 4 o más
INCOME_EDGES = [5000, 3900, 2700, 1800, 1100]  # mensuales; índice 0 = más de 5.000


# --------------------------------------------------------------------------- utilidades


def parse_period(label):
    """«1 de enero de 2026» -> date. Rechaza referencias posteriores al corte del reto."""
    months = "enero febrero marzo abril mayo junio julio agosto septiembre octubre noviembre diciembre".split()
    m = re.match(r"\s*(\d{1,2}) de (\w+) de (\d{4})", label)
    if not m or m.group(2) not in months:
        raise ValueError(f"Periodo no reconocido: {label!r}")
    value = date(int(m.group(3)), months.index(m.group(2)) + 1, int(m.group(1)))
    if value > CUTOFF:
        raise ValueError(f"El periodo {label!r} es posterior al 30/06/2026: no se puede usar")
    return value


def income_tramo(monthly):
    """Ingreso mensual del hogar (€) -> índice de cat.INGRESOS (0 = más de 5.000 €)."""
    monthly = np.asarray(monthly, dtype=float)
    return np.select([monthly > 5000, monthly > 3900, monthly > 2700, monthly > 1800, monthly >= 1100], [0, 1, 2, 3, 4], 5)


def group_index(ages):
    """Edades -> índice de cat.GRUPOS_EDAD (-1 si <18)."""
    ages = np.asarray(ages)
    return np.where(ages < 18, -1, np.searchsorted(_GROUP_LOWER, ages, side="right") - 1)


def municipality_size_class(estimated_population):
    """Población estimada de municipio (>10.000) -> índice de cat.TAMUNI."""
    edges = np.array([50_000, 100_000, 400_000, 1_000_000])
    return 2 + np.searchsorted(edges, np.asarray(estimated_population), side="left")


def ipf(weights, margins, max_iter=200, tol=1e-7):
    """Raking sobre una semilla. margins: lista de dict(index=array int, target=array, mask=bool|None).

    ``index`` codifica la celda de cada fila (-1 = fuera del margen). Devuelve (pesos, informe).
    """
    w = np.asarray(weights, dtype=float).copy()
    error = np.inf
    for iteration in range(max_iter):
        error = 0.0
        for m in margins:
            idx, target = m["index"], np.asarray(m["target"], dtype=float).ravel()
            inside = idx >= 0
            current = np.bincount(idx[inside], weights=w[inside], minlength=len(target))
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(current > 0, target / current, 1.0)
            w[inside] *= ratio[idx[inside]]
            reachable = current > 0
            if reachable.any():
                error = max(error, float(np.max(np.abs(current[reachable] - target[reachable]) / np.maximum(target[reachable], 1.0))))
        if error < tol:
            break
    report = {"iterations": iteration + 1, "max_relative_gap": error, "empty_cells": []}
    for k, m in enumerate(margins):
        idx, target = m["index"], np.asarray(m["target"], dtype=float).ravel()
        inside = idx >= 0
        current = np.bincount(idx[inside], weights=w[inside], minlength=len(target))
        missing = (current == 0) & (target > 0)
        report["empty_cells"].append({"margin": m.get("name", k), "n": int(missing.sum()), "target_lost": float(target[missing].sum())})
    return w, report


def systematic_sample(weights, n, rng):
    """Muestreo sistemático PPS: n índices (con posible repetición de filas de peso alto)."""
    order = rng.permutation(len(weights))
    cumulative = np.cumsum(weights[order])
    points = (rng.random() + np.arange(n)) * (cumulative[-1] / n)
    return order[np.searchsorted(cumulative, points, side="right").clip(max=len(order) - 1)]


# --------------------------------------------------------------------------- lectura de fuentes


def _extract(zip_path, member, tmp):
    with zipfile.ZipFile(zip_path) as z:
        z.extract(member, tmp)
    return Path(tmp) / member


def load_census(zip_path):
    """Adultos del Censo 2021 como arrays compactos + conteos por municipio."""
    columns = ["CPRO", "CMUN", "VAREDAD", "SEXO", "PNACIO", "ESREAL_CNEDA", "RELA", "TAM_HOG"]
    municipality_counts = {}
    parts = []
    with tempfile.TemporaryDirectory() as tmp:
        path = _extract(zip_path, "md_CensoPersonas_2021.parquet", tmp)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=500_000, columns=columns):
            df = batch.to_pandas()
            for key, n in df.groupby(["CPRO", "CMUN"], observed=True).size().items():
                municipality_counts[key] = municipality_counts.get(key, 0) + int(n)
            age = pd.to_numeric(df.VAREDAD, errors="coerce")
            adult = age >= 18
            df = df[adult]
            parts.append(
                pd.DataFrame(
                    {
                        "cpro": df.CPRO.to_numpy(),
                        "cmun": df.CMUN.to_numpy(),
                        "edad": age[adult].to_numpy(dtype=np.int16),
                        "sexo": (df.SEXO == "6").to_numpy(dtype=np.int8),  # 0 hombre, 1 mujer
                        "extranjera": (df.PNACIO != "108").to_numpy(dtype=np.int8),
                        "cneda": df.ESREAL_CNEDA.to_numpy(),
                        "rela": df.RELA.to_numpy(),
                        "tam_hog": pd.to_numeric(df.TAM_HOG, errors="coerce").fillna(2).to_numpy(dtype=np.int8),
                    }
                )
            )
    return pd.concat(parts, ignore_index=True), municipality_counts


def census_seed(frame, municipality_counts):
    total_rows = sum(municipality_counts.values())
    scale = CENSO_2021_POBLACION / total_rows
    ccaa = frame.cpro.map(PROV_TO_CCAA)
    sizes = {k: v * scale for k, v in municipality_counts.items()}
    small = {"991": 0, "992": 1, "993": 1}
    keys = pd.Series(list(zip(frame.cpro, frame.cmun)))
    lookup = {}
    for key in keys.unique():
        if key[1] in small:
            lookup[key] = small[key[1]]
        else:
            lookup[key] = int(municipality_size_class(sizes.get(key, 0)))
    tamuni = keys.map(lookup).to_numpy(dtype=np.int8)
    edu = frame.cneda.map(_CNEDA)
    seed = pd.DataFrame(
        {
            "ccaa": ccaa.to_numpy(dtype=float),
            "sexo": frame.sexo.to_numpy(),
            "edad": frame.edad.to_numpy(),
            "grupo": group_index(frame.edad.to_numpy()),
            "nac": frame.extranjera.to_numpy(),
            "edu": edu.to_numpy(dtype=float),
            "tamuni": tamuni,
            "laboral": frame.rela.map(_LABOR).fillna("otra_inactividad").to_numpy(),
            "tam_hog": np.clip(frame.tam_hog.to_numpy(), 1, 4),
        }
    )
    seed = seed.dropna(subset=["ccaa", "edu"]).copy()
    seed["ccaa"] = seed.ccaa.astype(int)
    seed["edu"] = seed.edu.astype(int)
    return seed.reset_index(drop=True)


def _spanish_number(series):
    return pd.to_numeric(series.str.replace(".", "", regex=False), errors="coerce")


def load_ecp_targets(ecp_dir, period):
    """Objetivo ECP: array [ccaa, sexo, nac, grupo] de adultos para el periodo dado."""
    parse_period(period)
    ecp_dir = Path(ecp_dir)
    big = pd.read_csv(
        ecp_dir / "INE_ECP_60127_ccaa_sexo_grupoedad_nacionalidad.csv", sep=";", encoding="utf-8-sig", dtype=str
    )
    big = big[big.Periodo == period]
    if big.empty:
        raise ValueError(f"Periodo sin datos en la ECP: {period}")
    big = big.assign(total=_spanish_number(big.Total))
    simple = pd.read_csv(ecp_dir / "INE_ECP_60124_nacional_sexo_edad.csv", sep=";", encoding="utf-8-sig", dtype=str)
    simple = simple[(simple.Periodo == period) & simple.Sexo.isin(["Hombres", "Mujeres"])]
    simple = simple.assign(age=pd.to_numeric(simple["Edad simple"].str.extract(r"^(\d+)")[0], errors="coerce"), total=_spanish_number(simple.Total))
    frac = {}
    for sexo in ["Hombres", "Mujeres"]:
        s = simple[simple.Sexo == sexo].set_index("age").total
        frac[sexo] = float(s.loc[18:19].sum() / s.loc[15:19].sum())
    labels_to_group = {}
    for lo in range(15, 95, 5):
        label = "90 y más años" if lo == 90 else f"De {lo} a {lo + 4} años"
        labels_to_group[label] = lo
    target = np.zeros((19, 2, 2, 7))
    rows = big[big.Nacionalidad.isin(["Española", "Extranjera"]) & big.Sexo.isin(["Hombres", "Mujeres"])]
    for row in rows.itertuples():
        if not row[1][:2].isdigit():
            continue
        label = row[3]
        lo = labels_to_group.get(label)
        if lo is None:
            continue
        c = int(row[1][:2]) - 1
        s = 0 if row.Sexo == "Hombres" else 1
        n = 0 if row.Nacionalidad == "Española" else 1
        if lo == 15:
            g, value = 0, row.total * frac[row.Sexo]
        else:
            g, value = int(group_index(np.array([lo]))[0]), row.total
        if g >= 0:
            target[c, s, n, g] += value
    if target.sum() < 3e7:
        raise ValueError("Objetivos de la ECP incompletos: revisar el fichero")
    return target


def load_epa(zip_path):
    """EPA: DataFrame con ccaa, sexo, coarse (0-4), edu y factor de elevación (25 años o más)."""
    with tempfile.TemporaryDirectory() as tmp:
        member = next(n for n in zipfile.ZipFile(zip_path).namelist() if n.endswith(".parquet"))
        df = pd.read_parquet(_extract(zip_path, member, tmp), columns=["CCAA", "EDAD1", "SEXO1", "NFORMA", "FACTOREL"])
    df = df[df.EDAD1.isin(["25", "30", "35", "40", "45", "50", "55", "60", "65"]) & df.NFORMA.isin(_EPA_EDU)]
    coarse = {"25": 0, "30": 0, "35": 1, "40": 1, "45": 2, "50": 2, "55": 3, "60": 3, "65": 4}
    return pd.DataFrame(
        {
            "ccaa": df.CCAA.map(_EPA_CCAA).to_numpy(),
            "sexo": (df.SEXO1 == "6").to_numpy(dtype=int),
            "coarse": df.EDAD1.map(coarse).to_numpy(),
            "edu": df.NFORMA.map(_EPA_EDU).to_numpy(),
            "w": pd.to_numeric(df.FACTOREL, errors="coerce").to_numpy(),
        }
    ).dropna().astype({"ccaa": int, "sexo": int, "coarse": int, "edu": int})


def load_ecv_adults(zip_path):
    """ECV: adultos con renta mensual del hogar, tamaño, estudios, CCAA y peso."""
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as tmp:
        tables = {}
        for key in ["Td", "Th", "Tp"]:
            inner = next(n for n in z.namelist() if n.startswith(f"ECV_{key}_"))
            with zipfile.ZipFile(io.BytesIO(z.read(inner))) as zi:
                member = next(n for n in zi.namelist() if n.endswith(".parquet"))
                zi.extract(member, tmp)
                tables[key] = pd.read_parquet(Path(tmp) / member)
    td = tables["Td"][["DB030", "DB040"]].rename(columns={"DB030": "hid", "DB040": "region"})
    th = tables["Th"][["HB030", "HY020", "HX040"]].rename(columns={"HB030": "hid"})
    tp = tables["Tp"][["PB030", "PB040", "PB140", "PB150", "PE041"]].copy()
    tp["hid"] = pd.to_numeric(tp.PB030) // 100
    year = int(tables["Td"].DB010.iloc[0])
    tp["age"] = year - pd.to_numeric(tp.PB140, errors="coerce")
    tp = tp[(tp.age >= 18) & tp.PE041.isin(_ECV_EDU)]
    merged = tp.merge(th.assign(hid=pd.to_numeric(th.hid)), on="hid").merge(td.assign(hid=pd.to_numeric(td.hid)), on="hid")
    return pd.DataFrame(
        {
            "grupo": group_index(merged.age.to_numpy()),
            "edu": merged.PE041.map(_ECV_EDU).to_numpy(),
            "tam_hog": np.clip(pd.to_numeric(merged.HX040), 1, 4).to_numpy(),
            "ccaa": merged.region.map(_ECV_REGION).to_numpy(dtype=float),
            "tramo": income_tramo(pd.to_numeric(merged.HY020).to_numpy() / 12),
            "w": pd.to_numeric(merged.PB040).to_numpy(),
        }
    ).dropna()


# --------------------------------------------------------------------------- imputación y ensamblado


class IncomeImputer:
    """P(tramo | edad, estudios, tamaño del hogar, CCAA) con contracción jerárquica hacia niveles más gruesos."""

    LEVELS = [["grupo", "tam_hog"], ["grupo", "edu", "tam_hog"], ["grupo", "edu", "tam_hog", "ccaa"]]

    def __init__(self, ecv, k=30.0):
        self.k = k
        self.ecv = ecv
        total = np.bincount(ecv.tramo, weights=ecv.w, minlength=6)
        self.base = total / total.sum()
        self.tables = []
        for keys in self.LEVELS:
            table = {}
            for key, g in ecv.groupby(keys):
                p = np.bincount(g.tramo, weights=g.w, minlength=6)
                table[key if isinstance(key, tuple) else (key,)] = (p / p.sum(), len(g))
            self.tables.append(table)

    def distribution(self, grupo, edu, tam_hog, ccaa):
        values = {"grupo": grupo, "edu": edu, "tam_hog": tam_hog, "ccaa": ccaa}
        p = self.base
        for keys, table in zip(self.LEVELS, self.tables):
            hit = table.get(tuple(float(values[k]) if k == "ccaa" else values[k] for k in keys))
            if hit is not None:
                cell, n = hit
                p = (n * cell + self.k * p) / (n + self.k)
        return p

    def impute(self, frame, rng):
        out = np.zeros(len(frame), dtype=int)
        keys = frame[["grupo", "edu", "tam_hog", "ccaa"]].drop_duplicates()
        for row in keys.itertuples(index=False):
            mask = (
                (frame.grupo == row.grupo) & (frame.edu == row.edu) & (frame.tam_hog == row.tam_hog) & (frame.ccaa == row.ccaa)
            ).to_numpy()
            out[mask] = rng.choice(6, size=int(mask.sum()), p=self.distribution(*row))
        return out


def build_margins(seed, target, epa):
    """Márgenes de IPF con códigos de celda para cada fila de la semilla."""
    g = seed.grupo.to_numpy()
    c, s, n, e = seed.ccaa.to_numpy(), seed.sexo.to_numpy(), seed.nac.to_numpy(), seed.edu.to_numpy()
    m1 = target.sum(axis=2)  # ccaa, sexo, grupo
    m2 = target.sum(axis=(1, 3))  # ccaa, nac
    m3 = target.sum(axis=0)  # sexo, nac, grupo -> reordenar a (nac, sexo, grupo)
    m3 = np.transpose(m3, (1, 0, 2))
    margins = [
        {"name": "ccaa_sexo_grupo", "index": (c * 2 + s) * 7 + g, "target": m1},
        {"name": "ccaa_nacionalidad", "index": c * 2 + n, "target": m2},
        {"name": "nacionalidad_sexo_grupo", "index": (n * 2 + s) * 7 + g, "target": m3},
    ]
    coarse = _COARSE[g + 0]
    adult25 = coarse >= 0
    # nivel de estudios x sexo x grupo grueso (EPA, nivel nacional)
    t_sc = np.zeros((2, 5))
    for s_i in range(2):
        for k in range(5):
            groups = [i for i in range(7) if _COARSE[i] == k]
            t_sc[s_i, k] = target[:, s_i, :, :][:, :, groups].sum()
    shares = np.zeros((6, 2, 5))
    for (s_i, k), sub in epa.groupby(["sexo", "coarse"]):
        counts = np.bincount(sub.edu, weights=sub.w, minlength=6)
        shares[:, s_i, k] = counts / counts.sum()
    t_e_sc = shares * t_sc[None]
    idx = np.where(adult25, (e * 2 + s) * 5 + coarse, -1)
    margins.append({"name": "estudios_sexo_grupo", "index": idx, "target": t_e_sc})
    # nivel de estudios x CCAA (EPA, 25 años o más)
    t_c = np.array([target[ci][:, :, 1:].sum() for ci in range(19)])
    sh = np.zeros((6, 19))
    for ci, sub in epa.groupby("ccaa"):
        counts = np.bincount(sub.edu, weights=sub.w, minlength=6)
        sh[:, int(ci)] = counts / counts.sum()
    idx = np.where(adult25, e * 19 + c, -1)
    margins.append({"name": "estudios_ccaa", "index": idx, "target": sh * t_c[None]})
    return margins


def run_population_etl(raw_dir, out_dir, n=50_000, seed_value=42, period="1 de enero de 2026", epa="INE_EPA_2t26.zip", ecv="INE_ECV_2025.zip"):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pop_dir = raw_dir / "general" / "poblacion"
    logger.info("Leyendo Censo 2021")
    frame, counts = load_census(pop_dir / "censo2021" / "INE_Censo2021_Personas.zip")
    seed = census_seed(frame, counts)
    del frame
    logger.info("Semilla: %s adultos del Censo", f"{len(seed):,}")
    target = load_ecp_targets(pop_dir / "ecp", period)
    epa_frame = load_epa(pop_dir / "epa" / epa)
    margins = build_margins(seed, target, epa_frame)
    weights, ipf_report = ipf(np.ones(len(seed)), margins)
    adult_total = float(target.sum())
    rng = np.random.default_rng(seed_value)
    picked = systematic_sample(weights, n, rng)
    sample = seed.iloc[picked].reset_index(drop=True)
    ecv_adults = load_ecv_adults(raw_dir / "vivienda" / "ecv" / ecv)
    tramo = IncomeImputer(ecv_adults).impute(sample, rng)
    version = f"{POPULATION_VERSION_PREFIX}-{parse_period(period).isoformat()}-n{n}-seed{seed_value}"
    population = pd.DataFrame(
        {
            "person_id": [f"SP{i + 1:07d}" for i in range(n)],
            "peso": adult_total / n,
            "sexo": np.array(cat.SEXO)[sample.sexo.to_numpy()],
            "edad": sample.edad.to_numpy().astype(float),
            "grupo_edad": np.array(cat.GRUPOS_EDAD)[sample.grupo.to_numpy()],
            "nivel_estudios": np.array(cat.NIVEL_ESTUDIOS)[sample.edu.to_numpy()],
            "ccaa": np.array(cat.CCAA)[sample.ccaa.to_numpy()],
            "tamuni": np.where(sample.tamuni >= 0, np.array(cat.TAMUNI)[sample.tamuni.clip(0)], None),
            "ingresos_hogar_tramo": np.array(cat.INGRESOS)[tramo],
            "nacionalidad": np.array(_NATIONALITY)[sample.nac.to_numpy()],
            "situacion_laboral": sample.laboral.to_numpy(),
            "version_poblacion": version,
        }
    )
    population.to_parquet(out_dir / "synthetic_population.parquet", index=False)
    metadata = {
        "population_version": version,
        "synthetic": False,
        "coverage": "Población adulta (18+) residente en España; Censo 2021 reponderado a Cifras de Población "
        f"a {period}. No distingue hogares privados de establecimientos colectivos.",
        "variable_provenance": {
            "person_id": "synthetic",
            "peso": "derived",
            "sexo": "observed",
            "edad": "observed",
            "grupo_edad": "observed",
            "nivel_estudios": "observed",
            "ccaa": "observed",
            "tamuni": "derived",
            "ingresos_hogar_tramo": "imputed",
            "nacionalidad": "observed",
            "situacion_laboral": "observed",
            "version_poblacion": "synthetic",
        },
        "categories": {
            "sexo": cat.SEXO,
            "grupo_edad": cat.GRUPOS_EDAD,
            "nivel_estudios": cat.NIVEL_ESTUDIOS + cat.NIVEL_ESTUDIOS_EXTRA,
            "ccaa": cat.CCAA,
            "tamuni": cat.TAMUNI,
            "ingresos_hogar_tramo": cat.INGRESOS + cat.INGRESOS_EXTRA,
            "nacionalidad": _NATIONALITY,
            "situacion_laboral": LABOR_CATEGORIES,
        },
        "sources": {
            "semilla": "INE Censo de Población 2021, microdatos de personas (muestra), adultos",
            "objetivos_demograficos": f"INE Estadística Continua de Población (Cifras de Población) {period}",
            "estudios": f"INE EPA, fichero {epa} (25 años o más)",
            "ingresos": f"INE ECV {ecv}: ingresos del hogar del año anterior / 12, imputados por contracción jerárquica",
            "tamuni": "Censo 2021: códigos 991-993 y tamaño estimado de municipio = muestra x factor de elevación",
        },
        "method": {
            "seed": seed_value,
            "n": n,
            "raking": "IPF sobre CCAA x sexo x grupo, CCAA x nacionalidad, nacionalidad x sexo x grupo, "
            "estudios x sexo x grupo (EPA) y estudios x CCAA (EPA)",
            "sampling": "muestreo sistemático con probabilidad proporcional al peso; peso uniforme",
            "education_mapping": "CNED-A: 01-02 sin_estudios, 03 primaria, 04 secundaria_1, 05 secundaria_2, "
            "06-07 fp, 08-12 superiores (FP superior en superiores, como en la EPA)",
            "limitations": [
                "situacion_laboral procede del Censo 2021 sin reponderar",
                "Los ingresos son imputados (ECV 2025, año de ingresos 2024) y no incluyen N.S./N.C.",
                "El tamaño de municipio mayor de 10.000 habitantes es una estimación",
            ],
        },
        "adult_population_total": adult_total,
        "ipf": ipf_report,
    }
    (out_dir / "population_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = validate_population(population, target, epa_frame, ecv_adults)
    (out_dir / "population_validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"version": version, "n": n, "adult_population": adult_total, "ipf": ipf_report, "validation": validation["summary"]}


def validate_population(population, target, epa, ecv):
    """Compara los márgenes de la muestra con los objetivos y resume la distribución."""

    def shares(frame, column, order):
        s = frame.groupby(column).peso.sum().reindex(order).fillna(0)
        return (s / s.sum()).round(5)

    total = population.peso.sum()
    ccaa_idx = {c: i for i, c in enumerate(cat.CCAA)}
    synthetic = np.zeros((19, 2, 7))
    p_c = population.ccaa.map(ccaa_idx).to_numpy()
    p_s = (population.sexo == "mujer").to_numpy().astype(int)
    p_g = population.grupo_edad.map({g: i for i, g in enumerate(cat.GRUPOS_EDAD)}).to_numpy()
    np.add.at(synthetic, (p_c, p_s, p_g), population.peso.to_numpy())
    goal = target.sum(axis=2)
    cell_error = np.abs(synthetic / total - goal / goal.sum())
    epa_shares = epa.groupby("edu").w.sum() / epa.w.sum()
    adult25 = population[population.grupo_edad != "18-24"]
    ecv_share = np.bincount(ecv.tramo, weights=ecv.w, minlength=6) / ecv.w.sum()
    return {
        "summary": {
            "max_abs_error_ccaa_sexo_grupo_share": float(cell_error.max()),
            "mean_abs_error_ccaa_sexo_grupo_share": float(cell_error.mean()),
            "total_weight": float(total),
        },
        "sexo": shares(population, "sexo", cat.SEXO).to_dict(),
        "grupo_edad": shares(population, "grupo_edad", cat.GRUPOS_EDAD).to_dict(),
        "nivel_estudios": shares(population, "nivel_estudios", cat.NIVEL_ESTUDIOS).to_dict(),
        "nivel_estudios_epa_25_mas": {cat.NIVEL_ESTUDIOS[int(k)]: round(float(v), 5) for k, v in epa_shares.items()},
        "nivel_estudios_sintetica_25_mas": shares(adult25, "nivel_estudios", cat.NIVEL_ESTUDIOS).to_dict(),
        "ccaa": shares(population, "ccaa", cat.CCAA).to_dict(),
        "tamuni": shares(population.dropna(subset=["tamuni"]), "tamuni", cat.TAMUNI).to_dict(),
        "ingresos_hogar_tramo": shares(population, "ingresos_hogar_tramo", cat.INGRESOS).to_dict(),
        "ingresos_ecv_adultos": {cat.INGRESOS[i]: round(float(v), 5) for i, v in enumerate(ecv_share)},
        "nacionalidad": shares(population, "nacionalidad", _NATIONALITY).to_dict(),
        "situacion_laboral": shares(population, "situacion_laboral", LABOR_CATEGORIES).to_dict(),
    }
