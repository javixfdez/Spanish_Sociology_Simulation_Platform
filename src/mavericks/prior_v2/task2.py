"""Tarea 2: prior aprendido (e5 congelado + Ridge/MLP). Solo casos dev; escribe outputs/prior_v2/task2_report.md.

Corte temporal por caso: los casos dev se agrupan por mes de trabajo de campo y para cada grupo se entrena un modelo con
las preguntas del almacén anteriores al trabajo de campo MÁS TEMPRANO del grupo (el corte más restrictivo) y de
estudios distintos de los de todos los casos del grupo. Así el estudio evaluado nunca se ve al entrenar (equivale a
dejar fuera cada estudio dev) y nada posterior a un caso entra en su modelo. Los hiperparámetros se eligen con validación
anidada por estudio sobre los errores dev, igual que en la Tarea 1.
"""

import json
import time
from collections import Counter, defaultdict

import numpy as np

from ..analog import load_national_table
from ..evidence import EvidenceRepository
from . import data, knn, learned
from .embcache import EmbCache, clean_label
from .task1 import EVIDENCE, OUT, TABLE, fmt, modes_of_cases

PCA_DIMS = (32, 64)
RIDGE_ALPHAS = (1.0, 10.0, 100.0)
MLP_SHAPES = ((64,), (128, 64))
CANDIDATE = ("mix", 3, 4)


def variants():
    out = []
    for d in PCA_DIMS:
        out += [(f"ridge α={a:g} pca{d}", "ridge", a, d) for a in RIDGE_ALPHAS]
        out += [(f"mlp {shape} pca{d}", "mlp", shape, d) for shape in MLP_SHAPES]
    return out


def train_group(cases, train, emb, variant_list, case_vectors):
    """Entrena todas las variantes con ``train`` y predice los ``cases`` del grupo: {variante: {id caso: distribución}}."""
    q, o, ex, y = learned.build_training_matrix(train, emb)
    uq = np.stack([emb.question(t.evidence_id) for t in train])
    labels = sorted({clean_label(x) for t in train for x in t.options})
    uo = np.stack([emb.option_query(x) for x in labels])
    predictions = defaultdict(dict)
    for d in sorted({v[3] for v in variant_list}):
        featurizer = learned.Featurizer(d)
        x = featurizer.fit(uq, uo, q, o, ex)
        for name, kind, param, dims in variant_list:
            if dims != d:
                continue
            model = learned.make_model(kind, param).fit(x, y)
            for c in cases:
                ov = np.stack([emb.option_query(opt) for opt in c["options"]])
                p = learned.predict_case(featurizer, model, case_vectors[c["evidence_id"]]["passage"], ov, [False] * len(ov))
                predictions[name][c["evidence_id"]] = p
    return predictions


def paired_line(name, reference, errors):
    r = knn.paired(reference, errors)
    p = "n insuficiente" if r["wilcoxon_p"] is None else f"{r['wilcoxon_p']:.3f}"
    return (
        f"| {name} | {fmt(np.mean(errors))} | {100 * r['mean_delta']:+.2f} [{100 * r['ci'][0]:+.2f}, {100 * r['ci'][1]:+.2f}] "
        f"| {r['better']} / {r['worse']} / {r['ties']} | {p} |"
    )


def main():
    t0 = time.time()
    cases = data.load_dev_cases()
    emb = EmbCache(OUT / "emb_cache")
    case_vectors = data.case_vectors(OUT / "emb_cache" / "dev_cases.npz")
    repo = EvidenceRepository(EVIDENCE)
    table = load_national_table(TABLE)
    questions = learned.train_questions(repo, table, emb)
    reserved = data.reserved_ids()
    data.assert_no_reserved([t.evidence_id for t in questions], "entrenamiento", reserved)
    print(f"preguntas de entrenamiento disponibles: {len(questions)}", flush=True)

    variant_list = variants()
    predictions = defaultdict(dict)
    by_month = defaultdict(list)
    for c in cases:
        by_month[c["fieldwork_date"][:7]].append(c)
    group_log = []
    for month, group in sorted(by_month.items()):
        cutoff = min(c["fieldwork_date"][:10] for c in group)
        excluded_studies = {str(c["estudio_id"]) for c in group}
        train = [t for t in questions if t.date < cutoff and t.study not in excluded_studies]
        assert all(t.date < cutoff and t.study not in excluded_studies for t in train), "fuga temporal o de estudio"
        assert not {c["evidence_id"] for c in group} & {t.evidence_id for t in train}, "caso en su propio entrenamiento"
        t1 = time.time()
        for name, per_case in train_group(group, train, emb, variant_list, case_vectors).items():
            predictions[name].update(per_case)
        group_log.append((month, len(group), cutoff, len(train), time.time() - t1))
        print(f"{month}: {len(group)} casos, corte {cutoff}, {len(train)} preguntas de entrenamiento, {time.time() - t1:.0f} s", flush=True)

    collected = {c["evidence_id"]: c for c in data.load(OUT / "dev_collected.pkl")}
    shift = knn.qq_shift(list(collected.values()))
    arrays = [knn.CaseArrays(collected[c["evidence_id"]], shift) for c in cases]
    ids = [a.evidence_id for a in arrays]
    studies = [a.study for a in arrays]
    observed = [a.observed for a in arrays]
    n_opts = np.array([len(a.options) for a in arrays])
    modes = np.array(modes_of_cases(ids))

    def errors_of(pred):
        return np.array([np.mean(np.abs(np.asarray(p) - o)) for p, o in zip(pred, observed, strict=True)])

    uniform = errors_of([np.full(len(o), 1 / len(o)) for o in observed])
    knn_now = [a.prior(*knn.CURRENT) for a in arrays]
    knn_cand = [a.prior(*CANDIDATE) for a in arrays]
    e_now, e_cand = errors_of(knn_now), errors_of(knn_cand)

    names = [v[0] for v in variant_list]
    matrix = np.array([errors_of([predictions[n][i] for i in ids]) for n in names])
    nested = {}
    for kind in ("ridge", "mlp"):
        rows = [k for k, v in enumerate(variant_list) if v[1] == kind]
        held, chosen = knn.nested_grouped(matrix[rows], studies)
        picked = [predictions[names[rows[chosen[i]]]][ids[i]] for i in range(len(ids))]
        nested[kind] = {"held": held, "chosen": [names[rows[j]] for j in chosen], "pred": picked}
    best_kind = min(nested, key=lambda k: nested[k]["held"].mean())
    average = [0.5 * np.asarray(a) + 0.5 * np.asarray(b) for a, b in zip(knn_cand, nested[best_kind]["pred"], strict=True)]
    e_avg = errors_of(average)

    rows = [
        ("Reparto uniforme", uniform), ("KNN actual (léxica k=5 pot=4)", e_now), ("KNN candidata (mezcla k=3 pot=4)", e_cand),
        ("Ridge (variante elegida con anidada)", nested["ridge"]["held"]),
        ("MLP (variante elegida con anidada)", nested["mlp"]["held"]),
        (f"Media 50/50: candidata + {best_kind}", e_avg),
    ]
    lines = [
        "# Tarea 2 — Prior aprendido (dev, 44 casos)",
        "",
        f"Entrenamiento: {len(questions)} preguntas del CIS (2-{learned.MAX_OPTIONS} opciones); un modelo por mes de trabajo de campo dev "
        f"({len(by_month)} grupos) con el corte más restrictivo del grupo y sin los estudios de sus casos. Variantes: {len(variant_list)} "
        f"(Ridge α∈{RIDGE_ALPHAS}, MLP {MLP_SHAPES}, PCA {PCA_DIMS}); elegidas con validación anidada por estudio.",
        "",
        "| Grupo (mes) | casos | corte | preguntas de entrenamiento | s |", "|---|---|---|---|---|",
        *[f"| {m} | {n} | {c} | {k} | {s:.0f} |" for m, n, c, k, s in group_log],
        "",
        "## Resultado (los modelos aprendidos y la media son fuera de muestra; la candidata KNN se eligió mirando dev)",
        "| Configuración | MAE (pp) | Δ frente al KNN actual [IC 95 %] | mejoran / empeoran / empatan | Wilcoxon p |",
        "|---|---|---|---|---|",
        *[paired_line(n, e_now, e) if n != "Reparto uniforme" else f"| {n} | {fmt(e.mean())} | — | — | — |" for n, e in rows],
        "",
        "### Frente a la candidata KNN (mezcla k=3 pot=4)",
        "| Configuración | MAE (pp) | Δ [IC 95 %] | mejoran / empeoran / empatan | Wilcoxon p |", "|---|---|---|---|---|",
        paired_line("Ridge", e_cand, nested["ridge"]["held"]),
        paired_line("MLP", e_cand, nested["mlp"]["held"]),
        paired_line("Media 50/50", e_cand, e_avg),
        "",
        "## Variante elegida en el bucle interno (por caso)",
        f"Ridge: {dict(Counter(nested['ridge']['chosen']))}",
        f"MLP: {dict(Counter(nested['mlp']['chosen']))}",
        "",
        "## Todas las variantes (MAE medio en dev; optimista, solo descriptivo)",
        "| Variante | MAE (pp) | 2 opciones | 3 o más |", "|---|---|---|---|",
        *[f"| {n} | {fmt(matrix[k].mean())} | {fmt(matrix[k][n_opts == 2].mean())} | {fmt(matrix[k][n_opts >= 3].mean())} |" for k, n in enumerate(names)],
    ]

    def breakdown(title, groups):
        out = ["", f"## {title}", "| Grupo | casos | uniforme | KNN actual | candidata | Ridge | MLP | media 50/50 |", "|---|---|---|---|---|---|---|---|"]
        for label, mask in groups:
            if mask.sum():
                out.append(
                    f"| {label} | {int(mask.sum())} | {fmt(uniform[mask].mean())} | {fmt(e_now[mask].mean())} | {fmt(e_cand[mask].mean())} | "
                    f"{fmt(nested['ridge']['held'][mask].mean())} | {fmt(nested['mlp']['held'][mask].mean())} | {fmt(e_avg[mask].mean())} |"
                )
        return out

    lines += breakdown("Por número de opciones", [("2 opciones", n_opts == 2), ("3 o más", n_opts >= 3)])
    lines += breakdown("Por modo", [("prior", modes == "prior"), ("sin ancla", modes == "none")])
    topics = np.array([a.topic for a in arrays])
    lines += breakdown("Por tema (descriptivo)", [(t, topics == t) for t in sorted(set(topics))])
    lines += ["", f"Tiempo total: {time.time() - t0:.0f} s."]
    text = "\n".join(lines)
    (OUT / "task2_report.md").write_text(text, encoding="utf-8")
    (OUT / "task2_report.json").write_text(
        json.dumps({n: float(matrix[k].mean()) for k, n in enumerate(names)}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(text)


if __name__ == "__main__":
    main()
