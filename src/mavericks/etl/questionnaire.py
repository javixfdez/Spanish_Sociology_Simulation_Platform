"""Extracción del enunciado literal de las preguntas desde el cuestionario del CIS (cuesNNNN.pdf)."""

import io
import re
import zipfile

_CODE = re.compile(
    r"^\s*(?:P\.?\s?\d+[A-Za-z]?(?:[_.]\d+)*|[A-Z]{1,3}\.?\s?\d+[A-Za-z]?(?:[_.]\d+)*)\.?\s+(?=\S)"
)
_TAG = re.compile(r"^\s*\[([A-Za-z0-9_]+)\]\s*(\(|$)")
_FILTER = re.compile(r"^\s*(?:Filtros?|Controles?)\s*:", re.I)
_CONDITION = re.compile(r"^\s*\S+\s+(?:si|SI|Si)\s+\S+")
_DIRECTIVE = re.compile(
    r"\((?:ENTREVISTADOR/A|ENTREVISTADOR|LEER|NO LEER|MOSTRAR|SI |ANOTAR|SONDEAR|RESPUESTA)[^)]*\)\s*", re.I
)
_PAGE = re.compile(r"^(Estudio:|Clave:|\d+ de \d+\s*$|«Información sujeta)")
MAX_LENGTH = 700


def read_pdf_text(zip_path):
    """Texto del cuestionario de un zip del CIS (admite PDF directo o un zip anidado). '' si no hay."""
    from pypdf import PdfReader

    with zipfile.ZipFile(zip_path) as z:
        members = [n for n in z.namelist() if n.lower().startswith("cues")]
        if not members:
            return ""
        data, name = z.read(members[0]), members[0]
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as inner:
                pdfs = sorted((i for i in inner.infolist() if i.filename.lower().endswith(".pdf")), key=lambda i: -i.file_size)
                if not pdfs:
                    return ""
                data = inner.read(pdfs[0].filename)
        try:
            return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)
        except Exception:
            return ""


def clean_stem(lines):
    text = " ".join(x.strip() for x in lines if x.strip())
    text = re.sub(r"</?span[^>]*>?", "", text)
    text = _DIRECTIVE.sub("", text)
    text = re.sub(r"\.{3,}.*$", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:MAX_LENGTH].rstrip()


def extract_stems(text):
    """{variable: enunciado}. El enunciado va desde el código de pregunta hasta la primera etiqueta [VARIABLE]."""
    stems, current, collecting, buffer = {}, None, False, []
    for line in text.splitlines():
        if _PAGE.match(line.strip()):
            continue
        tag = _TAG.match(line)
        if tag:
            if collecting:
                current = clean_stem(buffer)
                collecting, buffer = False, []
            if current:
                stems.setdefault(tag.group(1), current)
            continue
        if _FILTER.match(line) or (_CODE.match(line) and _CONDITION.match(line)):
            collecting, buffer = False, []
            continue
        if _CODE.match(line):
            collecting, buffer = True, [_CODE.sub("", line, count=1)]
        elif collecting:
            buffer.append(line)
    return {k: v for k, v in stems.items() if len(v) >= 8}


def stem_for(variable, stems):
    """Enunciado de una variable; las baterías y multirrespuestas se etiquetan sin sufijo ([P6] para P6_1...)."""
    name = variable
    while True:
        if name in stems:
            return stems[name]
        if "_" not in name:
            return None
        name = name.rsplit("_", 1)[0]


def compose_text(label, variables, stems, shared):
    """Enunciado literal de una pregunta. En baterías (varias variables comparten enunciado) se añade el ítem."""
    stem = next((s for s in (stem_for(v, stems) for v in variables) if s), None)
    if not stem:
        return None
    if shared.get(stem, 0) > 1 and len(variables) == 1 and label and label.lower() not in stem.lower():
        return f"{stem} [{label}]"
    return stem
