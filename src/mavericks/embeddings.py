"""Embebedores multilingües locales para el recuperador (sentence-transformers, opcional)."""

import numpy as np

# Modelos que esperan prefijos de consulta/pasaje (familia E5).
_E5_PREFIXES = ("intfloat/multilingual-e5", "intfloat/e5")


class SentenceTransformerEmbedder:
    """Carga un modelo local (ruta o nombre ya descargado) y codifica consultas y pasajes.

    No descarga nada por sí solo salvo que se le pase un nombre de Hugging Face y exista conexión;
    ``local_files_only=True`` lo impide.
    """

    def __init__(self, model, device="cpu", local_files_only=False, batch_size=32):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model, device=device, local_files_only=local_files_only)
        self.name = f"{model}@{device}"
        self.batch_size = batch_size
        self.e5 = str(model).startswith(_E5_PREFIXES)

    def _encode(self, texts, prefix=""):
        return np.asarray(
            self.model.encode([prefix + t for t in texts], batch_size=self.batch_size, normalize_embeddings=True,
                              show_progress_bar=False)
        )

    def encode_passages(self, texts):
        return self._encode(texts, "passage: " if self.e5 else "")

    def encode_queries(self, texts):
        return self._encode(texts, "query: " if self.e5 else "")
