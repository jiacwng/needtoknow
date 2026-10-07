# Text embeddings with BAAI/bge-small-en-v1.5, run locally on the CPU through fastembed (ONNX).
# The model is downloaded once into NEEDTOKNOW_MODEL_CACHE and loaded once per process.

from functools import cache

import numpy as np
from fastembed import TextEmbedding
from numpy.typing import NDArray

from needtoknow.config import load_settings

MODEL = "BAAI/bge-small-en-v1.5"
DIMENSIONS = 384
# The bge model card gives this instruction for short queries searched against passages.
# fastembed's query_embed does not add it for this model, so it is added here.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

Vector = NDArray[np.float32]


def embed_passages(texts: list[str]) -> list[Vector]:
    return [np.asarray(vector, dtype=np.float32) for vector in _model().passage_embed(texts)]


def embed_query(text: str) -> Vector:
    vector = next(iter(_model().query_embed(QUERY_INSTRUCTION + text)))
    return np.asarray(vector, dtype=np.float32)


@cache
def _model() -> TextEmbedding:
    return TextEmbedding(MODEL, cache_dir=str(load_settings().model_cache))
