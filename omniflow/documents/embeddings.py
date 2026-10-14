"""Local text embeddings behind a small interface.

Only free, locally executed models are allowed. They run on the CPU via ONNX Runtime
(`fastembed`); the model files are downloaded once from Hugging Face into the model
cache directory and no text is ever sent to a remote service.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

import numpy as np

# Allowlist: every entry is free, open-licensed and runs locally.
LOCAL_EMBEDDING_MODELS: dict[str, dict[str, str]] = {
    "BAAI/bge-small-en-v1.5": {
        "license": "MIT",
        "dim": "384",
        "download": "~67 MB (Hugging Face: Qdrant/bge-small-en-v1.5-onnx-Q)",
        "cache_dir": "models--Qdrant--bge-small-en-v1.5-onnx-Q",
    },
    "sentence-transformers/all-MiniLM-L6-v2": {
        "license": "Apache-2.0",
        "dim": "384",
        "download": "~90 MB (Hugging Face: Qdrant/all-MiniLM-L6-v2-onnx)",
        "cache_dir": "models--Qdrant--all-MiniLM-L6-v2-onnx",
    },
}


class Embedder(Protocol):
    model_name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """L2-normalised float32 matrix, one row per text."""

    def embed_query(self, text: str) -> np.ndarray:
        """L2-normalised float32 vector."""


def normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def is_model_cached(model_name: str, cache_dir: Path) -> bool:
    info = LOCAL_EMBEDDING_MODELS.get(model_name)
    return bool(info) and (Path(cache_dir) / info["cache_dir"]).is_dir()


class FastEmbedEmbedder:
    def __init__(self, model_name: str, cache_dir: Path, batch_size: int = 32):
        if model_name not in LOCAL_EMBEDDING_MODELS:
            raise ValueError(f"Embedding model is not on the local allowlist: {model_name}")
        # No telemetry; silence a harmless Windows symlink warning from huggingface_hub.
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from fastembed import TextEmbedding

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        try:  # use the local copy without contacting the network when it exists
            self._model = TextEmbedding(model_name, cache_dir=str(cache_dir), local_files_only=True)
        except Exception:  # noqa: BLE001 - not cached yet: one-time download
            self._model = TextEmbedding(model_name, cache_dir=str(cache_dir))
        self.model_name = model_name
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        return normalise(np.vstack(list(self._model.embed(texts, batch_size=self.batch_size))))

    def embed_query(self, text: str) -> np.ndarray:
        return normalise(np.asarray(next(iter(self._model.query_embed(text)))))


def create_embedder(model_name: str, cache_dir: Path) -> Embedder:
    """The app's single entry point for embedders (tests replace it with a fake)."""
    return FastEmbedEmbedder(model_name, cache_dir)
