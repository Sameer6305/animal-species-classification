"""Runtime settings (env-driven, prefix ``WILDLENS_``).

Relative paths are resolved against the repository root, so the app behaves the
same whether uvicorn is started from ``backend/`` or from the repo root.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WILDLENS_", extra="ignore")

    repo_root: Path = REPO_ROOT
    registry_path: Path = Path("models/registry.json")
    results_dir: Path = Path("outputs/results")
    figures_dir: Path = Path("outputs/figures")

    # Comma-separated list, e.g. WILDLENS_CORS_ORIGINS="http://localhost:5173,https://x.app"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    max_upload_mb: float = Field(10.0, gt=0)
    # Decompression-bomb guard (decoded pixels, not bytes). 50 MP covers any phone photo.
    max_image_pixels: int = Field(50_000_000, gt=0)

    # top-1 softmax probability below this => low_confidence=true (see backend/README.md).
    low_confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)

    # onnxruntime intra-op threads per session run; 0 = auto (min(4, cpu_count)).
    intra_op_threads: int = Field(0, ge=0)

    # /api/explain: overlay + heatmap are rendered at the uploaded image size,
    # downscaled so the longest side is at most this (keeps the JSON payload small).
    explain_max_side: int = Field(1024, ge=224, le=4096)
    overlay_alpha: float = Field(0.45, ge=0.0, le=1.0)

    def resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else (self.repo_root / p)

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def cors_origin_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def effective_intra_op_threads(self) -> int:
        if self.intra_op_threads > 0:
            return self.intra_op_threads
        return max(1, min(4, os.cpu_count() or 1))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

