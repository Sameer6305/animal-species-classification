"""WildLens API: FastAPI app factory.

Run from backend/:  uvicorn app.main:app --reload
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.config import Settings, get_settings
from app.routers import health, metrics, models, predict
from app.services.inference import ModelManager
from app.services.metrics import load_metrics
from app.services.registry import load_registry

log = logging.getLogger("wildlens")

DESCRIPTION = """
WildLens backend: identify an animal (80 classes) from a photo with an ONNX
ResNet50, explain the prediction with a class activation map, and serve the
project's training/evaluation metrics and figures.

* Inference runs on CPU with ONNX Runtime; the model is loaded once at startup.
* Uploads: JPEG, PNG or WebP, validated by decoding; EXIF orientation applied.
* Errors use `{"detail": "..."}` (validation errors use FastAPI's list form).
"""

UPLOAD_PATHS = ("/api/predict", "/api/explain")
MULTIPART_OVERHEAD = 64 * 1024


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry = load_registry(settings)
        app.state.settings = settings
        app.state.manager = ModelManager(registry, settings.effective_intra_op_threads)
        figure_dir = settings.resolve(settings.figures_dir)
        figure_names = metrics.list_figure_names(figure_dir)
        app.state.metrics = load_metrics(settings.resolve(settings.results_dir), figure_names)
        log.info(
            "WildLens API %s ready: models loaded=%s", __version__, list(app.state.manager.loaded)
        )
        yield

    app = FastAPI(
        title="WildLens API",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        openapi_tags=[
            {"name": "inference", "description": "Classify photos and explain predictions."},
            {"name": "models", "description": "Model registry."},
            {"name": "research", "description": "Metrics and figures from the training runs."},
            {"name": "health", "description": "Liveness/readiness."},
        ],
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def reject_oversized_uploads(request: Request, call_next):
        # Fail fast on a declared oversized body, before multipart parsing spools it.
        # The exact per-file check happens again in the handler (Content-Length can be absent).
        if request.method == "POST" and request.url.path in UPLOAD_PATHS:
            length = request.headers.get("content-length")
            if length and length.isdigit() and int(length) > settings.max_upload_bytes + MULTIPART_OVERHEAD:
                return JSONResponse(
                    status_code=413,
                    content={"detail": f"File too large; the limit is {settings.max_upload_mb:g} MB."},
                )
        return await call_next(request)

    for r in (predict.router, models.router, metrics.router, health.router):
        app.include_router(r, prefix="/api")
    return app


app = create_app()
