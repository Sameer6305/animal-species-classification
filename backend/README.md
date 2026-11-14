# WildLens backend (FastAPI)

Prediction and explanation API for the WildLens platform (Phase 1 of
`docs/platform/PLAN.md`). Runs the Phase-0 ONNX ResNet50 on CPU with ONNX Runtime;
no PyTorch at runtime.

## Prerequisites

- Python 3.9+ (3.11 recommended; the Docker image in Phase 6 will use 3.11).
- Model artifacts in `../models/`: `registry.json`, `classes.json`, `resnet50_fc.npz`
  (committed) and `resnet50.onnx` (gitignored, ~95 MB; generate with
  `scripts/export_onnx.py`, see `scripts/download_models.sh`). Without the ONNX file
  the API still starts (`/api/health` reports `degraded`) and the metrics/figures
  endpoints work; `/api/predict` and `/api/explain` return 503.

## Run

```bash
cd backend
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/uvicorn app.main:app --reload            # http://127.0.0.1:8000/docs
```

Tests:

```bash
cd backend
.venv/bin/python -m pytest                         # model-dependent tests skip if models/resnet50.onnx is missing
```

## Endpoints

All routes are under `/api`. Interactive docs: `/docs` (Swagger UI), `/redoc`, schema at `/openapi.json`.

| Method | Path | Description |
|---|---|---|
| GET | `/api/health` | `status` (`ok` / `degraded`), version, default model, per-model load status. |
| GET | `/api/models` | Registered models: id, display name, served variant, classes, input size, loaded, test-metrics summary. |
| POST | `/api/predict` | Multipart: `file` (required), `model` (optional, default from registry), `top_k` (1–10, default 5). Returns top-k `{label, index, probability}`, `low_confidence`, `inference_ms`, image `width/height/format`. |
| POST | `/api/explain` | Multipart: `file`, `model`, `class_index` (default top-1), `colormap` (`jet` \| `amber`). One forward pass; returns base64 `overlay_png` (RGB) and `heatmap_png` (8-bit grayscale) at the uploaded image size (longest side capped at 1024 px), the 7×7 `grid`, label/probability for the explained class and the top-5. |
| GET | `/api/metrics` | All four models' results from `outputs/results/*` in one stable shape keyed by model id: `summary`, `hparams`, `per_class_accuracy`, `per_class_prf`, `history` (column lists), `figures`, `sources`, `raw`. Missing values are `null`. |
| GET | `/api/figures` | List of PNGs in `outputs/figures/` (name, url, owning model, size). |
| GET | `/api/figures/{name}` | One PNG. Only plain `*.png` names that exist in the directory listing are served; anything else is 404. |

Errors: `{"detail": "..."}`. Upload rules:

| Case | Status |
|---|---|
| Body/file larger than `WILDLENS_MAX_UPLOAD_MB` (10 MB) | 413 |
| Empty, undecodable, truncated, or > 50 MP image | 422 |
| Decodable but not JPEG/PNG/WebP (e.g. GIF) | 415 |
| `top_k` outside 1–10, `class_index` outside 0–79, bad `colormap` | 422 |
| Unknown `model` | 404 |
| Model registered but ONNX not loaded | 503 |

The format is checked by decoding, never by the client's content type. EXIF
orientation is applied before inference, and `width/height` are reported after it.

## Configuration

Environment variables (prefix `WILDLENS_`); relative paths resolve against the repo root.

| Variable | Default | Meaning |
|---|---|---|
| `WILDLENS_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated allowed origins. |
| `WILDLENS_MAX_UPLOAD_MB` | `10` | Upload size limit. |
| `WILDLENS_MAX_IMAGE_PIXELS` | `50000000` | Decoded-pixel limit (decompression-bomb guard). |
| `WILDLENS_LOW_CONFIDENCE_THRESHOLD` | `0.5` | `low_confidence` when top-1 probability is below this. |
| `WILDLENS_INTRA_OP_THREADS` | `0` (auto = min(4, CPUs)) | ONNX Runtime threads per request. |
| `WILDLENS_EXPLAIN_MAX_SIDE` | `1024` | Longest side of the explain PNGs. |
| `WILDLENS_OVERLAY_ALPHA` | `0.45` | Heatmap opacity in `overlay_png`. |
| `WILDLENS_REGISTRY_PATH`, `WILDLENS_RESULTS_DIR`, `WILDLENS_FIGURES_DIR` | `models/registry.json`, `outputs/results`, `outputs/figures` | Data locations. |

### Low-confidence threshold

Measured on the full test split (7,576 images, ONNX FP32, overall top-1 0.7992):

| Threshold | Flagged | Top-1 acc. when flagged | Top-1 acc. when not flagged | Share of all errors flagged |
|---|---|---|---|---|
| 0.4 | 6.6% | 36.1% | 83.0% | 20.9% |
| **0.5** | **11.7%** | **41.0%** | **85.1%** | **34.5%** |
| 0.6 | 18.8% | 45.8% | 87.8% | 50.8% |
| 0.7 | 25.9% | 49.6% | 90.5% | 65.0% |

0.5 is the default because a flagged prediction is then more likely wrong than right
(41% correct). Raising it to 0.7 flags predictions that are still right about half
the time and would warn on a quarter of clean test crops. The test images are tight
crops of the 80 classes, so real uploads (full scenes, animals outside the 80
classes) will be flagged more often; that is the case the warning is for. Change it
with `WILDLENS_LOW_CONFIDENCE_THRESHOLD`.

## Layout

```
app/main.py            app factory: CORS, upload-size guard, lifespan loads models once
app/config.py          pydantic-settings (WILDLENS_*)
app/schemas.py         response models (OpenAPI)
app/routers/           health, models, predict (+explain), metrics (+figures)
app/services/          registry, inference (onnxruntime), cam, preprocessing, images, metrics
tests/                 pytest + FastAPI TestClient
```

`app/services/preprocessing.py` is the canonical torch-free preprocessing (moved from
`scripts/preprocessing.py`, which now re-exports it for the Phase-0 scripts and tests).
