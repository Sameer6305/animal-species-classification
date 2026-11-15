# WildLens backend (FastAPI)

Prediction and explanation API for the WildLens platform (Phases 1-2 of
`docs/platform/PLAN.md`). Runs the Phase-0 ONNX ResNet50 on CPU with ONNX Runtime;
no PyTorch at runtime. Phase 2 adds the species knowledge base (`/api/species`) and the
grounded Field Guide chat (`/api/chat`, Server-Sent Events).

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
                                                   # chat tests mock all HTTP (no network, no key needed)
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
| GET | `/api/species` | The 80 classes in model order: slug, label, common/scientific name, `is_group`, animal group, thumbnail, IUCN code/label, model per-class accuracy. Query: `q` (substring of names/taxonomy), `group` (`Mammal`, `Bird`, `Reptile`, `Amphibian`, `Fish`, `Insect`, `Arachnid`, `Crustacean`, `Mollusc`, `Other`), `status` (IUCN code, or `NA` = no species-level status). |
| GET | `/api/species/{slug}` | Full KB entry (summary, taxonomy, image + licence/attribution, IUCN status + source, GBIF taxonKey, iNaturalist id/observations, model stats). 404 for unknown slugs. |
| GET | `/api/species-images` | Phase 4: slug → URLs of locally cached photos (`hero` ~1280 px, `card` ~500 px) from `scripts/cache_species_images.py`; `{"count": 0, "images": {}}` when the cache is absent (clients fall back to Commons). |
| GET | `/api/species-images/{name}` | One cached photo. Only well-formed `<slug>-hero|card.<ext>` names listed in the cache manifest and present on disk are served; anything else is 404. |
| POST | `/api/chat` | JSON `{"messages": [{"role": "user"\|"assistant", "content": "..."}], "context": {"species_slug" or "label"}}` → `text/event-stream`. See [Chat](#chat-field-guide). |

`GET /api/species?extended=true` (Phase 4, used by the Species Atlas) adds `rank`, `image` (with
attribution), `inaturalist_observations` and `test_images` to each row; without it the row shape is
unchanged.

`/api/predict` and `/api/explain` also return `slugs` (`{label: slug}`) so the frontend can link
each prediction to its species page (additive; existing fields unchanged).

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

Phase 2 (chat and knowledge base):

| Variable | Default | Meaning |
|---|---|---|
| `WILDLENS_GROQ_API_KEY` (or `GROQ_API_KEY`) | unset | Enables Groq as the first chat provider. Server-side only. |
| `WILDLENS_GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq chat model id ([list](https://console.groq.com/docs/models)); `llama-3.1-8b-instant` is faster. |
| `WILDLENS_OLLAMA_URL` | unset (disabled) | e.g. `http://127.0.0.1:11434` to use a local Ollama after Groq. |
| `WILDLENS_OLLAMA_MODEL` | `llama3.2` | Ollama model name (`ollama pull llama3.2`). |
| `WILDLENS_CHAT_RATE_LIMIT_PER_MINUTE` | `20` | Per client IP (in-memory, single process; `0` disables). |
| `WILDLENS_CHAT_MAX_MESSAGES` / `WILDLENS_CHAT_MAX_MESSAGE_CHARS` / `WILDLENS_CHAT_MAX_BODY_KB` | `20` / `2000` / `64` | Request limits (422 / 422 / 413). |
| `WILDLENS_CHAT_TIMEOUT_S` | `45` | Total time budget for one answer (retrieval + generation). |
| `WILDLENS_RETRIEVAL_TIMEOUT_S` | `10` | Live Wikipedia/Wikidata/iNaturalist lookup budget. |
| `WILDLENS_LLM_CONNECT_TIMEOUT_S` / `WILDLENS_LLM_READ_TIMEOUT_S` | `5` / `20` | Per-request LLM timeouts. |
| `WILDLENS_CHAT_LIVE_LOOKUP` | `true` | Look up animals outside the 80 classes online. |
| `WILDLENS_SPECIES_KB_PATH` | `data/species_kb.json` | Knowledge base file. |
| `WILDLENS_SPECIES_IMAGES_DIR` | `data/species_images` | Offline photo cache (Phase 4, gitignored); build with `python3 scripts/cache_species_images.py`. |

Secrets go in `backend/.env` (gitignored; template in `backend/.env.example`) or the process
environment. Keys are never logged or returned by the API.

**Free Groq key (no credit card):** sign up at <https://console.groq.com>, open *API Keys*,
create a key, and put it in `backend/.env` as `WILDLENS_GROQ_API_KEY=...`. Restart the server;
the startup log lists the active chat providers.

## Chat (Field Guide)

`POST /api/chat` answers questions about any animal, grounded in retrieved sources:

1. **Find the animal(s):** the `context` from a prediction (`species_slug` or class `label`), then
   names of the 80 KB classes in the latest message (incl. plurals and a few aliases), then a live
   Wikipedia search for anything else; a hit counts only if its Wikidata item has a taxon name (P225)
   and iNaturalist classifies it as an animal. Follow-ups ("how long do they live?") reuse animals
   named in earlier user turns.
2. **Retrieve:** the KB entry for the 80 classes (works offline); otherwise the live Wikipedia intro,
   Wikidata (IUCN status) and iNaturalist (common name, observations). Questions about the project
   add a "WildLens project facts" source.
3. **Generate:** Groq → Ollama (only if configured) → a retrieval-only answer composed from the
   facts (summary, scientific name, status, habitat sentence, sources), clearly labelled as such.
   Groq gets one retry on 429/5xx; any provider that fails before its first token is skipped.
4. **Stream** (`text/event-stream`), each `data:` line is JSON:

```
event: meta
data: {"provider": "groq", "model": "llama-3.3-70b-versatile", "retrieval": "kb", "subjects": [{"name": "Red panda", "scientific_name": "Ailurus fulgens", "slug": "red-panda"}], "sources": [{"id": 1, "title": "Wikipedia: Red panda", "url": "https://en.wikipedia.org/wiki/Red_panda", "kind": "wikipedia"}, ...], "skipped": []}

event: token
data: {"text": "Red pandas live in "}

event: done
data: {"provider": "groq", "finish_reason": "stop", "elapsed_ms": 812.4}
```

`error` (`{"message"}`) replaces `done` only if a provider fails after it started streaming or the
total time budget runs out mid-answer. The system prompt limits scope to animals/wildlife/biology/
conservation and this project, requires facts from the numbered sources only with `[n]` citations,
and treats source text as data: sources are wrapped in `<sources>/<source>` delimiters, tag-like
text inside them is neutralised, and the model is told to ignore instructions found there.

Example: `curl -N -X POST localhost:8000/api/chat -H 'content-type: application/json' -d '{"messages":[{"role":"user","content":"Where do axolotls live?"}]}'`

Rate limiting is per IP in process memory (`request.client.host`; behind a reverse proxy
configure uvicorn's `--proxy-headers`/`--forwarded-allow-ips` so the real client IP is used).

## Species knowledge base

`data/species_kb.json` is generated by `scripts/build_species_kb.py` from
`data/species_overrides.json` (manual label → Wikipedia article/taxon mapping; many labels
such as "Bear", "Fish", "Monkey", "Moths and butterflies" are groups, a few are domestic forms
or a hybrid). Rebuild:

```bash
backend/.venv/bin/python scripts/build_species_kb.py            # from the repo root; uses data/.kb_cache/
backend/.venv/bin/python scripts/build_species_kb.py --refresh  # refetch everything (~10-30 min, polite rate)
```

Sources: Wikipedia REST summary + lead image, Wikimedia Commons `imageinfo` (author, licence),
Wikidata (P141, P225), iNaturalist `/v1/taxa` (id, ancestry → taxonomy and `animal_group`,
observations, fallback photo), GBIF `species/match` (taxonKey for the Maps API). Per-class
accuracy comes from `outputs/results/resnet50_results.json` (`resnet50_evaluation.json` has no
per-class data) and test image counts from `data/manifest.csv`. Raw responses are cached in
`data/.kb_cache/` (gitignored); `fetched_at` is the fetch time of the cached responses, so reruns
produce an identical file. The script prints per-field coverage and warnings at the end.

**IUCN status precedence:** groups, domestic forms and the hybrid get `code: null` with an
explicit "Not applicable (…)" label (never a species-level status); otherwise Wikidata P141 of
the article's item (only if that item's P225 equals the taxon); otherwise iNaturalist's global
IUCN status. Disagreements are reported by the build script.

**GBIF caveat:** the GBIF backbone has no super-/infra-/sub- ranks, so some groups use the nearest
backbone taxon for the occurrence map (e.g. Otter → Mustelidae); `gbif_taxon.note` says so, and
`Fish` (Actinopterygii is not in the current backbone) and `Mule` (hybrid) have no GBIF key.

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
app/routers/           health, models, predict (+explain), metrics (+figures), species, chat
app/services/          registry, inference (onnxruntime), cam, preprocessing, images, metrics,
                       knowledge (species KB), ratelimit, chat/ (retrieval, prompt, providers, service)
tests/                 pytest + FastAPI TestClient
```

`app/services/preprocessing.py` is the canonical torch-free preprocessing (moved from
`scripts/preprocessing.py`, which now re-exports it for the Phase-0 scripts and tests).
