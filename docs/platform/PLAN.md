# WildLens — End-to-End Platform Plan

> **How to use this file.** This is the single source of truth for building the
> showcase platform. Planning and review happen in one Claude Code session;
> **each phase below is implemented in its own fresh session.** Start a new
> session, paste the phase's *Kickoff prompt*, and finish by ticking the
> phase's acceptance checklist. Bring the result back to the planning session
> for review before starting the next phase.

_Last updated: 2026-11-14_

---

## 1. Constraints (decided)

| Decision | Value |
|---|---|
| Budget | **₹0 / $0.** No paid services, no credit-card-gated tiers. |
| Model for now | **ResNet50** (`outputs/checkpoints/resnet50_best.pt`). EfficientNetV2-S added later if its weights are found — the code must accept new models via config only. |
| Deadline | None fixed. Quality over speed. |
| Look & feel | Professional, editorial, wildlife/conservation tone. **Not** playful, cartoonish, or emoji-heavy. |
| Run modes | (a) **Local**: one command, works offline (except chat). (b) **Deployed**: public link for reviewers, free hosting. |

## 2. What already exists (reuse, don't rebuild)

- Trained ResNet50 checkpoint — a dict with `model_state`, `optimizer_state`, `epoch`, `best_val_acc`, `hparams` (284 MB; weights alone ≈ 95 MB).
- Model definition: `src/models/resnet50.py` — torchvision ResNet50, `fc` replaced by `nn.Linear(2048, 80)`.
- Preprocessing: resize 224×224 → ToTensor → Normalize with **dataset mean/std** in `src/utils/config.py` (NOT ImageNet defaults).
- Class order: `sorted(unique labels)` from `data/manifest.csv` (see `src/utils/dataset.py`) — 80 classes, Bear … Zebra.
- Metrics & figures for all 4 models: `outputs/results/`, `outputs/figures/` (19 PNGs).
- Write-ups: `docs/paper_summary.md`, `docs/model_comparison.md`, `docs/dataset_notes.md`.
- Processed images (local only, 823 MB) in `data/processed/` — source for a curated sample gallery.

## 3. Product

**WildLens** — identify an animal from a photo, see *why* the model decided, explore
where the species lives, and ask questions about any animal in the world.

### Pages

| Route | Purpose |
|---|---|
| `/` **Home** | Editorial, scroll-driven story: why automated species identification matters (camera traps, biodiversity monitoring) → the dataset → the pipeline → the models → the result. Ends with a CTA into Identify. |
| `/identify` **Identify** | Upload / drag-drop / camera / curated samples. Top-5 predictions with confidence, class-activation heatmap overlay (opacity slider), inference time, low-confidence warning ("may not be one of the 80 species"). Links to the species page and "Ask about this animal". Model selector appears automatically once ≥2 models are registered. |
| `/species` **Species Atlas** | Grid of all 80 species: photo, common + scientific name, conservation status badge. Search & filter (class: mammal/bird/…, status). |
| `/species/:slug` **Species profile** | Hero photo, summary, taxonomy, IUCN status, key facts, **world occurrence map (GBIF)**, model's per-class accuracy for this species, "Ask about this species". |
| `/ask` **Field Guide (chat)** | Chat about **any** animal (not just the 80). Answers grounded in retrieved sources with citations. Accepts an image-prediction handoff as context. |
| `/research` **Research** | 4-model comparison table, charts, confusion matrices, ROC, training curves, dataset & preprocessing, reference paper. Honest limitations section. |
| `/about` **About** | Team, tech stack, architecture diagram, links to repo. |

## 4. Architecture

```
                ┌──────────────────────────── Frontend (React) ────────────────────────────┐
                │  Pages · Design system · Inference adapter · Chat client                  │
                └──────────────┬──────────────────────────────────────┬────────────────────┘
                               │ /api/*                               │ (deployed mode only)
             LOCAL MODE        ▼                       DEPLOYED MODE  ▼
   ┌────────────────────────────────────┐     ┌─────────────────────────────────────────┐
   │ FastAPI backend (Docker)           │     │ Browser: ONNX Runtime Web runs the       │
   │  • /predict, /explain (ONNX RT)    │     │   same model.onnx (WebGPU/WASM)          │
   │  • /species, /metrics              │     │ Vercel serverless /api/chat, /api/*      │
   │  • /chat (Groq → fallback)         │     │   (Groq proxy + knowledge lookup)        │
   └────────────────────────────────────┘     └─────────────────────────────────────────┘
                 ▲                                             ▲
                 └──── models/resnet50.onnx  ·  data/species_kb.json (shared artifacts) ──┘
```

### Key technical decisions

1. **Export the model to ONNX once** (`scripts/export_onnx.py`). The same `.onnx` file
   is used by the FastAPI backend (onnxruntime, CPU) and by the browser (onnxruntime-web).
   PyTorch is then only needed for the export, which makes the backend image small and fast.
   - Export with **two outputs**: `logits` (1×80) and `features` (1×2048×7×7 from `layer4`).
   - Optional: int8 dynamic quantization for the browser build (~25 MB). Verify top-1
     agreement ≥ 99% vs. FP32 on the test set before shipping it.
   - Export a parity test: PyTorch vs ONNX logits max-abs-diff < 1e-4 on 50 test images.

2. **Explainability without gradients.** ResNet50 ends in `layer4 → global avg pool → Linear`.
   For that architecture **Grad-CAM is mathematically identical to CAM**:
   `heatmap_c = ReLU(Σ_k W_fc[c,k] · A_k)`. So the heatmap is computed from the `features`
   output and the exported `fc` weights (`models/resnet50_fc.json` or `.npy`) — in NumPy on the
   server and in plain JS in the browser. No autograd needed anywhere at runtime.
   (Same holds for EfficientNetV2-S later — its head is also pool → dropout → Linear.)

3. **Preprocessing parity.** Resize to 224×224 (bilinear), scale to [0,1], normalize with
   `[0.46290031, 0.44127762, 0.38908589]` / `[0.26581346, 0.2520165, 0.25977104]`.
   Write a parity test: server preprocessing == torchvision eval transform; browser
   preprocessing produces the same top-5 as the server on 20 sample images.

4. **Model registry.** `models/registry.json` lists models (id, display name, onnx path,
   class list, metrics file, input size, mean/std). Adding EfficientNet later = export + one entry.

5. **Chat: free, grounded, never broken.** Provider chain:
   1. **Groq free tier** (no card; Llama-family model) — fast hosted LLM.
   2. **Ollama** (local mode only, optional) — fully offline LLM.
   3. **No-LLM fallback** — answer assembled from retrieved knowledge (summary + facts + links).
   Every answer is **grounded** (RAG-lite): before calling the LLM, look up the animal via
   Wikipedia search + summary, Wikidata facts, and iNaturalist taxon data; pass those as
   context; show citations. System prompt restricts scope to animals/wildlife and tells the
   model to say "I don't know" rather than invent facts. Stream tokens to the UI.
   Server-side rate limit per IP; API key only in server env, never in the browser.

6. **Knowledge base, pre-built.** `scripts/build_species_kb.py` fetches once for the 80
   classes and writes `data/species_kb.json` (summary, image URL + license/attribution,
   scientific name, taxonomy, IUCN status, GBIF taxonKey, iNaturalist taxon id, per-class
   accuracy from our results). Species pages work offline in local mode.
   Some class labels are not species ("Fish", "Bull", "Moths and butterflies", "Monkey") —
   map them to a sensible representative taxon/article manually in `data/species_overrides.json`.

7. **Distribution map.** Leaflet/MapLibre with a dark/neutral basemap plus GBIF Maps API v2
   occurrence-density tiles: `https://api.gbif.org/v2/map/occurrence/density/{z}/{x}/{y}@1x.png?taxonKey=…`.

### Free data sources (no key)

| Source | Used for |
|---|---|
| Wikipedia REST (`/page/summary`) + search | Summary text, lead image |
| Wikidata | Scientific name, IUCN status, taxonomy, lifespan/mass where present |
| iNaturalist API (`/v1/taxa`) | Common names, photos, observation counts |
| GBIF species match + Maps API v2 | taxonKey, occurrence map tiles |
| Wikimedia Commons | Licensed imagery (store attribution, display it) |

## 5. Tech stack

| Layer | Choice |
|---|---|
| Frontend | React 18 + Vite + TypeScript, Tailwind CSS, shadcn/ui (Radix), Framer Motion (restrained), TanStack Query, React Router, Recharts, Leaflet/MapLibre, onnxruntime-web |
| Backend | Python 3.11, FastAPI, Uvicorn, onnxruntime, Pillow, NumPy, httpx, pydantic-settings, slowapi (rate limit), pytest |
| Chat LLM | Groq free tier (primary), Ollama (local optional), no-LLM fallback |
| Packaging | Docker, docker compose, Makefile (`make dev`, `make up`, `make test`) |
| Deployed | Vercel Hobby (frontend + `/api` serverless functions, free, no card); model file on Hugging Face Hub model repo (free public) or GitHub Releases, fetched by the browser |

**Hosting notes (verified Nov 2026, re-check before Phase 7):**
- Hugging Face *Docker/Gradio* Spaces now require a PRO plan to create → not used.
- Render free web service (512 MB, sleeps after 15 min) may require a card → not used as primary.
- Koyeb now requires a card → not used.
- Hence the deployed build runs inference **in the browser**: zero cost, no cold start, and a
  good talking point ("on-device inference, your photo never leaves the browser").

## 6. Design direction

Reference tone: National Geographic / WWF / natural-history museum sites — editorial, calm, photographic.

- **Typography:** serif display (e.g. Fraunces or Playfair Display) for headings, Inter for UI/body; tabular numerals for metrics.
- **Palette (tokens, light + dark):** deep forest `#0F1F17`, moss `#2F4A3A`, bone/paper `#F4F1EA`, stone grey `#8A8F87`, one accent — ochre/amber `#C8913A`. Status colours only for IUCN badges.
- **Imagery:** large full-bleed photographs, generous whitespace, subtle grain. No illustrations/cartoons/emoji.
- **Motion:** slow fades and parallax on Home only; elsewhere micro-interactions (bar fills, heatmap fade-in). Respect `prefers-reduced-motion`.
- **Data viz:** consistent with the site palette; one chart system (Recharts) for all metrics.
- **Accessibility:** WCAG AA contrast, keyboard navigable, alt text, focus rings.
- **Target devices: laptops/desktops only** (≥ 1280 px wide, Chrome/Edge/Safari/Firefox). No phone/tablet layouts required; below 1024 px show a simple "best viewed on a laptop" notice rather than building mobile layouts.
- **Signature details:** heatmap reveal with a scanning line; "field notes" style species cards (scientific name in italics, coordinates-style metadata); IUCN status scale (LC → EX) as a horizontal indicator.

## 7. Repository layout (target)

```
backend/
  app/main.py            # FastAPI app factory, CORS, routers
  app/config.py          # pydantic-settings (env-driven)
  app/routers/           # predict.py, species.py, metrics.py, chat.py, health.py
  app/services/          # inference.py (onnxruntime), cam.py, knowledge.py, chat/ (providers)
  tests/
  Dockerfile
frontend/
  src/pages/ components/ lib/ (api client, inference adapter, cam.ts) styles/
  api/                   # Vercel serverless functions (deployed mode)
  public/samples/        # curated sample images (licensed / from our dataset)
  Dockerfile
models/                  # registry.json, *.onnx (gitignored), fc weights; download script
scripts/                 # export_onnx.py, build_species_kb.py, make_samples.py
data/species_kb.json, data/species_overrides.json
docker-compose.yml, Makefile, .env.example
src/                     # existing ML code — unchanged
```

## 8. Phases

Each phase = one implementation session. Do them in order.

### Phase 0 — Model artifacts
**Goal:** portable model files + parity tests.
- `scripts/export_onnx.py`: load checkpoint (`model_state`), build ResNet50 with 80-class `fc`, export ONNX with `logits` + `features` outputs; dump `fc` weights/bias; write `models/registry.json` and `models/classes.json`.
- Optional quantized browser variant + agreement check.
- Parity tests (PyTorch vs ONNX; preprocessing).
- Re-evaluate ONNX model on the test split → must reproduce test accuracy 0.7992 (±0.001).
- `.gitignore` `models/*.onnx`; add `scripts/download_models.*` placeholder.

**Accept when:** ONNX accuracy matches; parity tests pass; artifact sizes recorded in this file.

**Results (2026-11-14, reviewed — approved)**
- Artifacts: `resnet50.onnx` 94.6 MB (FP32, opset 17, outputs `logits` N×80 + `features` N×2048×7×7, dynamic batch; gitignored) · `resnet50.int8.onnx` 23.8 MB (gitignored) · `resnet50_fc.npz` 0.66 MB · `resnet50_fc.json` 2.09 MB · `classes.json`, `registry.json`, `resnet50_onnx_eval.json` (all small, committable).
- Accuracy: ONNX FP32 + NumPy/PIL preprocessing on full test split (7,576 imgs) = **0.799234** top-1 / 0.973073 top-5 — identical to reported 0.799234 (diff 0.0).
- Parity: PyTorch vs ONNX max |Δlogit| = 2.1e-5 on 50 test images; NumPy preprocessing == torchvision eval transform (≤1e-6) incl. non-224 inputs; CAM from ONNX `features` + fc == torch Grad-CAM × 49 (exact up to the 1/49 GAP scale). `pytest tests/phase0` → 15 passed.
- Int8 dynamic quantization: test top-1 0.7896, **agreement with FP32 93.3% (< 99%) → marked `usable: false`** in the registry. Per-channel weights gave the same 93.2% on a 500-image probe; signed-int8 weights fail to load in ORT CPU (`ConvInteger` s8 not implemented), so uint8 weights are used. Browser build should use FP32 until a better option (e.g. FP16 or static QDQ quantization with calibration) is evaluated.
- Deviations: added `scripts/preprocessing.py` (torch-free, for backend reuse), `scripts/model_common.py`, `scripts/evaluate_onnx.py`; deps in `scripts/requirements-export.txt`; used existing `venv/` (Python 3.9.6, torch 2.8.0) rather than 3.11.

### Phase 1 — Backend core
**Goal:** FastAPI service for prediction & explanation.
- Endpoints: `GET /api/health`, `GET /api/models`, `POST /api/predict` (multipart image, `model` param → top-5, latency ms, low-confidence flag), `POST /api/explain` (returns heatmap PNG or base64 + overlay), `GET /api/metrics` (serves `outputs/results/*`), `GET /api/figures/{name}`.
- Validation: content type, max 10 MB, decode errors → 422 with clear message; EXIF rotation handled.
- Model loaded once at startup; thread-safe session.
- OpenAPI docs at `/docs` with examples.
- pytest suite (happy path, bad file, large file, unknown model).

**Accept when:** `pytest` green; `/docs` works; p50 predict < 150 ms on laptop CPU.

**Results (2026-11-14, reviewed — approved)**
- `backend/` FastAPI app (`app/main.py` factory + lifespan, `config.py` with `WILDLENS_*` env, routers health/models/predict(+explain)/metrics(+figures), services registry/inference/cam/preprocessing/images/metrics). Python 3.11 venv in `backend/.venv` (gitignored); deps pinned in `backend/requirements.txt`, also verified on Python 3.9.6.
- Preprocessing moved to `backend/app/services/preprocessing.py`; `scripts/preprocessing.py` is now a re-export shim, so Phase-0 scripts/tests are unchanged except the no-torch AST check, which now points at the backend file (+1 test asserting the shim re-exports the same functions).
- Tests: `cd backend && .venv/bin/python -m pytest` → **51 passed** (3.11 and 3.9); model/dataset-dependent tests skip with a reason if `models/resnet50.onnx` or `data/processed` is missing. `pytest tests/phase0` → 16 passed. API top-10 equals a direct Phase-0 ONNX run (|Δp| < 1e-5); explain grid equals the Phase-0 CAM formula.
- Latency (Apple M5, warm, 30 requests, intra-op threads = 4): `/api/predict` round-trip **p50 26.8 ms / p95 27.7 ms** on a 224×224 test crop and p50 35.2 / p95 38.8 ms on a 1600×1067 JPEG; `/api/explain` p50 37 ms (224 px) / 98 ms (1600 px, PNG encoding dominates).
- Low-confidence threshold 0.5 (top-1 prob): on the test split it flags 11.7% of images, which are right only 41% of the time (85.1% for the rest) — see `backend/README.md`.
- Deviations: 415 for decodable-but-unsupported formats (e.g. GIF); explain renders at upload size with longest side capped at 1024 px and also returns the 7×7 grid; `/api/metrics` adds `raw` file contents with absolute local paths reduced to file names.

### Phase 2 — Knowledge base & chat
**Goal:** species data + grounded free chat.
- `scripts/build_species_kb.py` + `data/species_overrides.json` → `data/species_kb.json` (80 entries, attribution included).
- `GET /api/species`, `GET /api/species/{slug}`.
- `POST /api/chat` (SSE streaming): retrieve → Groq → fallbacks; citations; scope guard; per-IP rate limit; timeout handling.
- Tests with mocked HTTP + mocked LLM.

**Accept when:** all 80 species have summary + image + status (or explicit "not evaluated"); chat works with Groq key, with Groq key removed (fallback), and offline (graceful message).

**Results (2026-11-15, reviewed — approved)**
- KB: `data/species_overrides.json` (manual label → Wikipedia article/taxon, `rank`, `is_group`, `rank_note`; optional `gbif_name`/`gbif_taxon_key`/`inat_taxon_id` pins) → `scripts/build_species_kb.py` → `data/species_kb.json` (80 entries in `classes.json` order, 219 KB). Raw responses cached in `data/.kb_cache/` (gitignored, 12 MB); reruns are byte-identical. 53 labels are groups, 9 domestic forms (Bull, Canary, Cattle, Chicken, Goat, Goldfish, Horse, Pig, Sheep), 1 hybrid (Mule), 17 species.
- Coverage: summary 80/80 · image with licence + author + source page 80/80 (all Wikimedia Commons) · scientific name 80/80 · IUCN 80/80 (17 species with a code: 16 via Wikidata P141, Giraffe via iNaturalist; 63 explicit "Not applicable (group / domesticated form / hybrid)") · iNaturalist id + observations 80/80 · GBIF taxonKey 78/80 (Fish: Actinopterygii not in the GBIF backbone; Mule: hybrid). Groups whose rank GBIF lacks use the nearest backbone taxon for the map, noted in `gbif_taxon.note` (e.g. Otter → Mustelidae, Squid → Cephalopoda).
- Backend: `GET /api/species` (`q`, `group`, `status` incl. `NA`), `GET /api/species/{slug}`, `POST /api/chat` (SSE `meta` → `token`* → `done`|`error`). Retrieval: context → KB names → live Wikipedia search + Wikidata P225/P141 + iNaturalist (animal check). Providers: Groq (`llama-3.3-70b-versatile`, from console.groq.com/docs/models on 2026-11-14; one retry on 429/5xx) → Ollama (only if `WILDLENS_OLLAMA_URL`) → retrieval-only answer. Per-IP limit 20/min, 64 KB body, 20 messages × 2000 chars, 45 s total budget. Generic User-Agent on all outbound calls; key only from env / `backend/.env`.
- Tests: backend **90 passed** (51 existing + 39 new; all chat HTTP mocked with `httpx.MockTransport`), `pytest tests/phase0` 16 passed. Manual: no-key fallback verified live for a KB species (red panda, 3 ms) and a non-KB animal (axolotl, ~4.5 s live lookup); off-topic → scope message; 21st request in a minute → 429.
- Not verified live: Groq and Ollama (no `GROQ_API_KEY`, no `ollama` on this machine) — paths covered by mocked tests only.
- Deviations: predict/explain link to species via a new top-level `slugs` `{label: slug}` field (a per-prediction `slug` would break the existing exact-key test); per-class accuracy comes from `resnet50_results.json` (the evaluation JSON has none); own in-memory rate limiter instead of slowapi; `GROQ_API_KEY` accepted as well as `WILDLENS_GROQ_API_KEY`.

### Phase 3 — Frontend foundation & design system
**Goal:** the look, before features.
- Vite + TS + Tailwind + shadcn setup; tokens (light/dark); typography; layout shell (nav, footer); reusable components (Section, StatTile, SpeciesCard, ConfidenceBar, StatusScale, Citation, EmptyState, ErrorState, Skeletons).
- A component preview page (`/__ui`, dev-only) for review.

**Accept when:** reviewed screenshots of the preview page in light & dark at 1280 px and 1920 px.

**Results (2026-11-15, reviewed — approved)**
- `frontend/`: React 18.3 + Vite 8.3 + TypeScript 5.9 (strict, `noUncheckedIndexedAccess`) + Tailwind CSS **v3.4** + shadcn/ui-style Radix primitives (hand-written, `components.json` kept for the CLI) + Framer Motion 14 (`MotionConfig reducedMotion="user"`) + React Router 7 + TanStack Query 5 + lucide-react + Recharts 3. ESLint 9 flat config, Prettier, `@/` alias, `/api` proxy → `:8000` in dev and preview. Node 22 (nvm), npm.
- Tokens: RGB-channel CSS variables on `:root` / `[data-theme="dark"]` / `prefers-color-scheme` (unless light chosen), wired into Tailwind; full scale incl. muted success/warning/danger, a separate 3:1 input-border token and an IUCN palette (LC…NE + N/A) with AA code-on-chip contrast in both themes. Contrast table in `frontend/README.md` (all text pairs ≥ 4.5:1; stone is decorative only).
- Type: Fraunces (variable, `opsz`) display/headings + Inter UI/body + JetBrains Mono labels, all self-hosted via `@fontsource-variable/*`; scale display→caption/overline/mono, `tabular` utility, `ScientificName` (italic, upright above genus).
- Shell: sticky header (typographic wordmark, 5 links, theme toggle), footer with data-source credits + repo link, skip link, section rhythm, placeholders for every section-3 route, 404, and a "designed for laptop and desktop" notice below 1024 px.
- Components (all on `/__ui`): Section/SectionHeader, StatTile/StatRow, SpeciesCard, IucnBadge, IucnScale, ConfidenceBar, PredictionList (low-confidence state, `slugs` links), HeatmapViewer (colourises `heatmap_png` client-side in the palette, opacity slider, toggle, scanning-line reveal), UploadDropzone (drag/click/paste/keyboard, type/size validation), ChatMessage/ChatComposer/CitationList/ProviderBadge (Markdown, `[n]` chips, streaming caret, "Searching sources…", "Retrieval-only answer"), Attribution, Button/Input/Select/Tabs/Tooltip/Dialog/Toast/Slider/Switch, EmptyState/ErrorState/Skeleton/Spinner, AccuracyBarChart (Recharts) + table view. `src/lib/api.ts`: types for every backend schema, fetch helpers (`explain` defaults to `amber`), incremental SSE parser + `streamChat` generator.
- Checks: `npm run typecheck`, `lint`, `format:check`, `build` pass; Vitest **12 passed** (IucnScale 3, PredictionList 3, SSE parser/event mapping 6). Production JS 379 KB (125 KB gzip), CSS 58 KB (13 KB gzip); `/__ui`, fixtures, Recharts and react-markdown are not in the production bundle.
- Screenshots (`npm run screenshots`): `/`, `/__ui`, `/identify` at 1280×800 and 1920×1080, light and dark, plus 404 and the narrow-screen notice.
- Deviations: fixtures copied to `frontend/src/fixtures/` (10 KB species subset, one captured `/api/explain` with `overlay_png` dropped, metrics summary, a CC0 red-panda photo); species-card photos hotlink Wikimedia Commons thumbnails (need network); Playwright is a devDependency; `npm audit` reports 7 moderate advisories in dev-only lint tooling (typescript-eslint chain).

### Phase 4 — Core pages
**Goal:** Identify, Species Atlas, Species profile, Field Guide chat.
- Inference adapter interface: `ServerInference` (calls `/api/predict`) and `BrowserInference` (onnxruntime-web + `cam.ts`), selected by env/build flag.
- Identify flow incl. heatmap overlay, samples, low-confidence state, handoff to species page & chat.
- Atlas with search/filter; profile with GBIF map.
- Chat UI with streaming, citations, suggested prompts, context from prediction.

**Accept when:** end-to-end demo works locally; same 20 sample images give identical top-5 in server and browser modes.

### Phase 5 — Story & research pages
**Goal:** explain the project.
- Home narrative (problem → data → pipeline → models → results → try it).
- Research page from `/api/metrics` + figures; honest limitations (dataset imbalance, crop-trained model vs full photos, hard classes).
- About page with architecture diagram.

**Accept when:** a reviewer unfamiliar with the project can explain what it does after reading Home.

### Phase 6 — Packaging & quality
- Dockerfiles (multi-stage, small images), `docker-compose.yml`, `Makefile`, `.env.example`.
- Lighthouse ≥ 90 (performance, accessibility, best practices) on Home and Identify.
- Error boundaries, 404 page, loading skeletons, favicon/OG image.
- README rewrite: screenshots, architecture, how to run (local + deployed).

**Accept when:** fresh clone → `docker compose up` → working app with no manual steps besides `.env`.

### Phase 7 — Free deployment
- Upload `model.onnx` to a public Hugging Face model repo (or GitHub Release); confirm CORS from the deployed origin.
- Vercel: frontend build in browser-inference mode + `/api/chat`, `/api/species` functions; Groq key as Vercel env var.
- Smoke-test checklist on laptop browsers (Chrome + one other); record a 3-minute backup demo video.

**Accept when:** public URL works on a different laptop/network; chat works; no secrets in client bundle.

### Phase 8 — Presentation kit
- Demo script (5 min), architecture slide, likely Q&A (why ResNet50 in prod, why CAM == Grad-CAM here, why browser inference, limitations, future work: EfficientNetV2-S, more species, detection stage from the paper).

## 9. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Groq free tier limits/terms change | Provider chain with no-LLM fallback; provider is one config value. |
| Model trained on tight crops, users upload full scenes | Upload tips, optional client-side crop tool, low-confidence warning, explained on Research page. |
| Non-animal / out-of-scope uploads | Confidence threshold + message; never claim certainty. |
| Large model download in browser | Quantized variant, cache via Cache Storage, progress indicator. |
| Wikipedia/Wikidata image licensing | Store and display attribution from the KB build. |
| Free host policy changes | Local Docker mode is the primary demo; deployment is the bonus. |

## 10. Kickoff prompts (paste into a new session)

**Phase N template**
> Read `docs/platform/PLAN.md` (sections 1–7 and Phase N). Implement Phase N only.
> Do not change files under `src/` or `outputs/` unless the phase says so.
> At the end, run the tests, report what passed/failed, and list anything that deviated from the plan.

## 11. Review log

| Date | Phase | Reviewer notes | Status |
|---|---|---|---|
| 2026-11-14 | Scope | Laptop/desktop only — mobile layouts dropped. | ✅ |
| 2026-11-14 | Plan | Plan drafted and agreed: free-only, ResNet50 first, browser inference for deployment. | ✅ |
| 2026-11-14 | Phase 0 | ONNX FP32 reproduces 0.7992 exactly; parity + CAM tests pass (15); int8 agreement 93.3% → not usable. Reviewer re-ran tests (15 passed), confirmed src/ & outputs/ untouched. Follow-ups: (1) try FP16 export for browser (~47 MB) in Phase 4/7 since FP32 is 94.6 MB; (2) Phase 1 moves `scripts/preprocessing.py` into the backend package (keep tests pointing at it); (3) venv is Py 3.9 — backend Docker image uses 3.11, pin onnxruntime same major. | ✅ approved |
| 2026-11-14 | Phase 1 | FastAPI backend: 51 backend tests + 16 phase0 tests pass; predict p50 27 ms (M5 CPU); preprocessing moved to backend with scripts shim. Reviewer re-ran both suites (51 + 16 pass), checked figure path-traversal (404), no abs paths in /api/metrics, live predict on a Zebra crop (0.999, 27 ms), bad file → 422; src/outputs/models untouched. Follow-ups: (1) Phase 2 chat router reuses the same app-factory/lifespan + deps pattern; (2) frontend should request `amber` colormap to match the palette; (3) keep 0.5 low-confidence threshold (data in backend/README.md). | ✅ approved |
| 2026-11-14 | Follow-ups | Phase 0 (2) preprocessing move and (3) version pinning resolved in Phase 1. Carried forward: FP16 browser model → Phase 4/7; chat router pattern → Phase 2; `amber` colormap → Phase 4. | ✅ |
| 2026-11-15 | Phase 2 | KB 80/80 (summary, licensed image, scientific name, IUCN or explicit N/A, iNat id; GBIF key 78/80), species + SSE chat endpoints, Groq → Ollama → retrieval-only chain; 90 backend + 16 phase0 tests pass. Groq/Ollama only tested with mocks (no key locally). Reviewer re-ran suites (90 backend + 16 phase0 pass); KB spot-check OK (Lion/Tiger/Cheetah species with IUCN; Elephant/Kangaroo/Penguin/Rhinoceros groups with no status); prompt has scope + injection guards; generic User-Agent; backend/.env ignored; src/outputs/models untouched. Follow-ups: (1) live Groq path untested until a free key is added — smoke-test then; (2) frontend shows a "searching sources…" state for non-KB animals (~4.5 s); (3) frontend renders image attribution from KB; (4) uses `slugs` map from predict/explain. | ✅ approved |
| 2026-11-15 | Phase 3 | Frontend foundation: tokens (light/dark, IUCN palette, AA contrast table), Fraunces + Inter + JetBrains Mono self-hosted, layout shell + placeholders + 404 + small-screen notice, component library and `/__ui` preview, typed API client + SSE parser; typecheck/lint/build pass, 12 Vitest tests pass; screenshots at 1280/1920 light/dark. Reviewer: typecheck/lint/build pass; 12 Vitest tests pass on Node 22 (fail to start on Node 20 → added `.nvmrc`=22, engines >=22); reviewed Home + /__ui screenshots light/dark at 1280/1920 — editorial tone as intended; fixed footer repo URL to new repo. Follow-ups: (1) Phase 4 adds a local thumbnail cache for species images so local mode works offline; (2) watch bundle size when Leaflet + onnxruntime-web land (lazy-load per route); (3) remove "arrives with Phase 5" placeholder copy in Phase 5. | ✅ approved |
