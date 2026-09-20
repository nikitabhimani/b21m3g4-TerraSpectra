# TerraSpectra — 15-Day Execution Plan (4 People, Minimal Dependency)

## Context
TerraSpectra forecasts crop disease (e.g. fungal blight) weeks before visible symptoms by analysing hyperspectral satellite cubes (200+ bands) with a 3D-CNN + Vision Transformer hybrid, served by a FastAPI inference service and visualised on a React + Deck.gl 3D map. The repo at `/home/vishal-prajapati/projects/TerraSpectra` is empty. The goal is a 15-day plan that splits the work across 4 people so that each person can work **in parallel with no blocking dependencies**.

**How dependencies are removed:** On Day 1, all four people agree on and freeze a set of **contracts** (data format, model input/output, API spec, output GeoJSON). After that, each person works only against the contracts, using **their own mock or synthetic data**. Nobody waits on anyone else until the planned integration on Days 13–15.

On approval, this plan will be saved into the repo as `PROJECT_PLAN.md`.

---

## 0. Assumptions
- Team: 1 geospatial/data engineer (P1), 1 ML engineer (P2), 1 backend engineer (P3), 1 frontend engineer (P4).
- Hardware: at least one CUDA GPU for P2 (training) and P3 (inference testing). Everyone else can use a CPU.
- Data sources (all free):
  - **EnMAP** (DLR) and **PRISMA** (ASI): current hyperspectral missions with about 230 bands.
  - **NASA EO-1 Hyperion** archive (242 bands) via USGS EarthExplorer.
  - Sentinel-2 is multispectral (13 bands), not hyperspectral. We use it only as an optional RGB/NDVI context layer.
  - Public labelled benchmarks for model training: Indian Pines, Salinas, Pavia University.
- **Labels, honestly:** no public dataset has ground-truth "disease 3 weeks before symptoms" labels. P2 therefore builds training data from two sources:
  - Physics-based simulation: the PROSAIL radiative-transfer model with chlorophyll, water and structure progressively degraded, which mimics early fungal stress.
  - Proxy labels from pre-visual stress indices: red-edge position, PRI, CCI, NDRE and the water-band index.
  - The "days to onset" output is a calibrated demo estimate, not a validated agronomic forecast.

---

## 1. Day-1 Contracts (frozen, stored in `contracts/`)

| # | Contract | Content |
|---|---|---|
| C1 | **Cube format** (`contracts/cube_spec.md`) | Cloud-Optimized GeoTIFF, `float32` surface reflectance in [0, 1], shape `(B, H, W)` with **B = 200** bands resampled to a canonical wavelength grid (400–2500 nm, listed in `contracts/wavelengths.json`). Water-absorption bands are filled by interpolation. `nodata = -1`. CRS is UTM (the EPSG code is stored in the file). A band-metadata tag `wavelength_nm` is required. |
| C2 | **Model I/O** (`contracts/model_spec.md`) | Input `float32[N, 200, 64, 64]` → output `float32[N, 4, 64, 64]` (softmax probabilities) plus `float32[N, 1, 64, 64]` (days_to_onset, 0–30). The 4 classes are 0 = healthy, 1 = early stress (pre-visual), 2 = high blight risk, 3 = visible disease. The model is exported as TorchScript at `model.pt` with the entry point `forward(x) -> (probs, onset)`. |
| C3 | **API spec** (`contracts/openapi.yaml`) | `POST /v1/scenes` (register/upload a cube) → `scene_id`. `POST /v1/jobs {scene_id, aoi: GeoJSON}` → `job_id`. `GET /v1/jobs/{id}` → `{status, progress}`. `GET /v1/jobs/{id}/zones` → GeoJSON. `GET /v1/jobs/{id}/tiles/{z}/{x}/{y}.png` → heatmap tile. `GET /v1/jobs/{id}/risk.tif`. `GET /v1/fields`. `GET /v1/health`. |
| C4 | **Zones GeoJSON** (`contracts/zones.schema.json`) | A FeatureCollection of Polygons in EPSG:4326. Properties: `zone_id`, `risk_class`, `risk_score` (0–1), `area_acres`, `days_to_onset`, `dominant_indicator`, `recommended_action`. |
| C5 | **Shared fixtures** (`contracts/fixtures/`) | Written together on Day 1: `synthetic_cube.py` (generates a C1-valid random cube), `stub_model.py` (a TorchScript model with random weights that matches C2), `sample_zones.geojson` (a 1,000-acre farm with a 5-acre red zone), and `sample_job.json`. |
| C6 | **Repo layout** | A monorepo with one folder per owner: `pipeline/` (P1), `model/` (P2), `api/` (P3), `dashboard/` (P4), `contracts/` (all four; changes need agreement from everyone). Each person works on their own branch prefix (`p1/*` … `p4/*`). |

**Rule:** a contract change needs a 5-minute agreement at stand-up. After Day 1, nobody edits another person's folder.

**Daily rhythm:**
- 15-minute stand-up.
- Each person merges to `main` behind their own folder, with CI running lint and tests only for that folder.

---

## 2. Person-wise / Day-wise Plan

### P1 — Geospatial Data Pipeline (Python, Rasterio, GDAL) → `pipeline/`
**Goal:** turn raw, multi-GB Hyperion, EnMAP or PRISMA scenes into C1-compliant, tiled, normalised cubes and spectral-index layers.

| Day | Tasks | Deliverable |
|---|---|---|
| 1 | Contract workshop (C1–C6). Set up the repo skeleton, `pyproject`, pre-commit and GDAL/Rasterio in Docker. | Frozen contracts, Docker image |
| 2 | Write downloaders for Hyperion (USGS M2M API) and EnMAP/PRISMA (manual download plus a loader). Write a metadata parser for wavelengths, FWHM, gains and scale factors. | `pipeline/ingest/` and one raw scene on disk |
| 3 | Write the readers: Hyperion L1 GeoTIFF bundle, EnMAP L2A, PRISMA HE5 (h5py). Read them windowed and lazily, never loading the full cube into RAM. | `read_cube()` returning a lazy windowed iterator |
| 4 | Radiometric step: DN → radiance → TOA reflectance. Add a hook for atmospheric correction (use the L2A product when available, otherwise a simple dark-object subtraction). | `reflectance.py` with unit tests |
| 5 | Band cleaning: remove bad and water-absorption bands, then spectrally resample to the canonical 200-band grid (Gaussian SRF, FWHM-aware). | Output matches C1 |
| 6 | Georeferencing: reproject to UTM (`gdalwarp` / `rasterio.warp`), clip to a farm AOI polygon, build the nodata mask and a cloud/shadow mask. | `clip_to_aoi()` and masks |
| 7 | Normalisation: per-band robust scaling (p2–p98), plus Savitzky–Golay spectral smoothing to reduce noise. Save the scaling stats to JSON. | `normalize.py`, `stats.json` |
| 8 | Write COGs with tiling, internal overviews and ZSTD compression. Add an optional Zarr export for chunked access. Benchmark on a scene larger than 2 GB. | COG writer and benchmark report |
| 9 | Spectral indices layer: NDVI, NDRE, red-edge position (REP), PRI, CCI, MCARI, NDWI. Save as a multi-band GeoTIFF. | `indices.py` and `indices.tif` |
| 10 | Chunking utility: split a cube into 64×64 windows with configurable overlap, plus the reverse "stitch" function with feathered blending. Pure NumPy, so P3 can vendor it later without waiting. | `pipeline/chunking.py` with tests |
| 11 | CLI with Typer: `terraspectra-pipeline ingest|process|index|tile`. Parallelise with Dask or multiprocessing. Add logging. | Working CLI |
| 12 | Hardening: validate outputs against C1 with a validator script, handle edge cases (scene edges, all-nodata windows), add pytest coverage above 70%. | `validate_cube.py`, green CI |
| 13 | **Integration:** run the real pipeline on 2–3 real scenes (farm AOI) and hand the COGs to P3's API. | Real processed cubes |
| 14 | Integration fixes. Produce a Sentinel-2 RGB/terrain context basemap for P4 (optional). | Basemap tiles |
| 15 | Documentation (`pipeline/README.md`, data-flow diagram), demo rehearsal. | Docs and demo |

### P2 — 3D-CNN + ViT Hybrid Model (PyTorch) → `model/`
**Goal:** a model that meets the C2 contract. It is trained on public benchmarks plus PROSAIL-simulated stress, so it does not need P1's pipeline output.

| Day | Tasks | Deliverable |
|---|---|---|
| 1 | Contract workshop. Write `contracts/fixtures/stub_model.py` together with the team. Set up the environment (PyTorch, Lightning, W&B/MLflow). | Env and stub model |
| 2 | Download Indian Pines, Salinas and Pavia U. Resample each to the 200-band canonical grid (own small utility, following C1). Build a Dataset/DataLoader for 64×64 windows. | `model/data/` loaders |
| 3 | Synthetic stress generator: PROSAIL (`prosail` pip package) sweeps of Cab↓, Cw↓, LAI↓ over time steps t = 0…30 days. Map the outputs to classes 0–3 and a `days_to_onset` target. Blend the synthetic spectra into real vegetation pixels. | `synth_stress.py` and a synthetic dataset |
| 4 | Proxy-label generator from the spectral indices (REP shift, PRI, CCI thresholds) for the unlabelled real scenes. | `proxy_labels.py` |
| 5 | Model v1, the **3D-CNN spectral-spatial encoder**: Conv3d blocks (kernel 7×3×3 → 5×3×3 → 3×3×3), BatchNorm, GELU, spectral pooling. | `model/arch/cnn3d.py` |
| 6 | Model v1, the **ViT head**: tokenise spectral groups and patches, add positional embeddings (spectral plus spatial), 4–6 transformer layers. Add the segmentation decoder (upsampling) and two heads: class and onset regression. | `model/arch/hybrid.py`; forward pass matches C2 |
| 7 | Training loop: focal loss for classes plus Huber loss for onset, mixed precision, cosine LR schedule, augmentation (flips, rotations, spectral jitter, band dropout). | `train.py` with a first run |
| 8 | First full training run and evaluation: OA, mIoU, per-class F1, onset MAE. Fix any issues found. | Baseline metrics report |
| 9 | Ablations: 3D-CNN only vs ViT only vs hybrid. Compare band-count sensitivity. | Ablation table |
| 10 | Early-detection evaluation: accuracy versus days-before-symptom curve on synthetic time series. Probability calibration (temperature scaling). | Lead-time curve |
| 11 | Explainability: per-band attention and integrated-gradient importance, used to fill `dominant_indicator` (e.g. "red-edge shift"). | `explain.py` |
| 12 | Export: TorchScript and ONNX, FP16. Verify numerical parity with the eager model. Benchmark throughput (windows/sec) on the GPU. | `model.pt`, `model.onnx`, benchmark |
| 13 | **Integration:** hand `model.pt` to P3 as a drop-in replacement for the stub. Check outputs on P1's real cubes. | Real model served by the API |
| 14 | Fine-tune or recalibrate thresholds based on the real-scene outputs. Write the model card. | `MODEL_CARD.md` |
| 15 | Documentation, training reproducibility script, demo rehearsal. | Docs and demo |

### P3 — Inference API (FastAPI, GPU) → `api/`
**Goal:** a high-throughput service that chunks huge rasters, batches windows on the GPU and returns risk rasters, zones and tiles. It is built against `stub_model.py` and `synthetic_cube.py`, so it does not wait on P1 or P2.

| Day | Tasks | Deliverable |
|---|---|---|
| 1 | Contract workshop. Draft `contracts/openapi.yaml` and write `synthetic_cube.py` together with the team. | OpenAPI spec |
| 2 | FastAPI skeleton: settings (pydantic-settings), logging, `/v1/health`, Dockerfile using a CUDA base image, docker-compose (API, Redis, MinIO). | Running container |
| 3 | Scene registry: `POST /v1/scenes` (multipart upload or S3/MinIO path), metadata stored in SQLite/Postgres (SQLModel), C1 validation on upload. | Scenes endpoints |
| 4 | Job system: `POST /v1/jobs` and `GET /v1/jobs/{id}`, using an RQ or Celery worker on Redis with progress reporting. | Async job lifecycle |
| 5 | Chunked reader: windowed Rasterio reads of 64×64 windows with overlap, via its own implementation (P1's version can be swapped in on Day 13 if it proves better). Streaming generator with a bounded memory footprint. | `api/core/chunker.py` |
| 6 | GPU inference engine: load a TorchScript model (the stub for now), dynamic batching, pinned memory, CUDA streams, AMP. Fall back to CPU when no GPU is present. | `api/core/engine.py` |
| 7 | Stitching: overlap blending into a full-size `(4, H, W)` risk raster plus an onset raster. Write a COG to `risk.tif`. | `GET /jobs/{id}/risk.tif` |
| 8 | Zone extraction: threshold → morphological cleanup → `rasterio.features.shapes` → polygons. Compute acres and mean scores, add `recommended_action` rules, reproject to EPSG:4326. Output matches C4. | `GET /jobs/{id}/zones` |
| 9 | Heatmap tiles: XYZ PNG tiles from the risk raster using rio-tiler, with a colormap (green → red) and caching. | `GET /jobs/{id}/tiles/...` |
| 10 | Performance: benchmark a synthetic 5 GB cube, tune batch size, add multiprocessing for reads, report throughput (km²/min). | Perf report |
| 11 | `GET /v1/fields` (farm AOIs), CORS, API-key auth, rate limiting, request validation, error model. | Hardened API |
| 12 | Tests: pytest with httpx for all endpoints, contract test against `openapi.yaml` (schemathesis), load test with locust. | Green CI and load report |
| 13 | **Integration:** swap the stub for P2's `model.pt` and run jobs on P1's real COGs. | Real end-to-end job |
| 14 | Serve P4's dashboard against the live API. Fix CORS, tile and latency issues. | Live dashboard backend |
| 15 | Deployment docs (docker-compose up), API README, demo rehearsal. | Docs and demo |

### P4 — GIS Dashboard (React, Deck.gl) → `dashboard/`
**Goal:** a 3D analyst dashboard showing risk heatmaps on terrain. It is built against a mock API (MSW) generated from `openapi.yaml` and `sample_zones.geojson`, so it does not wait on P3.

| Day | Tasks | Deliverable |
|---|---|---|
| 1 | Contract workshop. Write `sample_zones.geojson` together with the team (1,000-acre farm with a 5-acre red zone). | Fixtures |
| 2 | Scaffold: Vite, React, TypeScript, Tailwind, React Query, Zustand. Generate a typed API client from `openapi.yaml` (openapi-typescript). Set up MSW mock handlers. | Running app with mock API |
| 3 | Base map: Deck.gl with a MapLibre basemap (satellite style, free tiles). Add `TerrainLayer` for 3D topography (AWS Terrarium elevation tiles). Camera controls. | 3D map |
| 4 | Farm layer: field boundary `GeoJsonLayer` from `/v1/fields`, field selector sidebar, zoom-to-field. | Field navigation |
| 5 | Risk zones layer: `GeoJsonLayer` coloured by `risk_class`, extruded by `risk_score`, with hover tooltips (acres, days to onset, indicator). | Red-zone highlight |
| 6 | Heatmap layer: `TileLayer` + `BitmapLayer` from the `/tiles` endpoint (MSW serves static sample PNGs). Opacity slider and layer toggles. | Heatmap overlay |
| 7 | Zone detail panel: risk score, onset countdown ("~21 days before visible symptoms"), recommended action, spectral indicator chart (Recharts, mock spectral curve). | Detail panel |
| 8 | Job workflow UI: upload or select a scene → create a job → progress bar (polling `GET /jobs/{id}`) → results load automatically. | End-to-end UX on mocks |
| 9 | Analytics: KPI tiles (acres at risk, zones by class), risk distribution chart, sortable zone table linked to the map. | Analytics view |
| 10 | Time slider for the forecast horizon (0–30 days), filtering zones by `days_to_onset`. Export zones as GeoJSON/CSV and a spray-plan PDF. | Forecast slider and export |
| 11 | Polish: responsive layout, dark theme, legend, loading and error states, accessibility pass. | Polished UI |
| 12 | Tests: Vitest and React Testing Library for components, one Playwright E2E test against MSW. Performance check (large GeoJSON, layer memoisation). | Green CI |
| 13 | **Integration:** switch the environment from MSW to the live API (`VITE_API_URL`). | Dashboard on the real backend |
| 14 | Fix integration issues (projections, tile URLs, CORS). Record the demo flow. | Stable demo |
| 15 | User guide, screenshots, final demo presentation. | Docs and demo |

---

## 3. Dependency Map (only at planned points)

| When | From → To | What | Fallback if late |
|---|---|---|---|
| Day 1 | Everyone | Contracts and fixtures | None; this day is mandatory |
| Day 13 | P1 → P3 | Real COG cubes | P3 keeps using `synthetic_cube.py` |
| Day 13 | P2 → P3 | `model.pt` | P3 keeps using `stub_model.py` |
| Day 13 | P3 → P4 | Live API URL | P4 keeps using MSW mocks |
| Day 14 | P1 → P4 | Context basemap (optional) | Public satellite basemap |

Between Days 2 and 12, nobody waits on anyone else.

## 4. Milestones
- **Day 1:** contracts frozen.
- **Day 6:** each module has a working "hello" version (pipeline reads a scene, model forward pass works, API returns a job, map shows zones).
- **Day 12:** each module is feature-complete and tested on its own.
- **Day 14:** end-to-end run: raw scene → pipeline → API with the real model → dashboard shows the red 5-acre zone.
- **Day 15:** demo, documentation, retrospective.

## 5. Verification (end-to-end, Day 14–15)
1. `terraspectra-pipeline process --scene <enmap_scene> --aoi farm.geojson` produces a COG, and `validate_cube.py` passes.
2. `docker compose up` in `api/`, then `POST /v1/scenes` and `POST /v1/jobs`. Poll until the status is `done`.
3. `GET /v1/jobs/{id}/zones` validates against `contracts/zones.schema.json`, and tiles render.
4. Open the dashboard at `VITE_API_URL=http://localhost:8000`, select the farm, and confirm the red zone appears with the correct acreage and days-to-onset.
5. Each folder's CI (pytest / vitest / Playwright) is green, and P2's metrics report and P3's throughput report are attached.
