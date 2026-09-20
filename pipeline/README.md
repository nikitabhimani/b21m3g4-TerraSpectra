# pipeline/ — Geospatial Data Pipeline (P1)

Turns raw hyperspectral scenes (EO-1 Hyperion, EnMAP L2A, PRISMA L2D) into **C1-compliant cubes**:
Cloud-Optimized GeoTIFF, `float32` reflectance in `[0, 1]`, 200 canonical bands (400–2500 nm),
`nodata = -1`, UTM CRS, per-band `wavelength_nm` tags — see
[`contracts/cube_spec.md`](../contracts/cube_spec.md). It also produces the pre-visual stress
index layer (NDVI, NDRE, REP, PRI, CCI, MCARI, NDWI) and the 64×64 chunking utility that P3
can vendor.

Everything is windowed: a multi-GB scene is never loaded into RAM.

## Setup

```bash
cd pipeline
uv sync --all-extras          # extras: prisma (h5py), zarr
uv run pytest -q              # no network, no real scenes: synthetic fixtures only
uv run ruff check . && uv run ruff format --check . && uv run mypy
```

Config comes from env (prefix `TS_PIPELINE_`, plus `USGS_M2M_USERNAME` / `USGS_M2M_TOKEN` from
the repo `.env`): `TS_PIPELINE_DATA_DIR`, `TS_PIPELINE_WORKERS`, `TS_PIPELINE_BLOCK_SIZE`,
`TS_PIPELINE_LOG_LEVEL`, `TS_PIPELINE_LOG_JSON`.

## CLI

```bash
# A C1 cube with a planted stressed patch, to work against without any download
uv run terraspectra-pipeline synthetic data/synthetic_cube.tif --size 512
uv run terraspectra-pipeline validate data/synthetic_cube.tif

# Register a manually downloaded EnMAP/PRISMA archive (unpacks + detects the sensor)
uv run terraspectra-pipeline ingest ~/Downloads/ENMAP01-____L2A-DT000.zip --dest data/raw

# Search / download Hyperion scenes from USGS M2M (needs credentials)
uv run terraspectra-pipeline ingest usgs --bbox 75.0,30.5,75.5,31.0 \
    --start 2004-01-01 --end 2004-12-31 --download

uv run terraspectra-pipeline info data/raw/ENMAP01-____L2A-DT000

# Raw scene -> C1 COG, clipped to a farm AOI, 4 worker processes
uv run terraspectra-pipeline process data/raw/EO1H1470392004123110KZ data/cube.tif \
    --aoi ../contracts/fixtures/sample_fields.geojson --workers 4 --smooth

# Normalisation statistics (p2/p98 per band) and a normalised re-run
uv run terraspectra-pipeline stats data/cube.tif data/stats.json
uv run terraspectra-pipeline process data/raw/<scene> data/cube_norm.tif --stats data/stats.json

uv run terraspectra-pipeline indices data/cube.tif data/indices.tif
uv run terraspectra-pipeline validate data/cube.tif      # exit code 1 on violations
```

Docker (build context is the repo root, service `pipeline` in the root compose file):

```bash
docker compose --profile tools build pipeline
docker compose --profile tools run --rm pipeline synthetic /data/synthetic_cube.tif
```

## Data flow

```mermaid
flowchart LR
  A[USGS M2M / manual download<br/>ingest] --> B[SensorReader<br/>hyperion | enmap | prisma | generic]
  B -->|windowed raw DN| C[radiometry<br/>DN→radiance→TOA reflectance]
  C --> D[spectral<br/>bad/water bands, Gaussian SRF → 200 bands, Savitzky-Golay]
  D --> E[geo<br/>reproject to UTM, AOI clip, cloud/shadow mask]
  E --> F[normalize<br/>p2–p98 robust scaling, optional]
  F --> G[writer<br/>COG: ZSTD, 256 px tiles, overviews, wavelength tags]
  G --> H[validate<br/>contract C1]
  G --> I[indices.tif<br/>NDVI NDRE REP PRI CCI MCARI NDWI]
  G --> J[chunking<br/>64×64 windows + feathered stitching → P3]
```

`workflow.process_scene` drives the chain block by block: for each output window it reads only
the source window it needs, converts it, warps it onto the target UTM grid and writes it, so peak
memory is roughly `block_size² × bands × 4 bytes`.

## Module map

| Module | Contents |
|---|---|
| `config.py` | `PipelineSettings` (pydantic-settings), `get_settings()` |
| `logging.py` | `configure_logging()` (plain or JSON), `get_logger()` |
| `sensors/base.py` | `SensorReader` ABC, `SceneMetadata`, registry, `open_scene()`, `detect_sensor()` |
| `sensors/hyperion.py` | L1 GeoTIFF bundle + MTL, 242-band wavelength table, bad-band list, VNIR/SWIR gains |
| `sensors/enmap.py` | L2A `SPECTRAL_IMAGE.TIF` + `METADATA.XML` parsing (wavelengths, FWHM, gain, sun angle) |
| `sensors/prisma.py` | L2D HE5 via h5py (`prisma` extra); CRS/geolocation still TODO |
| `sensors/generic.py` | Any GeoTIFF with `wavelength_nm` band tags (incl. our own C1 cubes) |
| `ingest/usgs.py` | M2M client: login-token, scene-search, download-options/request, streamed download |
| `ingest/local.py` | Safe archive extraction, sensor detection, `scene.json` manifest |
| `radiometry.py` | Gain/offset, Earth–Sun distance, solar irradiance, TOA reflectance, clipping, DOS hook |
| `spectral.py` | Bad/water-band masks, FWHM-aware Gaussian-SRF resampling, gap interpolation, Savitzky-Golay |
| `geo.py` | UTM grid planning, windowed reprojection, AOI load/clip/mask, nodata + cloud/shadow heuristics |
| `normalize.py` | `BandStats` (JSON), sampled p2/p98 statistics, `apply_scaling()` |
| `writer.py` | `CubeWriter` (block writes → COG), `write_cog()`, `export_zarr()` (`zarr` extra) |
| `indices.py` | Seven indices + multi-band GeoTIFF writer with band descriptions |
| `chunking.py` | `iter_windows()`, `iter_blocks()`, `feather_weights()`, `Stitcher` (pure numpy) |
| `validate.py` | `validate_cube()` → list of human-readable C1 violations |
| `workflow.py` | `process_scene()` orchestration, `ProcessOptions`, multiprocessing |
| `cli.py` | Typer app: `ingest process indices stats validate synthetic info` |

## Day 1–15 checklist

- [x] **Day 1** — contracts C1–C6, `pyproject`, Docker image, module skeleton.
- [x] **Day 2** — USGS M2M client + local archive ingest; metadata parsers (MTL, EnMAP XML, HE5 attrs).
- [x] **Day 3** — readers with lazy windowed reads (Hyperion bundle, EnMAP L2A, PRISMA skeleton, generic).
- [x] **Day 4** — DN → radiance → TOA reflectance; dark-object-subtraction hook.
- [x] **Day 5** — bad/water band removal and FWHM-aware resampling to the canonical 200-band grid.
- [x] **Day 6** — reprojection to the scene's UTM zone, AOI clipping, nodata and cloud/shadow masks.
- [x] **Day 7** — p2–p98 robust scaling with `stats.json`, Savitzky-Golay smoothing.
- [x] **Day 8** — COG writer (ZSTD, 256 px tiles, overviews) and optional Zarr export. _Benchmark on a >2 GB scene still pending real data._
- [x] **Day 9** — `indices.py` and `indices.tif`.
- [x] **Day 10** — `chunking.py`: 64×64 windows, feathered `Stitcher`, tests.
- [x] **Day 11** — Typer CLI, multiprocessing, logging.
- [x] **Day 12** — `validate.py` C1 validator, edge cases (all-nodata windows, scene edges), tests.
- [ ] **Day 13** — integration: run on 2–3 real scenes, hand the COGs to P3.
- [ ] **Day 14** — integration fixes; optional Sentinel-2 RGB context basemap for P4.
- [ ] **Day 15** — docs polish, demo rehearsal.

## Known caveats / TODOs

- Solar irradiance is a 5778 K blackbody approximation — `TODO(Day 4)` to swap in Thuillier/ESUN tables.
- Hyperion wavelengths are the nominal linear VNIR/SWIR tables; per-scene values would be better.
- Cloud/shadow masking is an explicit threshold **heuristic**, not a cloud product.
- PRISMA reads cubes and metadata but has no CRS yet, so `process` refuses it (`TODO(Day 6)`).
- USGS `download-request` does not yet poll `download-retrieve` for queued products.
