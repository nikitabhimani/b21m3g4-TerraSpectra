"""End-to-end scene processing: raw scene -> C1 COG, one block at a time."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from rasterio.enums import Resampling
from rasterio.windows import Window

from terraspectra_contracts.constants import NODATA
from terraspectra_pipeline import geo, radiometry, spectral
from terraspectra_pipeline.chunking import iter_blocks
from terraspectra_pipeline.logging import get_logger
from terraspectra_pipeline.normalize import BandStats, apply_scaling
from terraspectra_pipeline.sensors.base import Quantity, SceneMetadata, SensorReader
from terraspectra_pipeline.writer import VALID_SOURCES, CubeWriter

log = get_logger(__name__)

ProgressFn = Callable[[int, int], None]


@dataclass
class ProcessOptions:
    """Knobs for :func:`process_scene`."""

    block_size: int = 256
    dst_crs: str | None = None  # None -> UTM zone of the scene centre
    resolution: float | None = None
    resampling: Resampling = Resampling.nearest
    cloud_mask: bool = True
    smooth: bool = False
    savgol_window: int = 7
    savgol_polyorder: int = 2
    dark_object: bool = False
    stats: BandStats | None = None
    workers: int = 1
    source: str | None = None


@dataclass
class ProcessResult:
    path: Path
    width: int
    height: int
    crs: str
    blocks: int
    valid_fraction: float
    seconds: float
    extra: dict[str, float] = field(default_factory=dict)


@dataclass
class BlockProcessor:
    """Picklable per-block transform; safe to ship to worker processes."""

    reader: SensorReader
    src_grid: geo.Grid
    dst_grid: geo.Grid
    resampler: spectral.SpectralResampler
    options: ProcessOptions
    esun: np.ndarray | None = None
    doy: int | None = None
    dark: np.ndarray | None = None
    aoi: list[geo.Geometry] | None = None

    @property
    def meta(self) -> SceneMetadata:
        return self.reader.metadata

    def sensor_values(self, raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw values -> unclipped reflectance in sensor bands, plus an invalid (h, w) mask."""
        m = self.meta
        good = ~self.resampler.src_bad
        invalid = ~np.isfinite(raw[good]).all(axis=0)
        if m.nodata is not None:
            invalid |= (raw[good] == m.nodata).all(axis=0)
        values = radiometry.dn_to_radiance(raw, m.scale, m.offset)
        if m.quantity is Quantity.RADIANCE:
            if m.sun_elevation_deg is None or self.esun is None:
                raise ValueError("radiance product without sun elevation; cannot compute TOA")
            values = radiometry.radiance_to_toa_reflectance(
                values, self.esun, m.sun_elevation_deg, self.doy
            )
        return values, invalid

    def to_reflectance(self, raw: np.ndarray) -> np.ndarray:
        """Raw sensor values (src geometry) -> canonical-grid reflectance with nodata."""
        values, invalid = self.sensor_values(raw)
        if self.dark is not None:
            values = radiometry.dark_object_subtraction(values, self.dark)
        refl = radiometry.clip_reflectance(values, invalid, NODATA)
        out = self.resampler(refl, NODATA)
        invalid = out[0] == NODATA
        if self.options.smooth:
            out = spectral.savgol_smooth(
                out, self.options.savgol_window, self.options.savgol_polyorder
            )
            out = np.clip(out, 0.0, 1.0)
        if self.options.cloud_mask:
            cloud, shadow = geo.cloud_shadow_mask(out, self.resampler.dst_wavelengths)
            invalid |= cloud | shadow
        out[:, invalid] = NODATA
        return out

    def __call__(self, window: Window) -> np.ndarray:
        h, w = int(window.height), int(window.width)
        offset = geo.aligned_offset(self.src_grid, self.dst_grid)
        if offset is not None:
            src_win: Window | None = Window(
                window.col_off + offset[1], window.row_off + offset[0], w, h
            )
        else:
            src_win = geo.source_window_for(window, self.dst_grid, self.src_grid)
        if src_win is None:
            return np.full((self.resampler.dst_wavelengths.size, h, w), NODATA, np.float32)

        block = self.to_reflectance(self.reader.read_window(src_win))
        if offset is None:
            block = geo.reproject_block(
                block,
                self.src_grid.window_transform(src_win),
                self.src_grid.crs,
                self.dst_grid.window_transform(window),
                self.dst_grid.crs,
                (h, w),
                NODATA,
                self.options.resampling,
            )
        if self.aoi:
            inside = geo.aoi_mask(self.aoi, self.dst_grid.window_transform(window), (h, w))
            block[:, ~inside] = NODATA
        if self.options.stats is not None:
            block = apply_scaling(block, self.options.stats, NODATA)
        return block


def _resolve_source(meta: SceneMetadata, override: str | None) -> str:
    for candidate in (override, meta.sensor, meta.extra.get("source")):
        if candidate in VALID_SOURCES:
            return str(candidate)
    raise ValueError(
        f"cannot map sensor {meta.sensor!r} to a C1 source tag; pass source= "
        f"(one of {sorted(VALID_SOURCES)})"
    )


def _estimate_dark(proc: BlockProcessor, n: int = 8) -> np.ndarray | None:
    from terraspectra_pipeline.normalize import sample_windows

    m = proc.meta
    samples = []
    for win in sample_windows(m.height, m.width, size=128, n=n):
        values, invalid = proc.sensor_values(proc.reader.read_window(win))
        samples.append(values.reshape(values.shape[0], -1)[:, ~invalid.ravel()])
    stacked = np.concatenate(samples, axis=1)
    if stacked.shape[1] == 0:
        log.warning("no valid pixels for dark-object estimation; skipping DOS")
        return None
    return radiometry.dark_object_values(stacked)


def build_processor(
    reader: SensorReader, aoi: str | Path | geo.Geometry | None, options: ProcessOptions
) -> BlockProcessor:
    """Plan grids and spectral/radiometric tables for a scene."""
    m = reader.metadata
    if m.crs is None:
        # TODO(Day 6): support swath products (PRISMA geolocation arrays) via GCP warping.
        raise NotImplementedError(f"{m.sensor} scene has no CRS; geolocation warping is TODO")
    src_grid = geo.Grid(m.crs, m.transform, m.width, m.height)
    dst_grid = geo.plan_target_grid(src_grid, options.dst_crs, options.resolution)
    if not geo.is_utm(dst_grid.crs):
        log.warning("target CRS %s is not UTM; output will fail C1 validation", dst_grid.crs)
    geoms = None
    if aoi is not None:
        geoms = geo.transform_geoms(geo.load_aoi(aoi), geo.WGS84, dst_grid.crs)
        dst_grid = geo.crop_grid_to_geoms(dst_grid, geoms)
    esun: np.ndarray | None = None
    doy: int | None = None
    if m.quantity is Quantity.RADIANCE:
        esun = radiometry.solar_irradiance(m.wavelengths_nm, m.fwhm_nm)
        doy = radiometry.day_of_year(m.acquired_at)
        if doy is None:
            log.warning("acquisition date unknown; assuming Earth-Sun distance of 1 AU")
    resampler = spectral.SpectralResampler(m.wavelengths_nm, m.fwhm_nm, m.bad_bands)
    proc = BlockProcessor(reader, src_grid, dst_grid, resampler, options, esun, doy, None, geoms)
    if options.dark_object:
        proc.dark = _estimate_dark(proc)
    return proc


_WORKER: BlockProcessor | None = None


def _init_worker(proc: BlockProcessor) -> None:
    global _WORKER
    _WORKER = proc


def _run_in_worker(window: Window) -> np.ndarray:
    assert _WORKER is not None
    return _WORKER(window)


def _iter_results(
    proc: BlockProcessor, windows: list[Window], workers: int
) -> Iterator[tuple[Window, np.ndarray]]:
    if workers <= 1:
        for win in windows:
            yield win, proc(win)
        return
    # TODO(Day 11): optional dask.distributed backend for multi-node runs.
    ctx = get_context("spawn")
    with ProcessPoolExecutor(
        workers, mp_context=ctx, initializer=_init_worker, initargs=(proc,)
    ) as pool:
        # Bounded look-ahead keeps at most ~2x workers blocks in memory.
        step = workers * 2
        for i in range(0, len(windows), step):
            batch = windows[i : i + step]
            yield from zip(batch, pool.map(_run_in_worker, batch), strict=True)


def process_scene(
    reader: SensorReader,
    out_path: str | Path,
    aoi: str | Path | geo.Geometry | None = None,
    options: ProcessOptions | None = None,
    progress: ProgressFn | None = None,
) -> ProcessResult:
    """Radiometry -> band cleaning -> resampling -> reprojection/AOI -> scaling -> COG."""
    opts = options or ProcessOptions()
    started = time.perf_counter()
    m = reader.metadata
    source = _resolve_source(m, opts.source)
    proc = build_processor(reader, aoi, opts)
    grid = proc.dst_grid
    windows = list(iter_blocks(grid.height, grid.width, opts.block_size))
    log.info(
        "processing %s -> %s (%dx%d, %d blocks, %d workers)",
        m.scene_id,
        out_path,
        grid.width,
        grid.height,
        len(windows),
        opts.workers,
    )
    extra_tags = {"sensor": m.sensor, "normalized": str(opts.stats is not None).lower()}
    if m.scene_id:
        extra_tags["scene_id"] = m.scene_id
    valid = 0
    with CubeWriter(
        out_path,
        grid.crs,
        grid.transform,
        grid.width,
        grid.height,
        source,
        m.acquired_at,
        extra_tags=extra_tags,
    ) as writer:
        for done, (win, block) in enumerate(_iter_results(proc, windows, opts.workers), 1):
            writer.write(block, win)
            valid += int(np.count_nonzero(block[0] != NODATA))
            if progress:
                progress(done, len(windows))
    total = grid.width * grid.height
    result = ProcessResult(
        path=Path(out_path),
        width=grid.width,
        height=grid.height,
        crs=grid.crs.to_string(),
        blocks=len(windows),
        valid_fraction=valid / total if total else 0.0,
        seconds=time.perf_counter() - started,
    )
    log.info(
        "wrote %s in %.1fs (valid %.1f%%)", out_path, result.seconds, 100 * result.valid_fraction
    )
    return result
