"""``terraspectra-pipeline`` command-line interface."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Annotated

import typer

from terraspectra_pipeline import __version__
from terraspectra_pipeline.config import get_settings
from terraspectra_pipeline.logging import configure_logging

app = typer.Typer(
    name="terraspectra-pipeline",
    help="Turn raw hyperspectral scenes into C1 cubes and spectral-index layers.",
    no_args_is_help=True,
    add_completion=False,
)

SensorOpt = Annotated[
    str | None, typer.Option("--sensor", "-s", help="hyperion|enmap|prisma|generic (auto)")
]


def _version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main_callback(
    log_level: Annotated[str | None, typer.Option(help="DEBUG|INFO|WARNING|ERROR")] = None,
    json_logs: Annotated[bool | None, typer.Option("--json-logs/--plain-logs")] = None,
    version: Annotated[bool, typer.Option("--version", callback=_version, is_eager=True)] = False,
) -> None:
    """Global options."""
    s = get_settings()
    configure_logging(log_level or s.log_level, s.log_json if json_logs is None else json_logs)


@app.command()
def ingest(
    source: Annotated[str, typer.Argument(help="Archive/dir path, or 'usgs' to search+download")],
    dest: Annotated[Path | None, typer.Option(help="Raw data dir (default: $DATA/raw)")] = None,
    sensor: SensorOpt = None,
    bbox: Annotated[str | None, typer.Option(help="usgs: lon_min,lat_min,lon_max,lat_max")] = None,
    start: Annotated[str | None, typer.Option(help="usgs: YYYY-MM-DD")] = None,
    end: Annotated[str | None, typer.Option(help="usgs: YYYY-MM-DD")] = None,
    max_results: Annotated[int, typer.Option(help="usgs: max scenes")] = 5,
    download: Annotated[bool, typer.Option(help="usgs: download hits (else list)")] = False,
) -> None:
    """Register a local scene archive, or search/download Hyperion scenes from USGS M2M."""
    from terraspectra_pipeline.ingest import USGSClient, register_scene

    raw_dir = dest or get_settings().raw_dir
    if source != "usgs":
        record = register_scene(source, raw_dir, sensor)
        typer.echo(json.dumps({"scene_dir": record.scene_dir, "sensor": record.sensor}))
        return
    if not (bbox and start and end):
        raise typer.BadParameter("usgs ingest needs --bbox, --start and --end")
    west, south, east, north = (float(v) for v in bbox.split(","))
    with USGSClient() as client:
        hits = client.scene_search(
            (west, south, east, north),
            date.fromisoformat(start),
            date.fromisoformat(end),
            max_results=max_results,
        )
        for hit in hits:
            typer.echo(f"{hit.entity_id}\t{hit.display_id}\t{hit.acquired}")
        if download and hits:
            for path in client.iter_download([h.entity_id for h in hits], raw_dir):
                register_scene(path, raw_dir, "hyperion")
                typer.echo(f"downloaded {path}")


@app.command()
def process(
    scene: Annotated[Path, typer.Argument(exists=True, help="Scene dir or file")],
    out: Annotated[Path, typer.Argument(help="Output C1 COG path")],
    sensor: SensorOpt = None,
    aoi: Annotated[Path | None, typer.Option(exists=True, help="AOI GeoJSON (EPSG:4326)")] = None,
    stats: Annotated[
        Path | None, typer.Option(exists=True, help="Apply p2-p98 scaling from stats.json")
    ] = None,
    dst_crs: Annotated[
        str | None, typer.Option(help="Target CRS (default: scene UTM zone)")
    ] = None,
    resolution: Annotated[float | None, typer.Option(help="Target pixel size (m)")] = None,
    block_size: Annotated[int | None, typer.Option(help="Processing block (px)")] = None,
    workers: Annotated[int | None, typer.Option(help="Worker processes")] = None,
    smooth: Annotated[bool, typer.Option(help="Savitzky-Golay spectral smoothing")] = False,
    cloud_mask: Annotated[bool, typer.Option(help="Heuristic cloud/shadow mask")] = True,
    dark_object: Annotated[bool, typer.Option(help="Dark-object subtraction")] = False,
    source: Annotated[str | None, typer.Option(help="Override the C1 source tag")] = None,
    zarr: Annotated[Path | None, typer.Option(help="Also export to this Zarr store")] = None,
) -> None:
    """Process a raw scene into a C1-compliant COG."""
    from terraspectra_pipeline.normalize import BandStats
    from terraspectra_pipeline.sensors import open_scene
    from terraspectra_pipeline.workflow import ProcessOptions, process_scene

    s = get_settings()
    opts = ProcessOptions(
        block_size=block_size or s.block_size,
        dst_crs=dst_crs,
        resolution=resolution,
        cloud_mask=cloud_mask,
        smooth=smooth,
        dark_object=dark_object,
        stats=BandStats.load(stats) if stats else None,
        workers=workers or s.workers,
        source=source,
    )
    with open_scene(scene, sensor) as reader:
        result = process_scene(reader, out, aoi=aoi, options=opts)
    if zarr is not None:
        from terraspectra_pipeline.writer import export_zarr

        export_zarr(result.path, zarr)
    typer.echo(
        json.dumps(
            {
                "path": str(result.path),
                "size": [result.width, result.height],
                "crs": result.crs,
                "valid_fraction": round(result.valid_fraction, 4),
                "seconds": round(result.seconds, 2),
            }
        )
    )


@app.command()
def indices(
    cube: Annotated[Path, typer.Argument(exists=True, help="C1 cube")],
    out: Annotated[Path, typer.Argument(help="Output indices GeoTIFF")],
    names: Annotated[str | None, typer.Option(help="Comma list, default all")] = None,
) -> None:
    """Compute NDVI, NDRE, REP, PRI, CCI, MCARI and NDWI into a multi-band GeoTIFF."""
    from terraspectra_pipeline.indices import INDEX_NAMES, indices_from_cube

    selected = tuple(n.strip() for n in names.split(",")) if names else INDEX_NAMES
    unknown = set(selected) - set(INDEX_NAMES)
    if unknown:
        raise typer.BadParameter(f"unknown indices {sorted(unknown)}; choose from {INDEX_NAMES}")
    indices_from_cube(cube, out, selected, get_settings().block_size)
    typer.echo(str(out))


@app.command()
def stats(
    cube: Annotated[Path, typer.Argument(exists=True, help="C1 cube")],
    out: Annotated[Path, typer.Argument(help="Output stats.json")],
    windows: Annotated[int, typer.Option(help="Random windows to sample")] = 32,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Compute per-band p2/p98 normalisation statistics."""
    from terraspectra_pipeline.normalize import compute_file_stats

    compute_file_stats(cube, n_windows=windows, seed=seed).save(out)
    typer.echo(str(out))


@app.command()
def validate(
    files: Annotated[list[Path], typer.Argument(exists=True, help="Cubes to check")],
    skip_values: Annotated[bool, typer.Option(help="Skip the value-range sample")] = False,
) -> None:
    """Validate cubes against contract C1 (exit code 1 on violations)."""
    from terraspectra_pipeline.validate import validate_cube

    failed = False
    for f in files:
        problems = validate_cube(f, check_values=not skip_values)
        if problems:
            failed = True
            typer.echo(f"FAIL {f}")
            for p in problems:
                typer.echo(f"  - {p}")
        else:
            typer.echo(f"OK   {f}")
    if failed:
        raise typer.Exit(code=1)


@app.command()
def synthetic(
    out: Annotated[Path, typer.Argument(help="Output COG path")],
    size: Annotated[int, typer.Option(help="Height and width (px)")] = 256,
    seed: Annotated[int, typer.Option()] = 0,
) -> None:
    """Write a synthetic C1 cube (contracts fixture)."""
    from terraspectra_contracts.fixtures import write_synthetic_cube

    out.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(str(write_synthetic_cube(out, size, size, seed)))


@app.command()
def info(
    scene: Annotated[Path, typer.Argument(exists=True, help="Scene dir or file")],
    sensor: SensorOpt = None,
) -> None:
    """Print scene metadata as JSON."""
    from terraspectra_pipeline.sensors import open_scene

    with open_scene(scene, sensor) as reader:
        typer.echo(json.dumps(reader.metadata.summary(), indent=2, default=str))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
