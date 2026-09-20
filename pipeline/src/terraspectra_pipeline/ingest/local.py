"""Register manually downloaded scene archives (EnMAP, PRISMA, Hyperion) under ``data/raw``."""

from __future__ import annotations

import json
import tarfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from terraspectra_pipeline.logging import get_logger
from terraspectra_pipeline.sensors import detect_sensor

log = get_logger(__name__)

MANIFEST = "scene.json"
ARCHIVE_SUFFIXES = (".zip", ".tar", ".tar.gz", ".tgz")


@dataclass
class SceneRecord:
    scene_dir: str
    sensor: str | None
    source: str
    registered_at: str

    @property
    def path(self) -> Path:
        return Path(self.scene_dir)


def _is_archive(path: Path) -> bool:
    return path.is_file() and path.name.lower().endswith(ARCHIVE_SUFFIXES)


def _stem(path: Path) -> str:
    name = path.name
    for suffix in sorted(ARCHIVE_SUFFIXES, key=len, reverse=True):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def safe_extract(archive: Path, dest: Path) -> None:
    """Extract zip/tar archives, refusing members that escape ``dest``."""
    dest = dest.resolve()
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.namelist():
                if not (dest / member).resolve().is_relative_to(dest):
                    raise ValueError(f"unsafe path in archive: {member}")
            zf.extractall(dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
    else:
        raise ValueError(f"unsupported archive: {archive}")


def _scene_root(folder: Path) -> Path:
    """Descend through single-directory wrappers that archives often add."""
    current = folder
    while True:
        children = [c for c in current.iterdir() if c.name != MANIFEST]
        if len(children) == 1 and children[0].is_dir():
            current = children[0]
        else:
            return current


def register_scene(
    source: str | Path, raw_dir: str | Path, sensor: str | None = None
) -> SceneRecord:
    """Unpack (if needed) ``source`` into ``raw_dir/<name>`` and write a ``scene.json`` manifest."""
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(src)
    raw = Path(raw_dir)
    if _is_archive(src):
        target = raw / _stem(src)
        if not target.exists():
            target.mkdir(parents=True)
            log.info("extracting %s -> %s", src, target)
            safe_extract(src, target)
        scene_dir = _scene_root(target)
    elif src.is_dir():
        scene_dir = src
    else:
        scene_dir = src  # a single-file product (e.g. PRISMA .he5 or a GeoTIFF cube)
    detected = sensor or detect_sensor(scene_dir)
    if detected is None:
        log.warning("could not detect sensor for %s", scene_dir)
    record = SceneRecord(
        scene_dir=str(scene_dir.resolve()),
        sensor=detected,
        source=str(src.resolve()),
        registered_at=datetime.now(UTC).isoformat(),
    )
    manifest_dir = scene_dir if scene_dir.is_dir() else raw
    manifest_dir.mkdir(parents=True, exist_ok=True)
    (manifest_dir / MANIFEST).write_text(json.dumps(asdict(record), indent=2))
    return record
