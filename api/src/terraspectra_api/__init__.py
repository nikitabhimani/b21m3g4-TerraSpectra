"""TerraSpectra inference API (owner: P3)."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("terraspectra-api")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "1.0.0"

__all__ = ["__version__"]
