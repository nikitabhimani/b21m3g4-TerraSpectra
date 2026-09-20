"""Sensor readers. Importing this package registers every built-in reader."""

from terraspectra_pipeline.sensors.base import (
    Quantity,
    SceneMetadata,
    SensorReader,
    available_sensors,
    detect_sensor,
    get_reader_class,
    open_scene,
    register_reader,
)
from terraspectra_pipeline.sensors.enmap import EnmapL2AReader, parse_enmap_metadata
from terraspectra_pipeline.sensors.generic import GenericGeoTIFFReader
from terraspectra_pipeline.sensors.hyperion import HyperionReader, parse_mtl
from terraspectra_pipeline.sensors.prisma import PrismaReader

__all__ = [
    "EnmapL2AReader",
    "GenericGeoTIFFReader",
    "HyperionReader",
    "PrismaReader",
    "Quantity",
    "SceneMetadata",
    "SensorReader",
    "available_sensors",
    "detect_sensor",
    "get_reader_class",
    "open_scene",
    "parse_enmap_metadata",
    "parse_mtl",
    "register_reader",
]
