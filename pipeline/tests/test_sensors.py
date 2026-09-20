from pathlib import Path

import numpy as np
import pytest
from rasterio.windows import Window

from terraspectra_contracts.constants import N_BANDS, WAVELENGTHS_NM
from terraspectra_pipeline.sensors import (
    EnmapL2AReader,
    GenericGeoTIFFReader,
    HyperionReader,
    PrismaReader,
    Quantity,
    available_sensors,
    detect_sensor,
    get_reader_class,
    open_scene,
    parse_enmap_metadata,
)
from terraspectra_pipeline.sensors.hyperion import hyperion_bad_bands, hyperion_wavelengths

from .helpers import write_multiband

ENMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<level_X xmlns="http://example.org/enmap">
  <metadata><name>ENMAP01-L2A-TEST</name></metadata>
  <base><temporalCoverage><startTime>2023-06-01T08:30:00.000Z</startTime></temporalCoverage></base>
  <specific>
    <sunElevationAngle><center>55.5</center></sunElevationAngle>
    <bandCharacterisation>
{bands}
    </bandCharacterisation>
  </specific>
</level_X>
"""


def make_enmap_product(root: Path, n: int = 12, size: int = 8) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    wl = np.linspace(420.0, 2440.0, n)
    bands = "\n".join(
        f'<bandID number="{i + 1}"><wavelengthCenterOfBand>{w:.3f}</wavelengthCenterOfBand>'
        f"<FWHMOfBand>8.0</FWHMOfBand><GainOfBand>0.0001</GainOfBand>"
        f"<OffsetOfBand>0</OffsetOfBand></bandID>"
        for i, w in enumerate(wl)
    )
    (root / "ENMAP01-L2A-TEST-METADATA.XML").write_text(ENMAP_XML.format(bands=bands))
    data = np.full((n, size, size), 2500, dtype=np.int16)
    data[:, 0, 0] = -32768
    write_multiband(root / "ENMAP01-L2A-TEST-SPECTRAL_IMAGE.TIF", data, nodata=-32768)
    return root


def make_hyperion_bundle(root: Path, size: int = 6, dn: float = 1000.0) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stem = "EO1H1470392004123110KZ"
    (root / f"{stem}_MTL_L1T.TXT").write_text(
        "GROUP = L1_METADATA_FILE\n"
        "  GROUP = PRODUCT_METADATA\n"
        f'    ENTITY_ID = "{stem}"\n'
        "    ACQUISITION_DATE = 2004-05-02\n"
        '    SCENE_CENTER_SCAN_TIME = "05:10:22.1"\n'
        "  END_GROUP = PRODUCT_METADATA\n"
        "  SUN_ELEVATION = 62.5\n"
        "END_GROUP = L1_METADATA_FILE\nEND\n"
    )
    band = np.full((1, size, size), dn, dtype=np.int16)
    band[:, 0, 0] = 0
    for b in range(1, 243):
        write_multiband(root / f"{stem}_B{b:03d}_L1T.TIF", band, nodata=0)
    return root


def test_registry_lists_all_sensors() -> None:
    assert set(available_sensors()) >= {"generic", "hyperion", "enmap", "prisma"}
    assert get_reader_class("enmap") is EnmapL2AReader
    with pytest.raises(KeyError):
        get_reader_class("landsat")


def test_generic_reader_on_synthetic_cog(synthetic_cog: Path) -> None:
    assert detect_sensor(synthetic_cog) == "generic"
    with open_scene(synthetic_cog) as reader:
        assert isinstance(reader, GenericGeoTIFFReader)
        m = reader.metadata
        assert m.band_count == N_BANDS
        np.testing.assert_allclose(m.wavelengths_nm, WAVELENGTHS_NM, atol=0.01)
        assert m.crs is not None and m.crs.to_epsg() == 32643
        assert m.quantity is Quantity.REFLECTANCE and m.nodata == -1.0
        assert m.extra["source"] == "synthetic"
        blocks = list(reader.iter_blocks(block_size=40))
        assert len(blocks) == 4
        assert sum(b.shape[1] * b.shape[2] for _, b in blocks) == 64 * 64
        assert blocks[0][1].dtype == np.float32 and blocks[0][1].shape[0] == N_BANDS
        assert m.summary()["bands"] == N_BANDS


def test_generic_reader_rejects_untagged(tmp_path: Path) -> None:
    path = write_multiband(tmp_path / "plain.tif", np.zeros((3, 4, 4), np.float32))
    assert not GenericGeoTIFFReader.can_read(path)
    assert detect_sensor(path) is None
    with pytest.raises(ValueError):
        open_scene(path)
    with pytest.raises(ValueError), GenericGeoTIFFReader(path) as reader:
        _ = reader.metadata


def test_enmap_reader(tmp_path: Path) -> None:
    root = make_enmap_product(tmp_path / "enmap")
    meta = parse_enmap_metadata(root / "ENMAP01-L2A-TEST-METADATA.XML")
    assert meta.wavelengths_nm.size == 12 and meta.sun_elevation_deg == 55.5
    assert meta.acquired_at is not None and meta.acquired_at.year == 2023
    assert meta.scene_id == "ENMAP01-L2A-TEST"
    assert detect_sensor(root) == "enmap"
    with open_scene(root) as reader:
        m = reader.metadata
        assert m.nodata == -32768 and m.quantity is Quantity.REFLECTANCE
        np.testing.assert_allclose(m.scale, 1e-4)
        block = reader.read_window(reader_window(2, 2))
        assert block.shape == (12, 2, 2) and block[0, 1, 1] == 2500


def reader_window(h: int, w: int) -> Window:
    return Window(0, 0, w, h)


def test_hyperion_tables() -> None:
    wl, fwhm = hyperion_wavelengths()
    assert wl.size == fwhm.size == 242
    assert wl[0] == pytest.approx(355.59) and wl[-1] == pytest.approx(2577.08)
    bad = hyperion_bad_bands()
    assert bad[:7].all() and not bad[7:57].any() and bad[57:76].all() and bad[224:].all()
    assert int((~bad).sum()) == 50 + 148


def test_hyperion_reader(tmp_path: Path) -> None:
    root = make_hyperion_bundle(tmp_path / "hyp")
    assert detect_sensor(root) == "hyperion"
    with HyperionReader(root) as reader:
        m = reader.metadata
        assert m.quantity is Quantity.RADIANCE
        assert m.sun_elevation_deg == 62.5
        assert m.acquired_at is not None and m.acquired_at.hour == 5
        assert m.scale[0] == pytest.approx(1 / 40) and m.scale[-1] == pytest.approx(1 / 80)
        block = reader.read_window(reader_window(3, 3))
        assert block.shape == (242, 3, 3)
        assert block[0].max() == 0  # bad band skipped
        assert block[10, 1, 1] == 1000 and block[10, 0, 0] == 0


def test_prisma_reader(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "PRS_L2D_STD_20230601_test.he5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields")
        grp["VNIR_Cube"] = np.full((5, 4, 3), 30000, dtype=np.uint16)
        grp["SWIR_Cube"] = np.full((5, 4, 2), 65535, dtype=np.uint16)
        f.attrs["List_Cw_Vnir"] = np.array([700.0, 600.0, 500.0])
        f.attrs["List_Fwhm_Vnir"] = np.array([10.0, 10.0, 10.0])
        f.attrs["List_Cw_Swir"] = np.array([2000.0, 0.0])
        f.attrs["List_Fwhm_Swir"] = np.array([10.0, 0.0])
        f.attrs["L2ScaleVnirMin"] = 0.0
        f.attrs["L2ScaleVnirMax"] = 1.0
        f.attrs["Product_StartTime"] = b"2023-06-01T10:00:00"
        f.attrs["Sun_zenith_angle"] = 30.0
    assert detect_sensor(path) == "prisma"
    with PrismaReader(path) as reader:
        m = reader.metadata
        assert m.wavelengths_nm.tolist()[:4] == [500.0, 600.0, 700.0, 2000.0]
        assert m.bad_bands.tolist() == [False, False, False, False, True]
        assert m.sun_elevation_deg == pytest.approx(60.0)
        assert m.crs is None
        block = reader.read_window(reader_window(2, 3))
        assert block.shape == (5, 2, 3)
