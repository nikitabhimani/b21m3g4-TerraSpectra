import json
from pathlib import Path

import numpy as np

from terraspectra_contracts import N_BANDS, WAVELENGTHS_NM, ZoneFeatureCollection
from terraspectra_contracts.fixtures import make_synthetic_cube

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def test_wavelengths():
    assert len(WAVELENGTHS_NM) == N_BANDS
    assert WAVELENGTHS_NM[0] == 400.0 and WAVELENGTHS_NM[-1] == 2500.0


def test_synthetic_cube_shape_and_range():
    cube, mask = make_synthetic_cube(64, 64)
    assert cube.shape == (N_BANDS, 64, 64) and cube.dtype == np.float32
    valid = cube[cube != -1.0]
    assert valid.min() >= 0.0 and valid.max() <= 1.0
    assert mask.any()


def test_sample_zones_match_schema():
    data = json.loads((FIXTURES / "sample_zones.geojson").read_text())
    fc = ZoneFeatureCollection.model_validate(data)
    assert any(f.properties.risk_class == 2 for f in fc.features)


def test_packaged_wavelengths_in_sync_with_canonical_copy():
    pkg = Path(__file__).resolve().parents[1] / "src/terraspectra_contracts/wavelengths.json"
    assert json.loads(pkg.read_text()) == json.loads((FIXTURES.parent / "wavelengths.json").read_text())


def test_stub_model_honours_c2(tmp_path):
    import torch

    from terraspectra_contracts.fixtures import export_stub_model

    model = torch.jit.load(str(export_stub_model(tmp_path / "stub.pt")))
    with torch.no_grad():
        probs, onset = model(torch.rand(3, N_BANDS, 64, 64))
    assert probs.shape == (3, 4, 64, 64) and onset.shape == (3, 1, 64, 64)
    assert torch.allclose(probs.sum(1), torch.ones(3, 64, 64), atol=1e-5)
    assert float(onset.min()) >= 0.0 and float(onset.max()) <= 30.0


def test_cog_fixture(tmp_path):
    import rasterio

    from terraspectra_contracts.fixtures import write_synthetic_cube

    with rasterio.open(write_synthetic_cube(tmp_path / "c.tif", 64, 64)) as ds:
        assert ds.count == N_BANDS and ds.nodata == -1.0
        assert float(ds.tags(1)["wavelength_nm"]) == WAVELENGTHS_NM[0]
