"""
Learning script: create a mock GeoTIFF following the team's C1 cube contract,
then read it back with rasterio.

C1 contract (see contracts/cube_spec.md):
- shape (B, H, W) with B = 200 canonical bands
- dtype float32, values in [0, 1]
- nodata = -1.0
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from terraspectra_contracts import N_BANDS, NODATA

# --- Part 1: create a fake GeoTIFF (stand-in for real satellite data) ---
height, width = 64, 64
bands = N_BANDS  # 200, from the shared contract — not hardcoded

data = np.random.rand(bands, height, width).astype(np.float32)  # rasterio wants (bands, H, W)

transform = from_origin(west=75.5, north=26.9, xsize=0.001, ysize=0.001)  # near Jaipur

with rasterio.open(
    "mock_scene.tif", "w",
    driver="GTiff",
    height=height, width=width, count=bands,
    dtype=data.dtype,
    crs="EPSG:4326",
    transform=transform,
    nodata=NODATA,  # from the contract, not a made-up number
) as dst:
    dst.write(data)

print("Created mock_scene.tif")

# --- Part 2: read it back, like you'd read a real satellite file ---
with rasterio.open("mock_scene.tif") as src:
    print("Shape (bands, height, width):", src.shape, src.count)
    print("CRS:", src.crs)
    print("Bounds:", src.bounds)
    print("Nodata value:", src.nodata)

    arr = src.read()  # shape: (bands, H, W)
    print("Read array shape:", arr.shape)

    band_1 = src.read(1)  # single band, shape (H, W)
    print("Band 1 shape:", band_1.shape)