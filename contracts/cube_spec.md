# C1 — Hyperspectral Cube Format (v1.0.0)

Every processed scene handed between modules **must** satisfy this spec. `pipeline/` produces it; `model/` and `api/` consume it.

| Property | Value |
|---|---|
| Container | Cloud-Optimized GeoTIFF (`driver=COG`), ZSTD compression, 256 px blocks, internal overviews |
| Data type | `float32` |
| Values | Surface reflectance in `[0, 1]` |
| Shape | `(B, H, W)` with **B = 200** |
| Band grid | Canonical 400–2500 nm grid in [`wavelengths.json`](wavelengths.json) (~10.55 nm step). Sensors are spectrally resampled onto it; water-absorption bands (~1350–1450, ~1800–1950 nm) are interpolated, never dropped |
| Band tags | Each band has the tag `wavelength_nm=<float>` |
| Dataset tags | `contract_version=1.0.0`, `source=<hyperion|enmap|prisma|synthetic>`, optional `acquired_at=<ISO-8601>` |
| Nodata | `-1.0` (clouds, shadows, outside the AOI, sensor gaps) |
| CRS | Projected UTM (the EPSG code is stored in the file); ground sampling distance is sensor-native (typically 30 m) |

Validation: `terraspectra-pipeline validate <file>` (in `pipeline/`).
Fixture: `terraspectra_contracts.fixtures.write_synthetic_cube()`.
