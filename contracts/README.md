# contracts/ — Day-1 frozen interfaces

Everything the four modules agree on. **Change only with whole-team agreement at stand-up**, and bump `CONTRACT_VERSION`.

| ID | File | Owner of producer side |
|---|---|---|
| C1 | [cube_spec.md](cube_spec.md), [wavelengths.json](wavelengths.json) | P1 pipeline |
| C2 | [model_spec.md](model_spec.md) | P2 model |
| C3 | [openapi.yaml](openapi.yaml) | P3 api |
| C4 | [zones.schema.json](zones.schema.json) | P3 api |
| C5 | [fixtures/](fixtures/) and `python/` | everyone |

## Python package

`python/` is the installable `terraspectra-contracts` package. It holds the constants, pydantic schemas and fixture generators. Every Python module depends on it by path:

```bash
cd contracts/python && uv sync --all-extras && uv run pytest
uv run python ../fixtures/synthetic_cube.py /tmp/cube.tif
uv run python ../fixtures/stub_model.py /tmp/stub_model.pt
```

`python/src/terraspectra_contracts/wavelengths.json` is a packaged copy of `wavelengths.json`; a test keeps the two copies in sync.
