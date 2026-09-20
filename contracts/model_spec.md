# C2 — Model I/O (v1.0.0)

| | |
|---|---|
| Artifact | TorchScript file `model.pt` (optional ONNX twin `model.onnx`) |
| Input | `float32[N, 200, 64, 64]`: C1 reflectance windows. Nodata pixels are replaced with `0` by the caller, which keeps its own mask |
| Output | Tuple `(probs, onset)` |
| `probs` | `float32[N, 4, 64, 64]`, softmax over classes (sums to 1 per pixel) |
| `onset` | `float32[N, 1, 64, 64]`, predicted days until visible symptoms, in `[0, 30]` |
| Classes | `0 healthy`, `1 early_stress` (pre-visual), `2 high_blight_risk`, `3 visible_disease` |
| Device | Must run on CPU and on CUDA; FP16 is allowed on CUDA |
| Batch | Dynamic `N` |

The caller (`api/`) owns tiling, overlap blending and masking. The model sees only fixed-size windows.

Fixture: `terraspectra_contracts.fixtures.export_stub_model()` (random weights, same signature).
