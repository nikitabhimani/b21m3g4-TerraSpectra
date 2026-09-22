# TerraSpectra ML Track (P2) — Architecture Ablation Report (Day 9)

**Date:** September 22, 2026  
**Dataset:** Pre-visual hyperspectral fungal blight stress progression (128 train / 32 val windows, 200 canonical bands)  
**Task:** 4-class segmentation (0=Healthy, 1=Early Stress, 2=High Blight Risk, 3=Visible Disease) + Days-to-Onset regression.

---

## 1. Comparative Performance Table

| Architecture Variant | Parameters | Train Loss | Val Loss | Overall Accuracy (OA) | Macro F1 | Mean IoU (mIoU) | Key Takeaway |
|---|---|---|---|---|---|---|---|
| **Hybrid (3D-CNN + ViT)** | **743 K** | **1.979** | **2.011** | **95.9%** | **0.519** | **0.455** | **Optimal:** Fuses local 3D spectral convolutions with transformer attention; best stress sensitivity (F1=0.575 on early stress). |
| **ViT Only** | 692 K | 2.431 | 2.176 | 94.6% | 0.247 | 0.238 | Fast convergence on background pixels, but drops early-stress F1 significantly without 3D spectral feature hierarchy. |
| **3D-CNN Only** | 422 K | 2.528 | 2.912 | 63.9% | 0.234 | 0.180 | Lacks long-range spatial-spectral context; struggles on multi-patch field boundaries. |

---

## 2. Key Architectural Insights

1. **Why the 3D-CNN Spectral-Spatial Encoder Matters:**
   - 200 contiguous spectral bands contain subtle slope changes (such as the 700–740 nm red-edge shift and 531/570 nm PRI variations).
   - 3D convolution kernels $(7\times 3\times 3 \to 5\times 3\times 3 \to 3\times 3\times 3)$ act as learnable derivative operators across neighboring wavelengths before spatial tokenization.
   - When replaced with a simple $1\times 1$ 2D projection (as in `vit_only`), Macro F1 plummeted from **0.519 to 0.247**.

2. **Why the Vision Transformer Head Matters:**
   - Crop disease outbreaks manifest as spatial clusters / infection foci. Self-attention across spatial patches enables the model to condition each pixel's risk score on canopy context rather than isolated pixel spectra.
   - Removing the ViT encoder (as in `cnn_only`) caused Overall Accuracy to collapse to **63.9%**.

3. **Conclusion:**
   - The **Hybrid (3D-CNN + ViT)** architecture is firmly validated as the production architecture for Contract C2.
