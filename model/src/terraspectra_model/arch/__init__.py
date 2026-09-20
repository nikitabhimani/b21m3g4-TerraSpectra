"""Network architecture: 3D-CNN spectral-spatial encoder + ViT + segmentation decoder."""

from terraspectra_model.arch.hybrid import TerraSpectraNet, build_model

__all__ = ["TerraSpectraNet", "build_model"]
