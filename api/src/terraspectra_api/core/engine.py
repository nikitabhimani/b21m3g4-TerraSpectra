"""TorchScript inference engine honouring contract C2."""

from __future__ import annotations

import logging
import os
import threading
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from terraspectra_contracts import MAX_ONSET_DAYS, N_BANDS, N_CLASSES, WINDOW_SIZE
from terraspectra_contracts.fixtures import export_stub_model

log = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """The model is missing or violates contract C2."""


def select_device(preference: str = "auto") -> torch.device:
    """``auto`` → ``cuda:0`` when available, else ``cpu``."""
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise ModelError("TS_DEVICE=cuda but CUDA is not available")
        return torch.device("cuda", 0)
    return torch.device("cpu")


class InferenceEngine:
    """Loads a TorchScript model once and runs batched, thread-safe inference."""

    def __init__(
        self,
        model_path: Path,
        device: str = "auto",
        batch_size: int = 32,
        allow_stub: bool = False,
        stub_dir: Path | None = None,
    ) -> None:
        self.model_path = model_path
        self.device_pref = device
        self.batch_size = batch_size
        self.allow_stub = allow_stub
        self.stub_dir = stub_dir
        self.device = torch.device("cpu")
        self.model: Any = None
        self.model_loaded = False  # True only when the configured (real) model is serving
        self.using_stub = False
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def device_name(self) -> str:
        return str(self.device) if self.device.type == "cpu" else f"cuda:{self.device.index or 0}"

    def load(self) -> None:
        """Load the model (or the contract stub in dev/test) and validate it against C2."""
        self.device = select_device(self.device_pref)
        path = self.model_path
        if not path.is_file():
            if not self.allow_stub:
                raise ModelError(f"model not found at {path}")
            stub_dir = self.stub_dir or Path.cwd()
            stub_dir.mkdir(parents=True, exist_ok=True)
            path = stub_dir / "stub_model.pt"
            if not path.is_file():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    tmp = path.with_suffix(f".{os.getpid()}.tmp")
                    export_stub_model(tmp)
                    os.replace(tmp, path)  # atomic: several workers may race here
            log.warning("model missing, serving contract stub", extra={"stub": str(path)})
            self.using_stub = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            model = torch.jit.load(str(path), map_location=self.device)
        model.eval()
        self.model = model
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
        self._predict_chunk(np.zeros((1, N_BANDS, WINDOW_SIZE, WINDOW_SIZE), np.float32))
        self.model_loaded = not self.using_stub
        log.info(
            "model loaded",
            extra={"path": str(path), "device": self.device_name, "stub": self.using_stub},
        )

    def predict(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run ``float32[N,200,64,64]`` → ``(probs[N,4,64,64], onset[N,1,64,64])`` on CPU numpy."""
        if self.model is None:
            raise ModelError("engine not loaded")
        if windows.ndim != 4 or windows.shape[1] != N_BANDS:
            raise ModelError(f"expected [N,{N_BANDS},H,W] input, got {windows.shape}")
        probs_out, onset_out = [], []
        for start in range(0, windows.shape[0], self.batch_size):
            p, o = self._predict_chunk(windows[start : start + self.batch_size])
            probs_out.append(p)
            onset_out.append(o)
        return np.concatenate(probs_out), np.concatenate(onset_out)

    def _predict_chunk(self, chunk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, _, h, w = chunk.shape
        tensor = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32))
        cuda = self.device.type == "cuda"
        if cuda:
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
        # TODO(Day 6): dedicated CUDA streams + multi-GPU sharding for concurrent jobs.
        autocast = torch.autocast("cuda", dtype=torch.float16) if cuda else nullcontext()
        with self._lock, torch.inference_mode(), autocast:
            out = self.model(tensor)
        if not isinstance(out, (tuple, list)) or len(out) != 2:
            raise ModelError("model must return a (probs, onset) tuple")
        probs = out[0].float().cpu().numpy()
        onset = out[1].float().cpu().numpy()
        if probs.shape != (n, N_CLASSES, h, w) or onset.shape != (n, 1, h, w):
            raise ModelError(
                f"C2 violation: probs {probs.shape}, onset {onset.shape} for input {chunk.shape}"
            )
        return probs, np.clip(onset, 0.0, MAX_ONSET_DAYS)
