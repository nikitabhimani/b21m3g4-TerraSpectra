import pytest
import torch
from terraspectra_contracts import MAX_ONSET_DAYS, N_BANDS, N_CLASSES

from terraspectra_model.arch.hybrid import TerraSpectraNet
from terraspectra_model.config import Config


@pytest.mark.parametrize("batch", [1, 3])
def test_forward_contract(tiny_model: TerraSpectraNet, batch: int) -> None:
    x = torch.rand(batch, N_BANDS, 64, 64)
    with torch.no_grad():
        probs, onset = tiny_model(x)
    assert probs.shape == (batch, N_CLASSES, 64, 64)
    assert onset.shape == (batch, 1, 64, 64)
    assert probs.dtype == onset.dtype == torch.float32
    torch.testing.assert_close(probs.sum(1), torch.ones(batch, 64, 64))
    assert probs.min() >= 0
    assert onset.min() >= 0 and onset.max() <= MAX_ONSET_DAYS


def test_forward_logits_shapes(tiny_model: TerraSpectraNet) -> None:
    logits, onset = tiny_model.forward_logits(torch.rand(2, N_BANDS, 64, 64))
    assert logits.shape == (2, N_CLASSES, 64, 64)
    assert onset.shape == (2, 1, 64, 64)


def test_rejects_wrong_band_count(tiny_model: TerraSpectraNet) -> None:
    with pytest.raises(ValueError, match="200"):
        tiny_model(torch.rand(1, 100, 64, 64))


@pytest.mark.parametrize("variant", ["cnn_only", "vit_only"])
def test_ablation_variants(tiny_cfg: Config, variant: str) -> None:
    cfg = tiny_cfg.model.model_copy(update={"variant": variant, "pos_embed": "sincos"})
    net = TerraSpectraNet(cfg).eval()
    probs, onset = net(torch.rand(2, N_BANDS, 64, 64))
    assert probs.shape == (2, N_CLASSES, 64, 64) and onset.shape == (2, 1, 64, 64)


def test_temperature_changes_sharpness(tiny_model: TerraSpectraNet) -> None:
    x = torch.rand(1, N_BANDS, 64, 64)
    with torch.no_grad():
        p1, _ = tiny_model(x)
        tiny_model.set_temperature(100.0)
        p2, _ = tiny_model(x)
    assert (p2 - 0.25).abs().max() < (p1 - 0.25).abs().max() + 1e-6


def test_backward_pass(tiny_model: TerraSpectraNet) -> None:
    tiny_model.train()
    logits, onset = tiny_model.forward_logits(torch.rand(2, N_BANDS, 64, 64))
    (logits.mean() + onset.mean()).backward()
    grads = [p.grad for p in tiny_model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)
