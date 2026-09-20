import torch
import torch.nn.functional as F

from terraspectra_model.losses import TerraSpectraLoss, focal_loss, masked_huber


def _data() -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.randn(2, 4, 8, 8)
    target = torch.randint(0, 4, (2, 8, 8))
    return logits, target


def test_focal_equals_ce_when_gamma_zero() -> None:
    logits, target = _data()
    target[0, :2] = -1
    w = torch.tensor([0.5, 1.0, 2.0, 1.5])
    ours = focal_loss(logits, target, gamma=0.0, weight=w, ignore_index=-1)
    ref = F.cross_entropy(logits, target, weight=w, ignore_index=-1)
    torch.testing.assert_close(ours, ref)


def test_focal_down_weights_easy_examples() -> None:
    logits, target = _data()
    assert focal_loss(logits, target, gamma=2.0) < focal_loss(logits, target, gamma=0.0)


def test_valid_mask_excludes_pixels() -> None:
    logits, target = _data()
    mask = torch.zeros(2, 8, 8, dtype=torch.bool)
    mask[:, :4] = True
    masked = focal_loss(logits, target, gamma=1.0, valid_mask=mask)
    garbage = logits.clone()
    garbage[:, :, 4:] = 1e4 * torch.randn_like(garbage[:, :, 4:])
    torch.testing.assert_close(masked, focal_loss(garbage, target, gamma=1.0, valid_mask=mask))
    sliced = focal_loss(logits[:, :, :4], target[:, :4], gamma=1.0)
    torch.testing.assert_close(masked, sliced)


def test_masked_huber() -> None:
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.full((1, 4, 4), 100.0)
    mask = torch.zeros(1, 4, 4, dtype=torch.bool)
    assert masked_huber(pred, target, mask) == 0
    target[0, 0, 0] = 1.0
    mask[0, 0, 0] = True
    torch.testing.assert_close(masked_huber(pred, target, mask, delta=2.0), torch.tensor(0.5))


def test_combined_loss_parts() -> None:
    logits, target = _data()
    crit = TerraSpectraLoss(gamma=2.0, class_weights=[1, 1, 1, 1], onset_weight=0.5)
    onset = torch.rand(2, 1, 8, 8) * 30
    total, parts = crit(logits, onset, target, torch.rand(2, 8, 8) * 30,
                        torch.ones(2, 8, 8, dtype=torch.bool))  # fmt: skip
    torch.testing.assert_close(total, parts["loss_cls"] + 0.5 * parts["loss_onset"])
