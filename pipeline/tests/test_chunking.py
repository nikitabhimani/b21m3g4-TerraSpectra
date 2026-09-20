import numpy as np
import pytest

from terraspectra_pipeline.chunking import Stitcher, feather_weights, iter_blocks, iter_windows


@pytest.mark.parametrize(
    ("h", "w", "size", "overlap"),
    [(64, 64, 64, 16), (100, 130, 64, 16), (30, 200, 64, 0), (257, 64, 64, 32), (1, 1, 64, 16)],
)
def test_windows_cover_every_pixel(h: int, w: int, size: int, overlap: int) -> None:
    hits = np.zeros((h, w), dtype=int)
    for win in iter_windows(h, w, size, overlap):
        r, c = int(win.row_off), int(win.col_off)
        assert r >= 0 and c >= 0 and r + win.height <= h and c + win.width <= w
        assert win.height == min(size, h) and win.width == min(size, w)
        hits[r : r + int(win.height), c : c + int(win.width)] += 1
    assert (hits >= 1).all()


def test_iter_blocks_is_disjoint_tiling() -> None:
    hits = np.zeros((100, 70), dtype=int)
    for win in iter_blocks(100, 70, 32):
        r, c = int(win.row_off), int(win.col_off)
        hits[r : r + int(win.height), c : c + int(win.width)] += 1
    assert (hits == 1).all()


def test_iter_windows_validates_args() -> None:
    with pytest.raises(ValueError):
        list(iter_windows(10, 10, size=8, overlap=8))
    assert list(iter_windows(0, 10)) == []


def test_feather_weights_positive_and_ramped() -> None:
    wts = feather_weights(64, 16)
    assert wts.shape == (64, 64) and (wts > 0).all()
    assert wts[32, 32] == 1.0 and wts[0, 0] < wts[8, 8] < wts[32, 32]
    assert feather_weights((4, 6), 0).min() == 1.0


@pytest.mark.parametrize(("h", "w"), [(64, 64), (150, 97)])
def test_stitch_constant_prediction(h: int, w: int) -> None:
    stitcher = Stitcher(channels=4, height=h, width=w, overlap=16)
    for win in iter_windows(h, w, 64, 16):
        pred = np.full((4, int(win.height), int(win.width)), 0.25, dtype=np.float32)
        stitcher.add(win, pred)
    out = stitcher.result()
    assert out.shape == (4, h, w)
    np.testing.assert_allclose(out, 0.25, rtol=1e-6)


def test_stitch_blends_and_respects_validity() -> None:
    stitcher = Stitcher(channels=1, height=64, width=100, overlap=16)
    wins = list(iter_windows(64, 100, 64, 16))
    assert len(wins) == 2
    stitcher.add(wins[0], np.zeros((1, 64, 64), np.float32))
    stitcher.add(wins[1], np.ones((1, 64, 64), np.float32))
    out = stitcher.result()[0]
    assert out[0, 0] == 0.0 and out[0, -1] == 1.0
    row = out[32]
    assert np.all(np.diff(row) >= -1e-6)  # monotone blend across the overlap

    partial = Stitcher(channels=1, height=8, width=8, overlap=0)
    valid = np.zeros((8, 8), bool)
    valid[:4] = True
    partial.add(next(iter_windows(8, 8, 8, 0)), np.ones((1, 8, 8), np.float32), valid)
    res = partial.result(fill=-1.0)
    assert (res[0, :4] == 1).all() and (res[0, 4:] == -1).all()
    with pytest.raises(ValueError):
        partial.add(wins[0], np.ones((1, 3, 3), np.float32))
