import numpy as np
import pytest
from terraspectra_contracts import MAX_ONSET_DAYS, N_BANDS
from terraspectra_contracts.fixtures import make_synthetic_cube

from terraspectra_model import indices
from terraspectra_model.config import SynthConfig
from terraspectra_model.synth.proxy_labels import proxy_labels
from terraspectra_model.synth.stress import (
    blend_stress_into_scene,
    generate_dataset,
    get_backend,
    label_for_days,
    make_field_patch,
    simulate_progression,
)


def test_progression_red_edge_blue_shift_and_pri_decline() -> None:
    r = simulate_progression(days=[30, 14, 5])
    rep = indices.red_edge_position(r["spectra"].T)
    healthy_rep = float(indices.red_edge_position(r["healthy"]))
    assert rep[2] < rep[1] < healthy_rep
    pri = indices.pri(r["spectra"].T)
    assert pri[2] < pri[0]
    green = indices.band_at(r["spectra"].T, 550)
    assert green[1] > indices.band_at(r["healthy"], 550)  # green peak rises as Cab falls
    assert r["spectra"].shape == (3, N_BANDS)


def test_label_rules() -> None:
    days = np.array([0, 3, 7, 8, 25, 26, 30, 0])
    infected = np.array([True] * 7 + [False])
    cls, onset = label_for_days(days, infected)
    assert cls.tolist() == [3, 2, 2, 1, 1, 0, 0, 0]
    assert onset.tolist() == [0, 3, 7, 8, 25, 30, 30, 30]


def test_field_patch_labels_and_ranges() -> None:
    rng = np.random.default_rng(0)
    cfg = SynthConfig(patch_infected_p=1.0, nodata_p=1.0)
    p = make_field_patch(rng, cfg)
    assert p.cube.shape == (N_BANDS, 64, 64) and p.cube.dtype == np.float32
    assert p.cube.min() >= 0 and p.cube.max() <= 1
    assert set(np.unique(p.class_map)) <= {0, 1, 2, 3}
    assert p.onset_map.min() >= 0 and p.onset_map.max() <= MAX_ONSET_DAYS
    assert (p.class_map > 0).any()
    assert np.all(p.cube[:, ~p.valid_mask] == 0)


def test_fixed_days_patch() -> None:
    p = make_field_patch(np.random.default_rng(1), SynthConfig(), fixed_days=3)
    assert set(np.unique(p.class_map)) == {0, 2}


def test_generate_dataset_shapes() -> None:
    d = generate_dataset(2, seed=0)
    assert d["x"].shape == (2, N_BANDS, 64, 64)
    assert d["y"].dtype == np.int64 and d["mask"].dtype == bool


def test_multistage_epidemic_progression_covers_all_classes() -> None:
    d = generate_dataset(15, seed=42, cfg=SynthConfig(patch_infected_p=1.0, max_blobs=3))
    classes_present = set(np.unique(d["y"]))
    assert classes_present == {0, 1, 2, 3}, f"Expected all 4 classes, got {classes_present}"


def test_blend_stress_into_real_pixels() -> None:
    cube, _ = make_synthetic_cube(64, 64, seed=0)
    cube = np.clip(cube, 0, 1)
    veg = np.ones((64, 64), dtype=bool)
    veg[:, :8] = False
    scene = blend_stress_into_scene(cube, veg, np.random.default_rng(0), foci_per_megapixel=2000)
    assert (scene.class_map[:, :8] == -1).all()
    changed = np.any(scene.cube != cube, axis=0)
    assert not changed[:, :8].any()
    assert changed.any() == (scene.class_map > 0).any()


def test_proxy_labels_flag_stressed_patch() -> None:
    cube, stressed = make_synthetic_cube(96, 96, seed=0, stressed_fraction=0.1)
    out = proxy_labels(cube)
    assert out.class_map.shape == (96, 96)
    assert out.class_map[0, 0] == -1  # nodata corner
    assert (out.class_map[stressed] > 0).mean() > 0.9
    assert (out.class_map[~stressed & (out.class_map >= 0)] == 0).mean() > 0.9
    assert out.confidence.min() >= 0 and out.confidence.max() <= 1


def test_prosail_backend_optional() -> None:
    pytest.importorskip("prosail")
    fn = get_backend("prosail")
    ones = np.ones(2)
    out = fn(np.array([40.0, 15.0]), 0.015 * ones, 3 * ones, 0 * ones, 0 * ones, ones)
    assert out.shape == (2, N_BANDS)
    rep = indices.red_edge_position(out.T)
    assert rep[1] < rep[0]
