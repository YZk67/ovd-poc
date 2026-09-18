from types import SimpleNamespace

import pytest
import torch

from lami_dino.diagnostic_ops import (
    fuse_detector_vlm_scores,
    sparse_fusion_candidate_pairs,
)
from tools import evaluate_ovd_fusion as evaluator
from tools.dump_ovd_raw_scores import (
    DEFAULT_SWEEP_BETAS,
    DEFAULT_SWEEP_SCALES,
    SWEEP_PROFILE_PREFIX,
    build_profiles,
    build_sweep_profiles,
    parse_float_list,
    sweep_profile_name,
)


def fake_model(alpha=0.0, beta=0.3, novel_scale=3.0):
    return SimpleNamespace(alpha=alpha, beta=beta, novel_scale=novel_scale)


def sweep_only(profiles):
    return {
        profile["name"]: profile
        for profile in profiles
        if profile["name"].startswith(SWEEP_PROFILE_PREFIX)
    }


def test_default_grid_covers_current_and_locked_protocols():
    profiles = build_profiles(fake_model())
    names = [profile["name"] for profile in profiles]
    assert len(names) == len(set(names))
    sweep = sweep_only(profiles)
    assert len(sweep) == len(DEFAULT_SWEEP_BETAS) * len(DEFAULT_SWEEP_SCALES)
    # Current evaluation protocol and EXPERIMENT_LOCK protocol are both cells.
    assert sweep_profile_name(0.3, 3.0) == "power_beta0.3_scale3"
    assert "power_beta0.3_scale3" in sweep
    assert "power_beta0.4_scale5" in sweep
    for profile in sweep.values():
        assert profile["fusion"] == "power"
        assert profile["detector_source"] == "logmeanexp"
        assert profile["base_weight"] == 0.0
        assert 0.0 <= profile["novel_weight"] <= 1.0
        assert profile["novel_scale"] > 0.0


def test_grid_is_independent_of_cli_fusion_overrides():
    first = build_profiles(fake_model(beta=0.3, novel_scale=3.0))
    second = build_profiles(fake_model(beta=0.5, novel_scale=5.0))
    assert sweep_only(first) == sweep_only(second)
    current_first = next(p for p in first if p["name"] == "current_power")
    current_second = next(p for p in second if p["name"] == "current_power")
    assert current_first["novel_weight"] == 0.3 and current_first["novel_scale"] == 3.0
    assert current_second["novel_weight"] == 0.5 and current_second["novel_scale"] == 5.0


def test_custom_grid_and_validation():
    custom = build_sweep_profiles(betas=(0.2, 0.2), scales=(2.0,))
    assert [profile["name"] for profile in custom] == ["power_beta0.2_scale2"]
    with pytest.raises(ValueError):
        build_sweep_profiles(betas=(1.5,), scales=(3.0,))
    with pytest.raises(ValueError):
        build_sweep_profiles(betas=(0.3,), scales=(0.0,))
    with pytest.raises(ValueError):
        build_sweep_profiles(betas=(), scales=(3.0,))
    assert parse_float_list("0.3,0.4, 0.5") == [0.3, 0.4, 0.5]
    with pytest.raises(Exception):
        parse_float_list("model.beta=0.3")


def test_candidate_pool_is_exact_for_every_grid_cell():
    torch.manual_seed(0)
    num_queries, num_classes, topk = 12, 30, 20
    detector = torch.randn(num_queries, num_classes) * 3.0
    vlm = torch.randn(num_queries, num_classes) * 2.0
    novel = torch.zeros(num_classes, dtype=torch.bool)
    novel[::3] = True
    profiles = build_profiles(fake_model())
    queries, classes = sparse_fusion_candidate_pairs(
        detector, vlm, novel, topk=topk, profiles=profiles
    )
    pool = set(zip(queries.tolist(), classes.tolist()))
    for profile in sweep_only(profiles).values():
        dense = fuse_detector_vlm_scores(
            detector,
            vlm,
            novel,
            fusion=profile["fusion"],
            base_weight=profile["base_weight"],
            novel_weight=profile["novel_weight"],
            novel_scale=profile["novel_scale"],
        )
        top = dense.reshape(-1).topk(topk).indices
        pairs = {(int(i) // num_classes, int(i) % num_classes) for i in top}
        assert pairs <= pool


def test_profile_selection_by_explicit_names_and_prefix():
    available = ["current_power", "detector_only", "power_beta0.3_scale3", "power_beta0.4_scale5"]
    assert evaluator.select_profile_names(available) == available
    assert evaluator.select_profile_names(available, ["detector_only"]) == ["detector_only"]
    assert evaluator.select_profile_names(available, prefix="power_beta") == [
        "power_beta0.3_scale3",
        "power_beta0.4_scale5",
    ]
    assert evaluator.select_profile_names(
        available, ["power_beta0.4_scale5", "current_power"], "power_beta"
    ) == ["power_beta0.4_scale5", "current_power", "power_beta0.3_scale3"]
    with pytest.raises(ValueError):
        evaluator.select_profile_names(available, ["missing"])
    with pytest.raises(ValueError):
        evaluator.select_profile_names(available, prefix="nothing")


def test_current_apr_reproduction_check():
    results = {"current_power": {"AP": 44.6979, "APr": 42.4229}}
    assert evaluator.check_current_apr(results, 42.43, 0.02) == pytest.approx(-0.0071)
    with pytest.raises(RuntimeError):
        evaluator.check_current_apr(results, 42.9, 0.02)
    with pytest.raises(ValueError):
        evaluator.check_current_apr({}, 42.4229, 0.02)


def test_sweep_grid_and_best_cell(capsys):
    profiles = build_profiles(fake_model())
    profiles_by_name = {profile["name"]: profile for profile in profiles}
    results = {
        "current_power": {"AP": 44.70, "APr": 42.42},
        "power_beta0.3_scale3": {"AP": 44.70, "APr": 42.42},
        "power_beta0.4_scale5": {"AP": 44.50, "APr": 43.90},
        "detector_only": {"AP": 40.0, "APr": 30.0},
    }
    grid = evaluator.sweep_grid(results, profiles_by_name)
    assert set(grid) == {0.3, 0.4}
    assert grid[0.4][5.0]["profile"] == "power_beta0.4_scale5"
    best = evaluator.best_sweep_cell(grid)
    assert best[0] == 0.4 and best[1] == 5.0
    evaluator.print_sweep_grid(grid, results["current_power"])
    captured = capsys.readouterr().out
    assert "beta=0.4 scale=5" in captured
    assert "APr +1.4800" in captured
    assert evaluator.best_sweep_cell({}) is None
