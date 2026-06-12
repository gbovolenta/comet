import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.mc.moves import choose_biased_move, choose_unbiased_move


def test_choose_unbiased_move_forces_insertion_when_species_absent(monkeypatch):
    monkeypatch.setattr("comet.mc.moves.random.choice", lambda xs: "H2")
    move_name, ins = choose_unbiased_move({"H2", "N2"}, {"H2": 0, "N2": 3})
    assert move_name == "H2"
    assert ins is True


def test_choose_unbiased_move_uses_random_direction(monkeypatch):
    monkeypatch.setattr("comet.mc.moves.random.choice", lambda xs: "N2")
    monkeypatch.setattr("comet.mc.moves.random.random", lambda: 0.8)
    move_name, ins = choose_unbiased_move({"N2"}, {"N2": 2})
    assert move_name == "N2"
    assert ins is False


def test_choose_biased_move_single_species_forces_direction():
    move_name, ins = choose_biased_move(
        mu_target={"H2": -0.5},
        mu_current={"H2": -1.0},
        unconverged={"H2"},
        gas_counts={"H2": 2},
        force_dmu_threshold=0.0,
        force_single_species=True,
    )
    assert move_name == "H2"
    assert ins is True


def test_choose_biased_move_prioritizes_missing_species(monkeypatch):
    monkeypatch.setattr("comet.mc.moves.random.choice", lambda xs: xs[0])
    move_name, ins = choose_biased_move(
        mu_target={"H2": -0.5, "N2": -1.0},
        mu_current={"H2": float("-inf"), "N2": -1.1},
        unconverged={"H2", "N2"},
        gas_counts={"H2": 0, "N2": 2},
    )
    assert move_name == "H2"
    assert ins is True


def test_choose_biased_move_uses_weighted_species_and_threshold_direction(monkeypatch):
    monkeypatch.setattr("comet.mc.moves.random.choices", lambda seq, weights, k: ["N2"])
    move_name, ins = choose_biased_move(
        mu_target={"H2": -0.5, "N2": -1.5},
        mu_current={"H2": -0.6, "N2": -1.0},
        unconverged={"H2", "N2"},
        gas_counts={"H2": 2, "N2": 3},
        force_dmu_threshold=0.1,
    )
    assert move_name == "N2"
    assert ins is False


def test_choose_biased_move_below_threshold_falls_back_to_random_direction(monkeypatch):
    monkeypatch.setattr("comet.mc.moves.random.choices", lambda seq, weights, k: ["H2"])
    monkeypatch.setattr("comet.mc.moves.random.random", lambda: 0.3)
    move_name, ins = choose_biased_move(
        mu_target={"H2": -0.50},
        mu_current={"H2": -0.52},
        unconverged={"H2"},
        gas_counts={"H2": 1},
        force_dmu_threshold=0.1,
        force_single_species=False,
    )
    assert move_name == "H2"
    assert ins is True
