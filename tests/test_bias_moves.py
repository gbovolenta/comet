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


def test_choose_biased_move_returns_none_when_all_converged():
    assert choose_biased_move({"H2": 3}, {"H2": 3}, set()) == (None, None)


def test_choose_biased_move_single_species_inserts_below_target():
    move_name, ins = choose_biased_move(
        n_targets={"H2": 4},
        gas_counts={"H2": 2},
        unconverged={"H2"},
        force_single_species=True,
    )
    assert move_name == "H2"
    assert ins is True


def test_choose_biased_move_single_species_deletes_above_target():
    move_name, ins = choose_biased_move(
        n_targets={"H2": 4},
        gas_counts={"H2": 7},
        unconverged={"H2"},
        force_single_species=True,
    )
    assert move_name == "H2"
    assert ins is False


def test_choose_biased_move_weights_species_by_count_mismatch(monkeypatch):
    captured = {}

    def fake_choices(seq, weights, k):
        captured["seq"] = list(seq)
        captured["weights"] = list(weights)
        return ["N2"]

    monkeypatch.setattr("comet.mc.moves.random.choices", fake_choices)
    move_name, ins = choose_biased_move(
        n_targets={"H2": 12, "N2": 6},
        gas_counts={"H2": 11, "N2": 9},   # ΔN = +1, -3
        unconverged={"H2", "N2"},
    )
    assert move_name == "N2"
    assert ins is False                    # above target -> delete
    weights = dict(zip(captured["seq"], captured["weights"]))
    assert weights == {"H2": 1, "N2": 3}   # |ΔN| weighting


def test_choose_biased_move_direction_is_deterministic_from_sign():
    # Below target -> insert, regardless of RNG state.
    move_name, ins = choose_biased_move(
        n_targets={"H2": 5},
        gas_counts={"H2": 0},
        unconverged={"H2"},
        force_single_species=True,
    )
    assert (move_name, ins) == ("H2", True)
