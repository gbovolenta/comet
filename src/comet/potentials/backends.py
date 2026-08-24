"""Energy-backend construction and injection.

`build_energy_backend` turns a validated :class:`~comet.config.schema.RunConfig`
into an :class:`EnergyBackend` — a small bundle of the two callables the
workflow needs. Passing this object into the workflow (rather than selecting the
backend inside ``run()``) is what lets tests drive the full pipeline with a stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from ase import Atoms

from comet.config.schema import RunConfig


@dataclass
class EnergyBackend:
    """The energy callables the GCMC workflow depends on.

    Attributes:
        name: Backend identifier (e.g. ``"mace"``, ``"orca"``, ``"stub"``).
        energy: ``(atoms, label) -> energy_eV`` for an arbitrary structure.
        gas_energies: ``{species: Atoms} -> {species: energy_eV}`` for the
            isolated gas references.
        calculator_factory: Optional ``() -> ase Calculator`` providing forces
            for the ASE-MD half of `comet cycle`. ``None`` for backends that
            cannot drive MD (e.g. orca).
    """

    name: str
    energy: Callable[[Atoms, str], float]
    gas_energies: Callable[[Dict[str, Atoms]], Dict[str, float]]
    calculator_factory: Optional[Callable[[], Any]] = None


def build_energy_backend(config: RunConfig) -> EnergyBackend:
    """Construct the energy backend selected by ``config.energy_backend``."""
    if config.energy_backend == "mace":
        try:
            from comet.potentials.mace import (
                compute_gas_energies,
                get_energy_mace,
                get_mace_calculator,
            )
        except ImportError as e:  # mace-torch is an optional dependency
            raise ImportError(
                "The 'mace' energy backend requires mace-torch. "
                "Install it with: pip install comet[mace]"
            ) from e

        model_dir = config.model_dir
        return EnergyBackend(
            name="mace",
            energy=lambda atoms, label: get_energy_mace(atoms, model_dir),
            gas_energies=lambda templates: compute_gas_energies(templates, model_dir),
            calculator_factory=lambda: get_mace_calculator(model_dir),
        )

    if config.energy_backend == "orca":
        from comet.potentials.orca import compute_gas_energies_orca, get_energy_orca

        # orca.py reads config via dict.get(); drop None-valued keys so that
        # `.get(key, default)` falls back to its default (matching raw-YAML
        # semantics where an unset key is simply absent).
        cfg = {k: v for k, v in config.model_dump().items() if v is not None}
        return EnergyBackend(
            name="orca",
            energy=lambda atoms, label: get_energy_orca(atoms, cfg, label=label),
            gas_energies=lambda templates: compute_gas_energies_orca(templates, cfg),
        )

    raise ValueError(f"Unsupported energy backend: {config.energy_backend}")
