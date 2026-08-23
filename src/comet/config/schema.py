"""Typed, validated configuration for COMET runs.

`load_config` (in :mod:`comet.config.io`) remains a thin ``yaml.safe_load``
wrapper returning a raw dict. This module adds a validated :class:`RunConfig`
built on top of it, so misconfigured runs fail fast at the boundary with a clear
message instead of deep inside the workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from comet.config.io import load_config

PressureUnit = Literal["Pa", "bar", "atm", "torr"]
EnergyBackend = Literal["mace", "orca"]


class MDConfig(BaseModel):
    """Settings for the ASE-MD half of `comet cycle` (GCMC <-> MD alternation).

    Defaults mirror the production LAMMPS recipe: CSVR (Bussi) thermostat at
    the run temperature, frozen bottom slab layers, reflective lid at the cell
    top so gas cannot escape the non-periodic z direction.
    """

    model_config = ConfigDict(extra="forbid")

    n_cycles: int = 3            # GCMC <-> MD alternations
    md_steps: int = 2000         # MD steps per cycle
    timestep_fs: float = 0.5     # [fs]
    tau_t_ps: float = 0.05       # CSVR (Bussi) time constant [ps]
    freeze_bottom: float = 2.0   # atoms with z below this [Å] are held fixed
    traj_every: int = 0          # extxyz frame interval during MD (0 = off)


class RunConfig(BaseModel):
    """Validated COMET run configuration.

    Unknown keys are rejected (``extra="forbid"``) so stale or mistyped keys
    surface immediately rather than being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    # --- backend ---
    energy_backend: EnergyBackend = "mace"

    # --- paths ---
    bdir: Path = Path(".")
    model_dir: Optional[Path] = None          # required for the mace backend
    restart_path: Path
    gas_template_dir: Path

    # --- system ---
    elements: List[str]
    slab: Optional[str] = None                # informational; sanity-checked against `elements`
    gas_list: List[str]
    gas_masses: Optional[List[float]] = None  # redundant: masses derive from templates
    z_cutoff: float
    temperature: float

    # --- pressure / chemical potential ---
    # Preferred input: total `pressure` + integer composition `ratios`
    # (e.g. {H2: 3, N2: 1}; p_i = r_i/Σr · pressure). Integer ratios make the
    # composition exactly representable by integer molecule counts, which the
    # count-based convergence targets require. `partial_pressures` remains
    # accepted as the absolute-pressure alternative (no composition claim).
    ratios: Optional[Dict[str, int]] = None
    partial_pressures: Optional[Dict[str, float]] = None
    pressure: Optional[float] = None
    pressure_unit: PressureUnit = "bar"
    y1: float = 0.75

    # --- GCMC <-> MD cycling (only used by `comet cycle`) ---
    md: Optional[MDConfig] = None

    # --- Monte Carlo controls ---
    steps: int
    seed: Optional[int] = None                # RNG seed for reproducible runs
    run_until_converged: bool = False
    max_steps: Optional[int] = None           # defaults to `steps` if unset
    biased_moves: bool = False
    log_mu_diagnostics: bool = False
    log_every: int = 1
    # Unused since count-based convergence targets (direction is deterministic
    # from ΔN); retained so existing configs remain valid.
    force_dmu_threshold: float = 0.0
    force_single_species: bool = True

    # --- ORCA backend settings (only used when energy_backend == "orca") ---
    orca_method: str = "PBE"
    orca_basis: str = "def2-SVP"
    orca_nprocs: int = 1
    orca_charge: int = 0
    orca_mult: int = 1
    orca_simpleinput: Optional[str] = None
    orca_blocks: str = ""
    orca_work_root: Optional[Path] = None
    orca_keep_workdirs: bool = False

    # ---- field-level coercion ----

    @field_validator("slab", mode="before")
    @classmethod
    def _normalize_slab(cls, v):
        """Accept `slab` as a symbol (`Fe`) or a single-item list (`[Fe]`)."""
        if isinstance(v, (list, tuple)):
            if len(v) != 1:
                raise ValueError(
                    "slab must be a single element symbol (or a 1-item list)"
                )
            return v[0]
        return v

    # ---- cross-field validation ----

    @model_validator(mode="after")
    def _check_consistency(self) -> "RunConfig":
        if self.gas_masses is not None and len(self.gas_masses) != len(self.gas_list):
            raise ValueError(
                f"gas_masses (len {len(self.gas_masses)}) must match "
                f"gas_list (len {len(self.gas_list)})"
            )
        if self.slab is not None and self.slab not in self.elements:
            raise ValueError(f"slab {self.slab!r} must be one of elements {self.elements}")

        if self.ratios is not None:
            if self.partial_pressures is not None:
                raise ValueError(
                    "provide either 'ratios' or 'partial_pressures', not both"
                )
            if self.pressure is None:
                raise ValueError("'ratios' requires a total 'pressure'")
            unknown = set(self.ratios) - set(self.gas_list)
            if unknown:
                raise ValueError(
                    f"ratios has species not in gas_list: {sorted(unknown)}"
                )
            for gas, r in self.ratios.items():
                if r < 1:
                    raise ValueError(f"ratios[{gas}] = {r} must be a positive integer")
            # Resolve to absolute partial pressures (p_i = r_i/Σr · P); a
            # species omitted from ratios stays frozen, same as with
            # partial_pressures.
            r_sum = sum(self.ratios.values())
            self.partial_pressures = {
                gas: (r / r_sum) * self.pressure for gas, r in self.ratios.items()
            }

        if self.partial_pressures is not None:
            unknown = set(self.partial_pressures) - set(self.gas_list)
            if unknown:
                raise ValueError(
                    f"partial_pressures has species not in gas_list: {sorted(unknown)}"
                )
        elif self.pressure is None:
            raise ValueError(
                "provide 'pressure' + 'ratios' (or 'partial_pressures') "
                "to set chemical-potential targets"
            )

        if self.energy_backend == "mace" and self.model_dir is None:
            raise ValueError("energy_backend 'mace' requires 'model_dir'")

        if self.max_steps is None:
            self.max_steps = self.steps

        return self

    def gas_masses_by_species(self) -> Optional[Dict[str, float]]:
        """Return the declared `{species: mass}` mapping, or ``None`` if unset.

        Masses are normally derived from the gas templates; a declared
        `gas_masses` list is only used as a consistency check.
        """
        if self.gas_masses is None:
            return None
        return dict(zip(self.gas_list, self.gas_masses))


def load_run_config(path: str) -> RunConfig:
    """Load and validate a COMET configuration file into a :class:`RunConfig`."""
    return RunConfig(**load_config(path))
