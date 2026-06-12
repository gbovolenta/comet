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
    slab: str
    gas_list: List[str]
    gas_masses: List[float]
    z_cutoff: float
    temperature: float

    # --- pressure / chemical potential ---
    partial_pressures: Optional[Dict[str, float]] = None
    pressure: Optional[float] = None
    pressure_unit: PressureUnit = "bar"
    y1: float = 0.75

    # --- Monte Carlo controls ---
    steps: int
    run_until_converged: bool = False
    max_steps: Optional[int] = None           # defaults to `steps` if unset
    biased_moves: bool = False
    log_mu_diagnostics: bool = False
    log_every: int = 1
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
        if len(self.gas_masses) != len(self.gas_list):
            raise ValueError(
                f"gas_masses (len {len(self.gas_masses)}) must match "
                f"gas_list (len {len(self.gas_list)})"
            )
        if self.slab not in self.elements:
            raise ValueError(f"slab {self.slab!r} must be one of elements {self.elements}")

        if self.partial_pressures is not None:
            unknown = set(self.partial_pressures) - set(self.gas_list)
            if unknown:
                raise ValueError(
                    f"partial_pressures has species not in gas_list: {sorted(unknown)}"
                )
        elif self.pressure is None:
            raise ValueError(
                "provide either 'partial_pressures' or 'pressure' to set chemical-potential targets"
            )

        if self.energy_backend == "mace" and self.model_dir is None:
            raise ValueError("energy_backend 'mace' requires 'model_dir'")

        if self.max_steps is None:
            self.max_steps = self.steps

        return self

    def gas_masses_by_species(self) -> Dict[str, float]:
        """Return the `{species: mass}` mapping (aligned `gas_list`/`gas_masses`)."""
        return dict(zip(self.gas_list, self.gas_masses))


def load_run_config(path: str) -> RunConfig:
    """Load and validate a COMET configuration file into a :class:`RunConfig`."""
    return RunConfig(**load_config(path))
