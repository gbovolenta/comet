"""Logging setup and log-formatting helpers shared across the workflow."""

from __future__ import annotations

import logging

import numpy as np

# Module-level logger handle. Handlers (and the log file) are attached lazily by
# setup_logging(), which run() calls — importing this module must not write files.
logger = logging.getLogger("gcmc")


def setup_logging() -> logging.Logger:
    """
    Configure and return a logger that writes DEBUG-level messages to the file 'gcmc_run.log'
    and INFO-level messages to the console. All log entries exclude timestamps.

    Safe to call more than once per process: handlers (and therefore the
    'gcmc_run.log' file) are created only on the first call. This is invoked from
    `run()` rather than at import time so merely importing the package does not
    create a log file.

    Returns:
        logging.Logger: The configured logger instance.
    """
    log = logging.getLogger("gcmc")
    log.setLevel(logging.DEBUG)

    # Already configured (e.g. a previous run() in this process) — don't add
    # duplicate handlers or truncate the existing log.
    if log.handlers:
        return log

    # Formatter without timestamps
    formatter = logging.Formatter("%(levelname)s: %(message)s")

    # File handler (DEBUG+)
    fh = logging.FileHandler("gcmc_run.log", mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)

    # Console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    log.addHandler(ch)

    return log


def _fmt_int(value) -> str:
    """Format an integer-like value for human-readable log output."""
    return f"{int(value)}"


def _fmt_energy(value: float) -> str:
    """Format an energy value in eV for the log."""
    return f"{float(value):.3f}"


def _fmt_prob(value: float) -> str:
    """Format an acceptance probability for the log."""
    return f"{float(value):.2f}"


def _fmt_pressure(value: float) -> str:
    """Format a pressure value for the log."""
    return f"{float(value):.2f}"


def _fmt_mu_scalar(value: float) -> str:
    """Format a chemical potential value or an inactive sentinel for the log."""
    if np.isfinite(value):
        return f"{float(value):.3f}"
    return "inactive"


def _format_mu_dict(mu_dict: dict) -> str:
    """Render a species-to-μ mapping as a compact single-line string.

    Args:
        mu_dict: Mapping from species name to chemical potential.

    Returns:
        str: Formatted dictionary-like string for logging.
    """
    return "{" + ", ".join(f"{gas}: {_fmt_mu_scalar(mu)}" for gas, mu in mu_dict.items()) + "}"


def _format_mu_status(mu_target: dict, mu_current: dict) -> str:
    """Render detailed per-species target/current μ diagnostics for logging.

    Args:
        mu_target: Mapping from species name to target chemical potential.
        mu_current: Mapping from species name to current chemical potential.

    Returns:
        str: Semicolon-separated diagnostic string.
    """
    fields = []
    for gas in mu_target:
        target = mu_target[gas]
        current = mu_current.get(gas, float("nan"))
        if np.isfinite(target) and np.isfinite(current):
            delta = target - current
            fields.append(
                f"{gas}: target={_fmt_mu_scalar(target)} current={_fmt_mu_scalar(current)} dmu={_fmt_mu_scalar(delta)}"
            )
        elif np.isfinite(target) and not np.isfinite(current):
            fields.append(
                f"{gas}: target={_fmt_mu_scalar(target)} current=no molecules dmu=n/a"
            )
        elif not np.isfinite(target):
            fields.append(
                f"{gas}: target=inactive current={_fmt_mu_scalar(current)} dmu=n/a"
            )
        else:
            fields.append(f"{gas}: target=unknown current=unknown dmu=n/a")
    return "; ".join(fields)
