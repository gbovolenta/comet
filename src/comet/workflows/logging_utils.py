"""Logging setup and log-formatting helpers shared across the workflow."""

from __future__ import annotations

import logging

import numpy as np

from comet import __version__

# Module-level logger handle. Handlers (and the log file) are attached lazily by
# setup_logging(), which run() calls — importing this module must not write files.
logger = logging.getLogger("gcmc")

_RULE = "=" * 80
_THIN_RULE = "-" * 80

_BANNER_ART = r"""
 ██████╗ ██████╗ ███╗   ███╗███████╗████████╗
██╔════╝██╔═══██╗████╗ ████║██╔════╝╚══██╔══╝
██║     ██║   ██║██╔████╔██║█████╗     ██║
██║     ██║   ██║██║╚██╔╝██║██╔══╝     ██║
╚██████╗╚██████╔╝██║ ╚═╝ ██║███████╗   ██║
 ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚══════╝   ╚═╝
"""


class _LevelAwareFormatter(logging.Formatter):
    """Plain messages for INFO; keep the level prefix for everything else."""

    def format(self, record: logging.LogRecord) -> str:
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "%(levelname)s: %(message)s"
        return super().format(record)


def log_banner(workflow: str) -> None:
    """Write the COMET welcome banner and program header to the log."""
    width = 54
    art_lines = [line for line in _BANNER_ART.splitlines() if line.strip()]
    dots = "·" * (width + 4)
    logger.info(dots)
    for line in art_lines:
        logger.info(": %s :", line.ljust(width))
    logger.info(dots)
    logger.info("")
    logger.info(_RULE)
    logger.info(f"COMET — v{__version__}".center(80))
    logger.info("Constant-pressure COntroller using the METropolis algorithm".center(80))
    logger.info(_RULE)
    logger.info("")
    logger.info("  Workflow:  %s", workflow)
    logger.info("")
    logger.info("  By: Giulia M. Bovolenta")
    logger.info("")


def log_section(title: str) -> None:
    """Write a framed section header separating the major workflow blocks."""
    logger.info("")
    logger.info(_RULE)
    logger.info("  %s", title)
    logger.info(_RULE)


def log_settings(pairs: dict) -> None:
    """Write an aligned key/value settings block framed by thin rules."""
    logger.info(_THIN_RULE)
    key_width = max(len(k) for k in pairs) + 2
    for key, value in pairs.items():
        logger.info("  %s%s", f"{key}:".ljust(key_width), value)
    logger.info(_THIN_RULE)


def setup_logging() -> logging.Logger:
    """
    Configure and return a logger that writes DEBUG-level messages to the file 'gcmc_run.log'
    and INFO-level messages to the console. All log entries exclude timestamps.

    The same handlers are attached to BOTH the workflow logger ("gcmc") and the
    package parent logger ("comet"), so module loggers created with
    ``logging.getLogger(__name__)`` (e.g. ``comet.system.molecules`` spectator
    warnings, ``comet.potentials.mace`` device info) land in the log file too —
    previously they propagated to the handler-less root logger and were lost.

    Safe to call more than once per process: handlers (and therefore the
    'gcmc_run.log' file) are created only on the first call. This is invoked from
    `run()` rather than at import time so merely importing the package does not
    create a log file.

    Returns:
        logging.Logger: The configured workflow logger instance.
    """
    gcmc = logging.getLogger("gcmc")
    pkg = logging.getLogger("comet")
    gcmc.setLevel(logging.DEBUG)
    pkg.setLevel(logging.DEBUG)

    # Already configured (e.g. a previous run() in this process) — don't add
    # duplicate handlers or truncate the existing log; just make sure the
    # package logger shares the existing handlers.
    if gcmc.handlers:
        for handler in gcmc.handlers:
            if handler not in pkg.handlers:
                pkg.addHandler(handler)
        return gcmc

    # No timestamps; DEBUG/INFO messages are printed without a level prefix.
    formatter = _LevelAwareFormatter()

    # File handler (INFO+; per-step DEBUG detail is omitted for a clean
    # narrative log — lower this to DEBUG when troubleshooting)
    fh = logging.FileHandler("gcmc_run.log", mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Console handler (INFO+)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    for log in (gcmc, pkg):
        log.addHandler(fh)
        log.addHandler(ch)

    return gcmc


_MARK_OK = "✔"
_MARK_NOT = "━"   # bold em dash: not converged


def log_species_status(
    n_targets: dict,
    gas_counts: dict,
    mu_target: dict,
    mu_current: dict,
    show_mu: bool = False,
) -> None:
    """Write one status line per species, ending in ✔ (at target) or ━ (not).

    Frozen species (non-finite μ target) are reported as such without a mark.
    With ``show_mu``, the target/current chemical potentials and Δμ are
    appended before the mark.
    """
    rows = []  # (body, mark or None) — marks are appended in one aligned column
    for gas, mu_t in mu_target.items():
        if not np.isfinite(mu_t):
            rows.append((f"{gas}: frozen", None))
            continue
        n = int(gas_counts.get(gas, 0))
        target = int(n_targets.get(gas, 0))
        mark = _MARK_OK if n == target else _MARK_NOT
        if show_mu:
            mu_c = mu_current.get(gas, float("nan"))
            mu_desc = (
                f" | μ = {mu_c:.3f} eV (target {mu_t:.3f}, Δμ {mu_t - mu_c:+.3f})"
                if np.isfinite(mu_c)
                else " | μ = no molecules"
            )
        else:
            mu_desc = ""
        rows.append((f"{gas}: N = {n}/{target}{mu_desc}", mark))

    width = max(len(body) for body, _ in rows)
    for body, mark in rows:
        if mark is None:
            logger.info("  %s", body)
        else:
            logger.info("  %s  %s", body.ljust(width), mark)


def _fmt_counts(counts: dict) -> str:
    """Render per-species molecule counts as `H2 15, N2 5` (no dict braces)."""
    if not counts:
        return "none"
    return ", ".join(f"{gas} {int(n)}" for gas, n in counts.items())


def _fmt_species_set(species) -> str:
    """Render a species set as a comma-separated list, or an em dash if empty."""
    return ", ".join(sorted(species)) if species else "—"


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
