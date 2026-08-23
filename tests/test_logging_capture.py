"""Module loggers under the `comet` namespace must reach gcmc_run.log.

Regression test: spectator warnings (comet.system.molecules) and backend info
(comet.potentials.mace) were silently dropped because setup_logging only
attached handlers to the "gcmc" logger.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from comet.workflows.logging_utils import setup_logging


def _reset_handlers():
    for name in ("gcmc", "comet"):
        log = logging.getLogger(name)
        for handler in list(log.handlers):
            log.removeHandler(handler)
            handler.close()


def test_package_module_loggers_reach_log_file(tmp_path, monkeypatch):
    _reset_handlers()
    monkeypatch.chdir(tmp_path)
    try:
        setup_logging()
        logging.getLogger("gcmc").info("workflow line")
        logging.getLogger("comet.system.molecules").warning("spectator fragment test")
        logging.getLogger("comet.potentials.mace").info("device line test")

        for handler in logging.getLogger("gcmc").handlers:
            handler.flush()
        text = (tmp_path / "gcmc_run.log").read_text()
    finally:
        _reset_handlers()

    assert "workflow line" in text
    assert "spectator fragment test" in text
    assert "device line test" in text


def test_setup_logging_is_idempotent(tmp_path, monkeypatch):
    _reset_handlers()
    monkeypatch.chdir(tmp_path)
    try:
        setup_logging()
        n_gcmc = len(logging.getLogger("gcmc").handlers)
        n_pkg = len(logging.getLogger("comet").handlers)
        setup_logging()   # second call must not duplicate handlers
        assert len(logging.getLogger("gcmc").handlers) == n_gcmc
        assert len(logging.getLogger("comet").handlers) == n_pkg
    finally:
        _reset_handlers()


def test_info_lines_have_no_level_prefix_warnings_keep_it(tmp_path, monkeypatch):
    _reset_handlers()
    monkeypatch.chdir(tmp_path)
    try:
        setup_logging()
        logging.getLogger("gcmc").info("plain info line")
        logging.getLogger("gcmc").warning("warned line")
        for handler in logging.getLogger("gcmc").handlers:
            handler.flush()
        text = (tmp_path / "gcmc_run.log").read_text()
    finally:
        _reset_handlers()

    assert "plain info line" in text
    assert "INFO: plain info line" not in text
    assert "WARNING: warned line" in text


def test_banner_contains_author_and_version_no_quote(tmp_path, monkeypatch):
    from comet import __version__
    from comet.workflows.logging_utils import log_banner

    _reset_handlers()
    monkeypatch.chdir(tmp_path)
    try:
        setup_logging()
        log_banner("GCMC pressure control")
        for handler in logging.getLogger("gcmc").handlers:
            handler.flush()
        text = (tmp_path / "gcmc_run.log").read_text()
    finally:
        _reset_handlers()

    assert f"COMET — v{__version__}" in text
    assert "Giulia M. Bovolenta" in text
    assert "GCMC pressure control" in text
    assert '"' not in text   # no quotation block
