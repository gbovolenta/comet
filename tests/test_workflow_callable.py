import pytest


def test_workflow_run_callable():
    workflow_run = pytest.importorskip("comet.workflows.run")
    assert callable(workflow_run.run)
