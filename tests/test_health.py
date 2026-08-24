import json

import pytest

from cli import main
from health import collect_health, to_artifact_dict


def test_health_report_is_healthy():
    report = collect_health("config/simulator_config.yaml")
    assert report["healthy"] is True
    assert all(report["dependencies"]["required"].values())


def test_health_artifact_rejects_malformed_reports():
    report = collect_health("config/simulator_config.yaml")
    for field, value, message in (("healthy", "yes", "health healthy flag must be boolean"), ("dependencies", [], "health dependencies must be an object")):
        malformed = {**report, field: value}
        with pytest.raises(ValueError, match=message):
            to_artifact_dict(malformed, "config/simulator_config.yaml")

    with pytest.raises(ValueError, match="config_path must be a non-empty string"):
        to_artifact_dict(report, "")


def test_cli_health(capsys):
    assert main(["health"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is True
    assert report["config"]["valid"] is True
