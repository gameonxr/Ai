import json

from cli import main
from health import collect_health


def test_health_report_is_healthy():
    report = collect_health("config/simulator_config.yaml")
    assert report["healthy"] is True
    assert all(report["dependencies"]["required"].values())


def test_cli_health(capsys):
    assert main(["health"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["healthy"] is True
    assert report["config"]["valid"] is True
