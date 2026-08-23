import json
from pathlib import Path

from cli import main


def test_cli_validate(capsys):
    assert main(["validate", "config/simulator_config.yaml"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True


def test_cli_benchmark(capsys):
    assert main(["benchmark", "--steps", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps"] == 2


def test_cli_evaluate(capsys):
    assert main(["evaluate", "--episodes", "2", "--max-steps", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["episodes"] == 2
    assert payload["total_steps"] == 6


def test_cli_run(capsys, tmp_path: Path):
    assert main(["run", "--run-id", "cli-test", "--manifest-dir", str(tmp_path), "--episodes", "1", "--max-steps", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert (tmp_path / "cli-test.json").exists()
