import json
from pathlib import Path

import pytest

from cli import main


def test_cli_validate(capsys):
    assert main(["validate", "config/simulator_config.yaml"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True


def test_cli_validate_reports_config_errors(capsys, tmp_path: Path, monkeypatch):
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("[broken", encoding="utf-8")
    assert main(["validate", str(invalid)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["valid"] is False
    assert captured.err == ""

    missing = tmp_path / "missing.yaml"
    assert main(["validate", str(missing)]) == 1
    assert json.loads(capsys.readouterr().out)["valid"] is False

    def fail_validation(_self, _path):
        raise ValueError("synthetic validation failure")

    monkeypatch.setattr("cli.ConfigurationValidator.validate", fail_validation)
    assert main(["validate", str(missing)]) == 1
    assert "Configuration validation failed: synthetic validation failure" in capsys.readouterr().err


def test_cli_rejects_invalid_numeric_arguments():
    invalid_args = [
        ["benchmark", "--steps", "0"],
        ["evaluate", "--episodes", "-1"],
        ["evaluate", "--reward-per-step", "nan"],
        ["run", "--checkpoint-every", "-1"],
    ]
    for argv in invalid_args:
        with pytest.raises(SystemExit) as error:
            main(argv)
        assert error.value.code == 2


def test_cli_runtime_commands_report_config_errors(capsys, tmp_path: Path):
    missing = tmp_path / "missing.yaml"
    for argv, message in (
        (["benchmark", "--config", str(missing), "--steps", "1"], "Benchmark failed:"),
        (["evaluate", "--config", str(missing), "--episodes", "1", "--max-steps", "1"], "Evaluation failed:"),
        (["run", "--config", str(missing), "--episodes", "1", "--max-steps", "1"], "Run failed:"),
    ):
        assert main(argv) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert message in captured.err


def test_cli_benchmark(capsys):
    assert main(["benchmark", "--steps", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["steps"] == 2


def test_cli_evaluate(capsys):
    assert main(["evaluate", "--episodes", "2", "--max-steps", "3"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["episodes"] == 2
    assert payload["total_steps"] == 6


def test_cli_sweep_reports_cases_file_errors(capsys, tmp_path: Path):
    missing = tmp_path / "missing-cases.json"
    assert main(["sweep", "--cases", str(missing)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Sweep failed: Unable to read sweep cases" in captured.err


def test_cli_run(capsys, tmp_path: Path):
    assert main(["run", "--run-id", "cli-test", "--manifest-dir", str(tmp_path), "--episodes", "1", "--max-steps", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "completed"
    assert (tmp_path / "cli-test.json").exists()


def test_cli_benchmark_writes_artifact(capsys, tmp_path: Path):
    json_out = tmp_path / "benchmark.json"
    assert main(["benchmark", "--steps", "2", "--seed", "7", "--json-out", str(json_out)]) == 0
    json.loads(capsys.readouterr().out)
    artifact = json.loads(json_out.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "benchmark"
    assert artifact["schema_version"] == 1
    assert artifact["steps"] == 2
    assert artifact["seed"] == 7
    assert artifact["config_path"]


def test_cli_health_writes_artifact(capsys, tmp_path: Path):
    json_out = tmp_path / "health.json"
    assert main(["health", "--json-out", str(json_out)]) == 0
    json.loads(capsys.readouterr().out)
    artifact = json.loads(json_out.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "health"
    assert artifact["schema_version"] == 1
    assert artifact["healthy"] is True
    assert artifact["config_path"]


def test_cli_health_reports_config_errors(capsys, tmp_path: Path, monkeypatch):
    missing = tmp_path / "missing.yaml"
    assert main(["health", "--config", str(missing)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["healthy"] is False
    assert captured.err == ""

    def fail_health(_config):
        raise ValueError("synthetic health failure")

    monkeypatch.setattr("cli.collect_health", fail_health)
    assert main(["health", "--config", str(missing)]) == 1
    assert "Health check failed: synthetic health failure" in capsys.readouterr().err


def test_cli_report_rejects_missing_manifest_directory(capsys, tmp_path: Path):
    missing = tmp_path / "missing-runs"
    assert main(["report", "--manifest-dir", str(missing)]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"Report failed: manifest directory not found: {missing}" in captured.err


def test_cli_report_strict_returns_nonzero_for_artifact_errors(capsys, tmp_path: Path):
    (tmp_path / "broken.json").write_text("{broken", encoding="utf-8")
    assert main(["report", "--manifest-dir", str(tmp_path), "--strict"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_errors"]


def test_cli_report_strict_succeeds_for_valid_artifacts(capsys, tmp_path: Path):
    assert main(["report", "--manifest-dir", str(tmp_path), "--strict"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_errors"] == []


def test_cli_evaluation_artifact_has_schema_version(capsys, tmp_path: Path):
    json_out = tmp_path / "evaluation.json"
    assert main(["evaluate", "--episodes", "1", "--max-steps", "1", "--json-out", str(json_out)]) == 0
    json.loads(capsys.readouterr().out)
    artifact = json.loads(json_out.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "evaluation"
    assert artifact["schema_version"] == 1
