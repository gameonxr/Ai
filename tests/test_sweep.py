import json
from pathlib import Path

import pytest

from cli import main
from experiments import SweepRunner
from reports import ReportBuilder


def test_sweep_runner_writes_ordered_manifests_and_metadata(tmp_path: Path):
    cases = [
        {"run_id": "sweep-a", "seed": 1, "episodes": 1, "max_steps": 2, "learning_rate": 0.1},
        {"run_id": "sweep-b", "seed": 2, "episodes": 2, "max_steps": 1, "learning_rate": 0.2},
    ]
    result = SweepRunner("lr-sweep", tmp_path).run(cases)
    assert result.sweep_id == "lr-sweep"
    assert result.cases_requested == 2
    assert result.cases_completed == 2
    assert [manifest.run_id for manifest in result.manifests] == ["sweep-a", "sweep-b"]
    payload = json.loads((tmp_path / "sweep-a.json").read_text(encoding="utf-8"))
    assert payload["metadata"]["sweep_id"] == "lr-sweep"
    assert payload["metadata"]["sweep_index"] == 0
    assert payload["metadata"]["parameters"] == {"learning_rate": 0.1}


def test_sweep_runner_loads_json_cases_and_cli(capsys, tmp_path: Path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([
        {"run_id": "cli-sweep-a", "seed": 3, "episodes": 1, "max_steps": 1},
        {"run_id": "cli-sweep-b", "seed": 4, "episodes": 1, "max_steps": 1},
    ]), encoding="utf-8")
    summary_path = tmp_path / "sweep-summary.json"
    assert main(["sweep", "--cases", str(cases_path), "--sweep-id", "cli-sweep", "--manifest-dir", str(tmp_path / "runs"), "--json-out", str(summary_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_type"] == "sweep"
    assert payload["cases_requested"] == 2
    assert payload["cases_completed"] == 2
    assert (tmp_path / "runs" / "cli-sweep-a.json").exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["artifact_type"] == "sweep"

    report = ReportBuilder().build(tmp_path / "runs")
    assert report.runs[0]["metadata"]["sweep_id"] == "cli-sweep"


def test_sweep_runner_rejects_empty_and_duplicate_cases(tmp_path: Path):
    runner = SweepRunner("invalid", tmp_path)
    with pytest.raises(ValueError):
        runner.run([])
    with pytest.raises(ValueError):
        runner.run([{"run_id": "same"}, {"run_id": "same"}])
    empty_path = tmp_path / "empty.json"
    empty_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        runner.run_file(empty_path)


def test_sweep_resume_reuses_matching_completed_manifests(tmp_path: Path):
    cases = [{"run_id": "resume-a", "seed": 12, "episodes": 1, "max_steps": 1, "temperature": 0.5}]
    first = SweepRunner("resume-sweep", tmp_path).run(cases)
    second = SweepRunner("resume-sweep", tmp_path, resume=True).run(cases)
    assert first.cases_completed == 1
    assert second.cases_completed == 1
    assert second.resumed_cases == 1
    assert second.manifests[0].started_at == first.manifests[0].started_at


def test_sweep_resume_rejects_mismatched_completed_manifest(tmp_path: Path):
    SweepRunner("resume-sweep", tmp_path).run([{"run_id": "resume-a", "seed": 12, "episodes": 1, "max_steps": 1}])
    with pytest.raises(ValueError, match="does not match"):
        SweepRunner("resume-sweep", tmp_path, resume=True).run([{"run_id": "resume-a", "seed": 12, "episodes": 2, "max_steps": 1}])


def test_sweep_runner_rejects_invalid_case_types(tmp_path: Path):
    runner = SweepRunner("strict", tmp_path)
    invalid_cases = [
        [{"run_id": "", "episodes": 1}],
        [{"run_id": "bad", "episodes": 0}],
        [{"run_id": "bad", "max_steps": 1.0}],
        [{"run_id": "bad", "seed": True}],
        [{"run_id": "bad", "checkpoint_every": -1}],
        [{"run_id": "bad", "config": ""}],
    ]
    for cases in invalid_cases:
        with pytest.raises(ValueError):
            runner.run(cases)


def test_sweep_runner_rejects_invalid_sweep_id(tmp_path: Path):
    with pytest.raises(ValueError):
        SweepRunner("", tmp_path)
