import json

import pytest
from pathlib import Path

from cli import main
from experiments import ExperimentRunner
from reports import ReportBuilder, ReportSummary


def test_report_summary_artifact_rejects_inconsistent_fields():
    base = ReportSummary(1, 1, 0, 1, 1, 0.0, 1.0, [{}])
    malformed = [
        (ReportSummary(1, 2, 0, 1, 1, 0.0, 1.0, [{}]), "completed and failed runs exceed manifest count"),
        (ReportSummary(1, 1, 0, 1, 1, 0.0, 1.0, []), "report runs count does not match run_count"),
        (ReportSummary(0, 0, 0, 0, 0, 0.0, 0.0, [], health_count=1, healthy_count=2, health_reports=[{}]), "healthy count exceeds health count"),
    ]
    for summary, message in malformed:
        with pytest.raises(ValueError, match=message):
            summary.to_dict()


def test_report_builder_aggregates_manifests(tmp_path: Path):
    ExperimentRunner("config/simulator_config.yaml", "report-a", tmp_path).run(episodes=2, max_steps=2, seed=1)
    ExperimentRunner("config/simulator_config.yaml", "report-b", tmp_path).run(episodes=1, max_steps=3, seed=2)
    summary = ReportBuilder().build(tmp_path)
    assert summary.manifest_count == 2
    assert summary.completed_runs == 2
    assert summary.total_episodes == 3
    assert summary.total_steps == 7
    markdown = tmp_path / "report.md"
    ReportBuilder().write_markdown(summary, markdown)
    assert "| report-a | completed |" in markdown.read_text(encoding="utf-8")


def test_cli_report(capsys, tmp_path: Path):
    ExperimentRunner("config/simulator_config.yaml", "cli-report", tmp_path).run(episodes=1, max_steps=2, seed=3)
    json_out = tmp_path / "summary.json"
    markdown_out = tmp_path / "summary.md"
    assert main(["report", "--manifest-dir", str(tmp_path), "--json-out", str(json_out), "--markdown-out", str(markdown_out)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["artifact_type"] == "report"
    assert payload["schema_version"] == 1
    assert payload["manifest_count"] == 1
    assert json_out.exists()
    assert markdown_out.exists()


def test_evaluation_artifact_is_persisted_and_reported(capsys, tmp_path: Path):
    evaluation_path = tmp_path / "evaluations" / "evaluation.json"
    assert main([
        "evaluate",
        "--episodes",
        "2",
        "--max-steps",
        "3",
        "--seed",
        "9",
        "--reward-per-step",
        "2.5",
        "--json-out",
        str(evaluation_path),
    ]) == 0
    json.loads(capsys.readouterr().out)
    artifact = json.loads(evaluation_path.read_text(encoding="utf-8"))
    assert artifact["artifact_type"] == "evaluation"
    assert artifact["episodes"] == 2
    assert artifact["total_steps"] == 6
    assert artifact["reward_per_step"] == 2.5

    ExperimentRunner("config/simulator_config.yaml", "mixed-run", tmp_path).run(episodes=1, max_steps=2, seed=4)
    summary = ReportBuilder().build(tmp_path)
    assert summary.manifest_count == 1
    assert summary.evaluation_count == 1
    assert summary.mean_evaluation_reward == 7.5
    markdown = tmp_path / "mixed-report.md"
    ReportBuilder().write_markdown(summary, markdown)
    markdown_text = markdown.read_text(encoding="utf-8")
    assert "Evaluation artifacts: 1" in markdown_text
    assert "Mean reward" in markdown_text


def test_evaluation_summary_artifact_contains_reproducibility_metadata():
    from evaluation import EvaluationSummary

    summary = EvaluationSummary(1, 2, 2.0, 0.0, 2.0, 2.0, 2.0, [])
    artifact = summary.to_artifact_dict("config/simulator_config.yaml", 11, 1.0)
    assert artifact["artifact_type"] == "evaluation"
    assert artifact["config_path"] == "config/simulator_config.yaml"
    assert artifact["seed"] == 11
    assert artifact["reward_per_step"] == 1.0


def test_report_builder_aggregates_benchmark_artifacts(tmp_path: Path):
    benchmark_path = tmp_path / "benchmarks" / "baseline.json"
    assert main(["benchmark", "--steps", "2", "--seed", "5", "--json-out", str(benchmark_path)]) == 0
    summary = ReportBuilder().build(tmp_path)
    assert summary.benchmark_count == 1
    assert summary.mean_realtime_factor >= 0.0
    markdown_path = tmp_path / "benchmark-report.md"
    ReportBuilder().write_markdown(summary, markdown_path)
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "Benchmark artifacts: 1" in markdown_text
    assert "Realtime factor" in markdown_text


def test_report_builder_aggregates_health_artifacts(tmp_path: Path):
    health_path = tmp_path / "health" / "snapshot.json"
    assert main(["health", "--json-out", str(health_path)]) == 0
    summary = ReportBuilder().build(tmp_path)
    assert summary.health_count == 1
    assert summary.healthy_count == 1
    markdown_path = tmp_path / "health-report.md"
    ReportBuilder().write_markdown(summary, markdown_path)
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "Health artifacts: 1" in markdown_text
    assert "Healthy snapshots: 1" in markdown_text


def test_report_builder_skips_malformed_health_artifacts(tmp_path: Path):
    (tmp_path / "bad-health.json").write_text(json.dumps({
        "artifact_type": "health",
        "schema_version": 1,
        "healthy": "yes",
        "config": [],
    }), encoding="utf-8")
    summary = ReportBuilder().build(tmp_path)
    assert summary.health_count == 0
    assert summary.healthy_count == 0
    assert summary.artifact_errors == [str(tmp_path / "bad-health.json")]


def test_report_builder_skips_malformed_numeric_artifacts(tmp_path: Path):
    (tmp_path / "bad-evaluation.json").write_text(json.dumps({
        "artifact_type": "evaluation",
        "schema_version": 1,
        "mean_reward": "not-a-number",
    }), encoding="utf-8")
    (tmp_path / "bad-benchmark.json").write_text(json.dumps({
        "artifact_type": "benchmark",
        "schema_version": 1,
        "realtime_factor": "not-a-number",
    }), encoding="utf-8")
    (tmp_path / "bad-manifest.json").write_text(json.dumps({
        "status": "completed",
        "metrics": {"not": "a-list"},
    }), encoding="utf-8")
    (tmp_path / "bad-evaluation-steps.json").write_text(json.dumps({
        "artifact_type": "evaluation",
        "schema_version": 1,
        "episodes": 1.5,
    }), encoding="utf-8")
    (tmp_path / "bad-benchmark-duration.json").write_text(json.dumps({
        "artifact_type": "benchmark",
        "schema_version": 1,
        "wall_seconds": "not-a-number",
    }), encoding="utf-8")
    summary = ReportBuilder().build(tmp_path)
    assert summary.evaluation_count == 0
    assert summary.benchmark_count == 0
    assert summary.manifest_count == 0
    assert len(summary.artifact_errors) == 5


def test_report_builder_skips_malformed_report_rows(tmp_path: Path):
    (tmp_path / "bad-report-row.json").write_text(json.dumps({
        "artifact_type": "report",
        "schema_version": 1,
        "runs": ["not-an-object"],
    }), encoding="utf-8")
    (tmp_path / "bad-report-errors.json").write_text(json.dumps({
        "artifact_type": "report",
        "schema_version": 1,
        "artifact_errors": [1],
    }), encoding="utf-8")
    summary = ReportBuilder().build(tmp_path)
    assert summary.artifact_errors == [
        str(tmp_path / "bad-report-errors.json"),
        str(tmp_path / "bad-report-row.json"),
    ]


def test_report_builder_skips_malformed_report_artifacts(tmp_path: Path):
    (tmp_path / "bad-report-count.json").write_text(json.dumps({
        "artifact_type": "report",
        "schema_version": 1,
        "manifest_count": -1,
    }), encoding="utf-8")
    (tmp_path / "bad-report-lists.json").write_text(json.dumps({
        "artifact_type": "report",
        "schema_version": 1,
        "runs": {},
    }), encoding="utf-8")
    (tmp_path / "bad-report-errors.json").write_text(json.dumps({
        "artifact_type": "report",
        "schema_version": 1,
        "artifact_errors": [1],
    }), encoding="utf-8")
    summary = ReportBuilder().build(tmp_path)
    assert summary.artifact_errors == [
        str(tmp_path / "bad-report-count.json"),
        str(tmp_path / "bad-report-errors.json"),
        str(tmp_path / "bad-report-lists.json"),
    ]


def test_report_builder_skips_malformed_checkpoint_artifacts(tmp_path: Path):
    (tmp_path / "bad-checkpoint-id.json").write_text(json.dumps({
        "artifact_type": "checkpoint",
        "schema_version": 1,
        "run_id": "",
        "version": 1,
        "episode": 0,
        "step": 0,
        "current_time": 0.0,
        "simulator_state": {},
        "metrics": {},
        "metadata": {},
    }), encoding="utf-8")
    (tmp_path / "bad-checkpoint-state.json").write_text(json.dumps({
        "artifact_type": "checkpoint",
        "schema_version": 1,
        "run_id": "bad-state",
        "version": 1,
        "episode": 0,
        "step": 0,
        "current_time": 0.0,
        "simulator_state": [],
        "metrics": {},
        "metadata": {},
    }), encoding="utf-8")
    summary = ReportBuilder().build(tmp_path)
    assert summary.checkpoint_count == 0
    assert summary.artifact_errors == [str(tmp_path / "bad-checkpoint-id.json"), str(tmp_path / "bad-checkpoint-state.json")]


def test_report_builder_skips_malformed_sweep_artifacts(tmp_path: Path):
    (tmp_path / "bad-sweep-counters.json").write_text(json.dumps({
        "artifact_type": "sweep",
        "schema_version": 1,
        "sweep_id": "bad-sweep",
        "cases_requested": 1,
        "cases_completed": 2,
        "resumed_cases": 0,
        "manifests": [],
    }), encoding="utf-8")
    (tmp_path / "bad-sweep-manifests.json").write_text(json.dumps({
        "artifact_type": "sweep",
        "schema_version": 1,
        "sweep_id": "bad-sweep",
        "cases_requested": 1,
        "cases_completed": 0,
        "resumed_cases": 0,
        "manifests": [""],
    }), encoding="utf-8")
    summary = ReportBuilder().build(tmp_path)
    assert summary.sweep_count == 0
    assert summary.artifact_errors == [str(tmp_path / "bad-sweep-counters.json"), str(tmp_path / "bad-sweep-manifests.json")]


def test_report_builder_skips_malformed_manifest_fields(tmp_path: Path):
    (tmp_path / "bad-status.json").write_text(json.dumps({
        "status": "unknown",
        "episodes_requested": 1,
        "metrics": [],
    }), encoding="utf-8")
    (tmp_path / "bad-counter.json").write_text(json.dumps({
        "status": "completed",
        "episodes_requested": -1,
        "metrics": [],
    }), encoding="utf-8")
    (tmp_path / "bad-episode.json").write_text(json.dumps({
        "status": "completed",
        "metrics": [{"episode": 0, "steps": 1, "total_reward": 0.0, "terminated": "yes"}],
    }), encoding="utf-8")
    summary = ReportBuilder().build(tmp_path)
    assert summary.manifest_count == 0
    assert len(summary.artifact_errors) == 3


def test_report_builder_records_malformed_artifacts_without_failing(tmp_path: Path):
    broken = tmp_path / "broken.json"
    broken.write_text("{not valid json", encoding="utf-8")
    non_mapping = tmp_path / "list.json"
    non_mapping.write_text("[]", encoding="utf-8")
    summary = ReportBuilder().build(tmp_path)
    assert summary.manifest_count == 0
    assert len(summary.artifact_errors) == 2
    markdown_path = tmp_path / "resilient-report.md"
    ReportBuilder().write_markdown(summary, markdown_path)
    assert "Artifact errors: 2" in markdown_path.read_text(encoding="utf-8")


def test_report_builder_aggregates_sweep_summaries(tmp_path: Path):
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(json.dumps([{"run_id": "report-sweep-a", "seed": 6, "episodes": 1, "max_steps": 1}]), encoding="utf-8")
    sweep_path = tmp_path / "sweeps" / "summary.json"
    assert main(["sweep", "--cases", str(cases_path), "--sweep-id", "report-sweep", "--manifest-dir", str(tmp_path / "runs"), "--json-out", str(sweep_path)]) == 0
    summary = ReportBuilder().build(tmp_path)
    assert summary.sweep_count == 1
    assert summary.sweeps[0]["sweep_id"] == "report-sweep"
    markdown_path = tmp_path / "sweep-report.md"
    ReportBuilder().write_markdown(summary, markdown_path)
    markdown_text = markdown_path.read_text(encoding="utf-8")
    assert "Sweep artifacts: 1" in markdown_text
    assert "report-sweep" in markdown_text


def test_report_builder_records_unsupported_known_schema(tmp_path: Path):
    future = tmp_path / "future.json"
    future.write_text(json.dumps({
        "artifact_type": "benchmark",
        "schema_version": 99,
        "steps": 1,
        "realtime_factor": 1.0,
    }), encoding="utf-8")
    summary = ReportBuilder().build(tmp_path)
    assert summary.benchmark_count == 0
    assert summary.artifact_errors == [str(future)]


def test_report_builder_aggregates_checkpoint_artifacts(capsys, tmp_path: Path):
    runs_dir = tmp_path / "runs"
    assert main(["run", "--run-id", "report-checkpoint", "--manifest-dir", str(runs_dir), "--episodes", "1", "--max-steps", "1", "--checkpoint-every", "1"]) == 0
    capsys.readouterr()
    summary = ReportBuilder().build(tmp_path)
    assert summary.checkpoint_count == 1
    assert summary.checkpoints[0]["run_id"] == "report-checkpoint"
    markdown_path = tmp_path / "checkpoint-report.md"
    ReportBuilder().write_markdown(summary, markdown_path)
    assert "Checkpoint artifacts: 1" in markdown_path.read_text(encoding="utf-8")
