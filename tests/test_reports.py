import json
from pathlib import Path

from cli import main
from experiments import ExperimentRunner
from reports import ReportBuilder


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
