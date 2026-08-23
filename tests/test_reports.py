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
