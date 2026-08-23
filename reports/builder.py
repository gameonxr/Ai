from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class ReportSummary:
    manifest_count: int
    completed_runs: int
    failed_runs: int
    total_episodes: int
    total_steps: int
    mean_reward: float
    mean_steps_per_episode: float
    runs: list[dict[str, Any]]
    evaluation_count: int = 0
    mean_evaluation_reward: float = 0.0
    evaluations: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ReportBuilder:
    """Aggregate experiment manifests into analysis-friendly reports."""

    def build(self, manifest_dir: str | Path) -> ReportSummary:
        paths = sorted(Path(manifest_dir).rglob("*.json"))
        artifacts = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
        evaluations = [item for item in artifacts if item.get("artifact_type") == "evaluation"]
        manifests = [item for item in artifacts if item.get("artifact_type") != "evaluation" and "status" in item]
        completed = [item for item in manifests if item.get("status") == "completed"]
        failed = [item for item in manifests if item.get("status") == "failed"]
        episodes = [episode for item in manifests for episode in item.get("metrics", [])]
        rewards = [float(episode.get("total_reward", 0.0)) for episode in episodes]
        steps = [int(episode.get("steps", 0)) for episode in episodes]
        evaluation_rewards = [float(item.get("mean_reward", 0.0)) for item in evaluations]
        run_rows = [{"run_id": item.get("run_id"), "status": item.get("status"), "episodes_completed": item.get("episodes_completed", 0), "total_steps": item.get("total_steps", 0), "metadata": item.get("metadata", {})} for item in manifests]
        evaluation_rows = [{"config_path": item.get("config_path"), "seed": item.get("seed"), "episodes": item.get("episodes", 0), "total_steps": item.get("total_steps", 0), "mean_reward": item.get("mean_reward", 0.0), "mean_steps": item.get("mean_steps", 0.0)} for item in evaluations]
        return ReportSummary(len(manifests), len(completed), len(failed), len(episodes), sum(steps), mean(rewards) if rewards else 0.0, mean(steps) if steps else 0.0, run_rows, len(evaluations), mean(evaluation_rewards) if evaluation_rewards else 0.0, evaluation_rows)

    def write_json(self, summary: ReportSummary, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary.to_dict(), indent=2, sort_keys=True), encoding="utf-8")

    def write_markdown(self, summary: ReportSummary, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Experiment Report", "", f"- Manifest count: {summary.manifest_count}", f"- Completed runs: {summary.completed_runs}", f"- Failed runs: {summary.failed_runs}", f"- Total episodes: {summary.total_episodes}", f"- Total steps: {summary.total_steps}", f"- Mean reward: {summary.mean_reward:.4f}", f"- Mean steps per episode: {summary.mean_steps_per_episode:.4f}", f"- Evaluation artifacts: {summary.evaluation_count}", f"- Mean evaluation reward: {summary.mean_evaluation_reward:.4f}", "", "| Run | Status | Episodes | Steps |", "|---|---|---:|---:|"]
        lines.extend(f"| {run['run_id']} | {run['status']} | {run['episodes_completed']} | {run['total_steps']} |" for run in summary.runs)
        if summary.evaluations:
            lines.extend(["", "| Evaluation config | Seed | Episodes | Steps | Mean reward |", "|---|---:|---:|---:|---:|"])
            lines.extend(f"| {item['config_path']} | {item['seed']} | {item['episodes']} | {item['total_steps']} | {item['mean_reward']:.4f} |" for item in summary.evaluations)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
