from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

from artifact_io import write_json_atomic, write_text_atomic


KNOWN_ARTIFACT_TYPES = {"experiment_manifest", "evaluation", "benchmark", "health", "sweep", "checkpoint", "report"}
SUPPORTED_SCHEMA_VERSION = 1


def _finite_float(value: Any) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("numeric value must be finite")
    return numeric


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("integer value must be non-negative")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
        raise ValueError("integer value must be non-negative")
    return int(numeric)


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
    benchmark_count: int = 0
    mean_realtime_factor: float = 0.0
    benchmarks: list[dict[str, Any]] = field(default_factory=list)
    health_count: int = 0
    healthy_count: int = 0
    health_reports: list[dict[str, Any]] = field(default_factory=list)
    artifact_errors: list[str] = field(default_factory=list)
    sweep_count: int = 0
    sweeps: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_count: int = 0
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"artifact_type": "report", "schema_version": 1, **asdict(self)}


class ReportBuilder:
    """Aggregate experiment manifests into analysis-friendly reports."""

    def build(self, manifest_dir: str | Path) -> ReportSummary:
        paths = sorted(Path(manifest_dir).rglob("*.json"))
        artifacts: list[dict[str, Any]] = []
        artifact_errors: list[str] = []
        for path in paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                artifact_errors.append(str(path))
                continue
            if not isinstance(payload, dict):
                artifact_errors.append(str(path))
                continue
            artifact_type = payload.get("artifact_type")
            if artifact_type in KNOWN_ARTIFACT_TYPES and payload.get("schema_version", SUPPORTED_SCHEMA_VERSION) != SUPPORTED_SCHEMA_VERSION:
                artifact_errors.append(str(path))
                continue
            try:
                if artifact_type == "evaluation":
                    for field in ("mean_reward", "mean_steps"):
                        _finite_float(payload.get(field, 0.0))
                    for field in ("episodes", "total_steps"):
                        _nonnegative_int(payload.get(field, 0))
                elif artifact_type == "benchmark":
                    _finite_float(payload.get("simulation_seconds", 0.0))
                    _finite_float(payload.get("wall_seconds", 0.0))
                    _finite_float(payload.get("realtime_factor", 0.0))
                    _nonnegative_int(payload.get("steps", 0))
                elif artifact_type == "health":
                    if "healthy" in payload and not isinstance(payload["healthy"], bool):
                        raise ValueError("health healthy flag must be boolean")
                    if "config" in payload and not isinstance(payload["config"], dict):
                        raise ValueError("health config must be an object")
                    if "dependencies" in payload and not isinstance(payload["dependencies"], dict):
                        raise ValueError("health dependencies must be an object")
                elif artifact_type == "sweep":
                    if not isinstance(payload.get("sweep_id"), str) or not payload["sweep_id"].strip():
                        raise ValueError("sweep_id must be a non-empty string")
                    cases_requested = _nonnegative_int(payload.get("cases_requested", 0))
                    cases_completed = _nonnegative_int(payload.get("cases_completed", 0))
                    resumed_cases = _nonnegative_int(payload.get("resumed_cases", 0))
                    if cases_completed > cases_requested or resumed_cases > cases_requested:
                        raise ValueError("sweep counters exceed requested cases")
                    manifests = payload.get("manifests", [])
                    if not isinstance(manifests, list) or any(not isinstance(item, str) or not item.strip() for item in manifests):
                        raise ValueError("sweep manifests must be a list of non-empty strings")
                elif artifact_type == "checkpoint":
                    if not isinstance(payload.get("run_id"), str) or not payload["run_id"].strip():
                        raise ValueError("checkpoint run_id must be a non-empty string")
                    _nonnegative_int(payload.get("version", 0))
                    _nonnegative_int(payload.get("episode", 0))
                    _nonnegative_int(payload.get("step", 0))
                    _finite_float(payload.get("current_time", 0.0))
                    if not isinstance(payload.get("simulator_state", {}), dict) or not isinstance(payload.get("metrics", {}), dict):
                        raise ValueError("checkpoint state and metrics must be objects")
                    if not isinstance(payload.get("metadata", {}), dict):
                        raise ValueError("checkpoint metadata must be an object")
                elif artifact_type not in {"evaluation", "benchmark", "health", "sweep", "checkpoint", "report"} and "status" in payload:
                    if payload["status"] not in {"running", "completed", "failed"}:
                        raise ValueError("manifest status must be recognized")
                    for field in ("episodes_requested", "episodes_completed", "total_steps"):
                        _nonnegative_int(payload.get(field, 0))
                    metrics = payload.get("metrics", [])
                    if not isinstance(metrics, list):
                        raise ValueError("manifest metrics must be a list")
                    for episode in metrics:
                        if not isinstance(episode, dict):
                            raise ValueError("manifest episode metrics must be objects")
                        _nonnegative_int(episode.get("episode", 0))
                        _finite_float(episode.get("total_reward", 0.0))
                        _nonnegative_int(episode.get("steps", 0))
                        if "terminated" in episode and not isinstance(episode["terminated"], bool):
                            raise ValueError("manifest episode terminated must be boolean")
            except (TypeError, ValueError, OverflowError):
                artifact_errors.append(str(path))
                continue
            artifacts.append(payload)
        evaluations = [item for item in artifacts if item.get("artifact_type") == "evaluation"]
        benchmarks = [item for item in artifacts if item.get("artifact_type") == "benchmark"]
        health_reports = [item for item in artifacts if item.get("artifact_type") == "health"]
        sweeps = [item for item in artifacts if item.get("artifact_type") == "sweep"]
        checkpoints = [item for item in artifacts if item.get("artifact_type") == "checkpoint"]
        manifests = [item for item in artifacts if item.get("artifact_type") not in {"evaluation", "benchmark", "health", "sweep", "checkpoint"} and "status" in item]
        completed = [item for item in manifests if item.get("status") == "completed"]
        failed = [item for item in manifests if item.get("status") == "failed"]
        episodes = [episode for item in manifests for episode in item.get("metrics", [])]
        rewards = [float(episode.get("total_reward", 0.0)) for episode in episodes]
        steps = [int(episode.get("steps", 0)) for episode in episodes]
        evaluation_rewards = [float(item.get("mean_reward", 0.0)) for item in evaluations]
        realtime_factors = [float(item.get("realtime_factor", 0.0)) for item in benchmarks]
        run_rows = [{"run_id": item.get("run_id"), "status": item.get("status"), "episodes_completed": item.get("episodes_completed", 0), "total_steps": item.get("total_steps", 0), "metadata": item.get("metadata", {})} for item in manifests]
        evaluation_rows = [{"config_path": item.get("config_path"), "seed": item.get("seed"), "episodes": item.get("episodes", 0), "total_steps": item.get("total_steps", 0), "mean_reward": item.get("mean_reward", 0.0), "mean_steps": item.get("mean_steps", 0.0)} for item in evaluations]
        benchmark_rows = [{"config_path": item.get("config_path"), "seed": item.get("seed"), "steps": item.get("steps", 0), "simulation_seconds": item.get("simulation_seconds", 0.0), "wall_seconds": item.get("wall_seconds", 0.0), "realtime_factor": item.get("realtime_factor", 0.0)} for item in benchmarks]
        health_rows = [{"config_path": item.get("config_path"), "healthy": item.get("healthy", False), "python": item.get("python"), "platform": item.get("platform")} for item in health_reports]
        sweep_rows = [{"sweep_id": item.get("sweep_id"), "cases_requested": item.get("cases_requested", 0), "cases_completed": item.get("cases_completed", 0), "resumed_cases": item.get("resumed_cases", 0), "manifests": item.get("manifests", [])} for item in sweeps]
        checkpoint_rows = [{"run_id": item.get("run_id"), "episode": item.get("episode", 0), "step": item.get("step", 0), "version": item.get("version", 0), "current_time": item.get("current_time", 0.0)} for item in checkpoints]
        return ReportSummary(len(manifests), len(completed), len(failed), len(episodes), sum(steps), mean(rewards) if rewards else 0.0, mean(steps) if steps else 0.0, run_rows, len(evaluations), mean(evaluation_rewards) if evaluation_rewards else 0.0, evaluation_rows, len(benchmarks), mean(realtime_factors) if realtime_factors else 0.0, benchmark_rows, len(health_reports), sum(item.get("healthy", False) for item in health_reports), health_rows, artifact_errors, len(sweeps), sweep_rows, len(checkpoints), checkpoint_rows)

    def write_json(self, summary: ReportSummary, path: str | Path) -> None:
        write_json_atomic(summary.to_dict(), path)

    def write_markdown(self, summary: ReportSummary, path: str | Path) -> None:
        lines = ["# Experiment Report", "", f"- Manifest count: {summary.manifest_count}", f"- Completed runs: {summary.completed_runs}", f"- Failed runs: {summary.failed_runs}", f"- Total episodes: {summary.total_episodes}", f"- Total steps: {summary.total_steps}", f"- Mean reward: {summary.mean_reward:.4f}", f"- Mean steps per episode: {summary.mean_steps_per_episode:.4f}", f"- Evaluation artifacts: {summary.evaluation_count}", f"- Mean evaluation reward: {summary.mean_evaluation_reward:.4f}", f"- Artifact errors: {len(summary.artifact_errors)}", "", "| Run | Status | Episodes | Steps |", "|---|---|---:|---:|"]
        lines.extend(f"| {run['run_id']} | {run['status']} | {run['episodes_completed']} | {run['total_steps']} |" for run in summary.runs)
        if summary.evaluations:
            lines.extend(["", "| Evaluation config | Seed | Episodes | Steps | Mean reward |", "|---|---:|---:|---:|---:|"])
            lines.extend(f"| {item['config_path']} | {item['seed']} | {item['episodes']} | {item['total_steps']} | {item['mean_reward']:.4f} |" for item in summary.evaluations)
        lines.extend(["", f"- Benchmark artifacts: {summary.benchmark_count}", f"- Mean realtime factor: {summary.mean_realtime_factor:.4f}"])
        if summary.benchmarks:
            lines.extend(["", "| Benchmark config | Seed | Steps | Sim seconds | Wall seconds | Realtime factor |", "|---|---:|---:|---:|---:|---:|"])
            lines.extend(f"| {item['config_path']} | {item['seed']} | {item['steps']} | {item['simulation_seconds']:.4f} | {item['wall_seconds']:.4f} | {item['realtime_factor']:.4f} |" for item in summary.benchmarks)
        lines.extend(["", f"- Health artifacts: {summary.health_count}", f"- Healthy snapshots: {summary.healthy_count}"])
        if summary.health_reports:
            lines.extend(["", "| Health config | Healthy | Python | Platform |", "|---|:---:|---|---|"])
            lines.extend(f"| {item['config_path']} | {item['healthy']} | {item['python']} | {item['platform']} |" for item in summary.health_reports)
        lines.extend(["", f"- Sweep artifacts: {summary.sweep_count}"])
        if summary.sweeps:
            lines.extend(["", "| Sweep | Requested | Completed | Resumed |", "|---|---:|---:|---:|"])
            lines.extend(f"| {item['sweep_id']} | {item['cases_requested']} | {item['cases_completed']} | {item['resumed_cases']} |" for item in summary.sweeps)
        lines.extend(["", f"- Checkpoint artifacts: {summary.checkpoint_count}"])
        if summary.checkpoints:
            lines.extend(["", "| Checkpoint run | Episode | Step | Version | Current time |", "|---|---:|---:|---:|---:|"])
            lines.extend(f"| {item['run_id']} | {item['episode']} | {item['step']} | {item['version']} | {item['current_time']:.4f} |" for item in summary.checkpoints)
        write_text_atomic("\n".join(lines) + "\n", path)
