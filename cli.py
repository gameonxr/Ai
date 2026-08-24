from __future__ import annotations

import argparse
import json
import math
import sys
from importlib.resources import files
from pathlib import Path

from artifact_io import write_json_atomic
from benchmarks import run_benchmark
from config_validation import ConfigurationValidator
from experiments import ExperimentRunner, SweepRunner
from evaluation import Evaluator
from health import collect_health, to_artifact_dict as health_to_artifact_dict
from reports import ReportBuilder
from simulator import Simulator
from training import RandomTorquePolicy


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _finite_float(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a finite number") from error
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("must be a finite number")
    return parsed


def default_simulator_config() -> str:
    local = Path("config/simulator_config.yaml")
    if local.exists():
        return str(local)
    return str(files("ai_body_simulator_resources").joinpath("config/simulator_config.yaml"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Body Simulator operator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate the canonical simulator YAML")
    validate.add_argument("path", nargs="?", default=default_simulator_config())

    benchmark = subparsers.add_parser("benchmark", help="run a deterministic step benchmark")
    benchmark.add_argument("--config", default=default_simulator_config())
    benchmark.add_argument("--steps", type=_positive_int, default=1000)
    benchmark.add_argument("--seed", type=int, default=42)
    benchmark.add_argument("--json-out", help="write a metadata-rich benchmark artifact")

    health = subparsers.add_parser("health", help="report runtime and dependency health")
    health.add_argument("--config", default=default_simulator_config())
    health.add_argument("--json-out", help="write a health snapshot artifact")

    report = subparsers.add_parser("report", help="aggregate experiment manifests")
    report.add_argument("--manifest-dir", default="artifacts/runs")
    report.add_argument("--json-out")
    report.add_argument("--markdown-out")
    report.add_argument("--strict", action="store_true", help="fail if malformed or non-object JSON artifacts are found")

    evaluate = subparsers.add_parser("evaluate", help="evaluate the seeded baseline policy")
    evaluate.add_argument("--config", default=default_simulator_config())
    evaluate.add_argument("--episodes", type=_positive_int, default=5)
    evaluate.add_argument("--max-steps", type=_positive_int, default=100)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--reward-per-step", type=_finite_float, default=1.0)
    evaluate.add_argument("--json-out", help="write a metadata-rich evaluation artifact")

    sweep = subparsers.add_parser("sweep", help="run a deterministic list of experiment cases")
    sweep.add_argument("--cases", required=True, help="JSON file containing a non-empty list of cases")
    sweep.add_argument("--sweep-id", default="sweep")
    sweep.add_argument("--manifest-dir", default="artifacts/runs")
    sweep.add_argument("--json-out", help="write the sweep summary artifact")
    sweep.add_argument("--resume", action="store_true", help="reuse matching completed manifests")

    run = subparsers.add_parser("run", help="run seeded experiment episodes")
    run.add_argument("--config", default=default_simulator_config())
    run.add_argument("--run-id", default="experiment")
    run.add_argument("--manifest-dir", default="artifacts/runs")
    run.add_argument("--episodes", type=_positive_int, default=1)
    run.add_argument("--max-steps", type=_positive_int, default=100)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--checkpoint-every", type=_nonnegative_int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        try:
            report = ConfigurationValidator().validate(args.path)
        except (OSError, ValueError) as error:
            print(f"Configuration validation failed: {error}", file=sys.stderr)
            return 1
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.valid else 1
    if args.command == "benchmark":
        try:
            benchmark_result = run_benchmark(args.config, args.steps, args.seed)
            if args.json_out:
                write_json_atomic(benchmark_result.to_artifact_dict(args.config, args.seed), args.json_out)
        except (OSError, ValueError) as error:
            print(f"Benchmark failed: {error}", file=sys.stderr)
            return 1
        print(json.dumps(benchmark_result.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "health":
        try:
            report = collect_health(args.config)
            if args.json_out:
                write_json_atomic(health_to_artifact_dict(report, args.config), args.json_out)
        except (OSError, ValueError) as error:
            print(f"Health check failed: {error}", file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["healthy"] else 1
    if args.command == "report":
        manifest_dir = Path(args.manifest_dir)
        if not manifest_dir.exists() or not manifest_dir.is_dir():
            print(f"Report failed: manifest directory not found: {manifest_dir}", file=sys.stderr)
            return 1
        try:
            summary = ReportBuilder().build(manifest_dir)
            if args.json_out:
                ReportBuilder().write_json(summary, args.json_out)
            if args.markdown_out:
                ReportBuilder().write_markdown(summary, args.markdown_out)
        except (OSError, ValueError) as error:
            print(f"Report failed: {error}", file=sys.stderr)
            return 1
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 1 if args.strict and summary.artifact_errors else 0
    if args.command == "evaluate":
        simulator = None
        try:
            simulator = Simulator(args.config)
            policy = RandomTorquePolicy(list(simulator.actuators), seed=args.seed)
            summary = Evaluator(simulator, policy, reward_fn=lambda observation, action: args.reward_per_step).run(args.episodes, args.max_steps, args.seed)
            if args.json_out:
                write_json_atomic(summary.to_artifact_dict(args.config, args.seed, args.reward_per_step), args.json_out)
        except (OSError, ValueError) as error:
            print(f"Evaluation failed: {error}", file=sys.stderr)
            return 1
        finally:
            if simulator is not None:
                simulator.shutdown()
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "sweep":
        try:
            result = SweepRunner(args.sweep_id, args.manifest_dir, resume=args.resume).run_file(args.cases)
            if args.json_out:
                result.write_json(args.json_out)
        except (OSError, ValueError) as error:
            print(f"Sweep failed: {error}", file=sys.stderr)
            return 1
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    try:
        manifest = ExperimentRunner(args.config, args.run_id, args.manifest_dir).run(args.episodes, args.max_steps, args.seed, args.checkpoint_every)
    except (OSError, ValueError) as error:
        print(f"Run failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"run_id": manifest.run_id, "status": manifest.status, "episodes_completed": manifest.episodes_completed, "total_steps": manifest.total_steps}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
