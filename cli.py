from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

from benchmarks import run_benchmark
from config_validation import ConfigurationValidator
from experiments import ExperimentRunner
from health import collect_health


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
    benchmark.add_argument("--steps", type=int, default=1000)
    benchmark.add_argument("--seed", type=int, default=42)

    health = subparsers.add_parser("health", help="report runtime and dependency health")
    health.add_argument("--config", default=default_simulator_config())

    run = subparsers.add_parser("run", help="run seeded experiment episodes")
    run.add_argument("--config", default=default_simulator_config())
    run.add_argument("--run-id", default="experiment")
    run.add_argument("--manifest-dir", default="artifacts/runs")
    run.add_argument("--episodes", type=int, default=1)
    run.add_argument("--max-steps", type=int, default=100)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--checkpoint-every", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        report = ConfigurationValidator().validate(args.path)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.valid else 1
    if args.command == "benchmark":
        print(json.dumps(run_benchmark(args.config, args.steps, args.seed).to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "health":
        report = collect_health(args.config)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["healthy"] else 1
    manifest = ExperimentRunner(args.config, args.run_id, args.manifest_dir).run(args.episodes, args.max_steps, args.seed, args.checkpoint_every)
    print(json.dumps({"run_id": manifest.run_id, "status": manifest.status, "episodes_completed": manifest.episodes_completed, "total_steps": manifest.total_steps}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
