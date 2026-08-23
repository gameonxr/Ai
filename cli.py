from __future__ import annotations

import argparse
import json

from benchmarks import run_benchmark
from config_validation import ConfigurationValidator
from experiments import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI Body Simulator operator CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate the canonical simulator YAML")
    validate.add_argument("path", nargs="?", default="config/simulator_config.yaml")

    benchmark = subparsers.add_parser("benchmark", help="run a deterministic step benchmark")
    benchmark.add_argument("--config", default="config/simulator_config.yaml")
    benchmark.add_argument("--steps", type=int, default=1000)
    benchmark.add_argument("--seed", type=int, default=42)

    run = subparsers.add_parser("run", help="run seeded experiment episodes")
    run.add_argument("--config", default="config/simulator_config.yaml")
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
    manifest = ExperimentRunner(args.config, args.run_id, args.manifest_dir).run(args.episodes, args.max_steps, args.seed, args.checkpoint_every)
    print(json.dumps({"run_id": manifest.run_id, "status": manifest.status, "episodes_completed": manifest.episodes_completed, "total_steps": manifest.total_steps}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
