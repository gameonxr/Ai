from __future__ import annotations

import argparse
import json
from importlib.resources import files
from pathlib import Path

from benchmarks import run_benchmark
from config_validation import ConfigurationValidator
from experiments import ExperimentRunner, SweepRunner
from evaluation import Evaluator
from health import collect_health
from reports import ReportBuilder
from simulator import Simulator
from training import RandomTorquePolicy


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

    report = subparsers.add_parser("report", help="aggregate experiment manifests")
    report.add_argument("--manifest-dir", default="artifacts/runs")
    report.add_argument("--json-out")
    report.add_argument("--markdown-out")

    evaluate = subparsers.add_parser("evaluate", help="evaluate the seeded baseline policy")
    evaluate.add_argument("--config", default=default_simulator_config())
    evaluate.add_argument("--episodes", type=int, default=5)
    evaluate.add_argument("--max-steps", type=int, default=100)
    evaluate.add_argument("--seed", type=int, default=42)
    evaluate.add_argument("--reward-per-step", type=float, default=1.0)
    evaluate.add_argument("--json-out", help="write a metadata-rich evaluation artifact")

    sweep = subparsers.add_parser("sweep", help="run a deterministic list of experiment cases")
    sweep.add_argument("--cases", required=True, help="JSON file containing a non-empty list of cases")
    sweep.add_argument("--sweep-id", default="sweep")
    sweep.add_argument("--manifest-dir", default="artifacts/runs")

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
    if args.command == "report":
        summary = ReportBuilder().build(args.manifest_dir)
        if args.json_out:
            ReportBuilder().write_json(summary, args.json_out)
        if args.markdown_out:
            ReportBuilder().write_markdown(summary, args.markdown_out)
        print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.command == "evaluate":
        simulator = Simulator(args.config)
        try:
            policy = RandomTorquePolicy(list(simulator.actuators), seed=args.seed)
            summary = Evaluator(simulator, policy, reward_fn=lambda observation, action: args.reward_per_step).run(args.episodes, args.max_steps, args.seed)
            if args.json_out:
                output = Path(args.json_out)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(summary.to_artifact_dict(args.config, args.seed, args.reward_per_step), indent=2, sort_keys=True), encoding="utf-8")
            print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
            return 0
        finally:
            simulator.shutdown()
    if args.command == "sweep":
        result = SweepRunner(args.sweep_id, args.manifest_dir).run_file(args.cases)
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    manifest = ExperimentRunner(args.config, args.run_id, args.manifest_dir).run(args.episodes, args.max_steps, args.seed, args.checkpoint_every)
    print(json.dumps({"run_id": manifest.run_id, "status": manifest.status, "episodes_completed": manifest.episodes_completed, "total_steps": manifest.total_steps}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
