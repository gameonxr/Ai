import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from benchmarks import run_benchmark


if __name__ == "__main__":
    config = yaml.safe_load(Path("config/benchmark_config.yaml").read_text(encoding="utf-8"))["benchmark"]
    result = run_benchmark(config["simulator_config"], config["steps"], config["seed"])
    print(f"{result.steps} steps in {result.wall_seconds:.4f}s ({result.realtime_factor:.2f}x real time)")
