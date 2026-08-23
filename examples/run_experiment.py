import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml
from experiments import ExperimentRunner


if __name__ == "__main__":
    config = yaml.safe_load(Path("config/experiment_config.yaml").read_text(encoding="utf-8"))["experiment"]
    runner = ExperimentRunner(config["simulator_config"], config["run_id"], config["manifest_dir"])
    manifest = runner.run(config["episodes"], config["max_steps"], config["seed"], config["checkpoint_every"])
    print(f"Run {manifest.run_id}: {manifest.status}, {manifest.total_steps} steps")
