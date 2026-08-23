from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from .runner import ExperimentRunner, RunManifest


@dataclass
class SweepResult:
    """Persisted result collection for one deterministic sweep."""

    sweep_id: str
    cases_requested: int
    manifests: list[RunManifest]

    @property
    def cases_completed(self) -> int:
        return sum(manifest.status == "completed" for manifest in self.manifests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "sweep",
            "sweep_id": self.sweep_id,
            "cases_requested": self.cases_requested,
            "cases_completed": self.cases_completed,
            "manifests": [manifest.run_id for manifest in self.manifests],
        }


class SweepRunner:
    """Run a finite list of reproducible experiment cases in declaration order."""

    def __init__(self, sweep_id: str, manifest_dir: str | Path = "artifacts/runs"):
        self.sweep_id = sweep_id
        self.manifest_dir = Path(manifest_dir)

    @staticmethod
    def load_cases(path: str | Path) -> list[dict[str, Any]]:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not payload:
            raise ValueError("sweep cases must be a non-empty JSON list")
        if any(not isinstance(case, dict) for case in payload):
            raise ValueError("each sweep case must be a JSON object")
        return payload

    def run(self, cases: Iterable[dict[str, Any]]) -> SweepResult:
        case_list = list(cases)
        if not case_list:
            raise ValueError("sweep cases must be a non-empty collection")
        manifests: list[RunManifest] = []
        seen_ids: set[str] = set()
        for index, case in enumerate(case_list):
            run_id = str(case.get("run_id", f"{self.sweep_id}-{index + 1}"))
            if run_id in seen_ids:
                raise ValueError(f"duplicate sweep run_id: {run_id}")
            seen_ids.add(run_id)
            simulator_config = str(case.get("config", case.get("simulator_config", "config/simulator_config.yaml")))
            episodes = int(case.get("episodes", 1))
            max_steps = int(case.get("max_steps", 100))
            seed = case.get("seed")
            checkpoint_every = int(case.get("checkpoint_every", 0))
            metadata = {
                "sweep_id": self.sweep_id,
                "sweep_index": index,
                "parameters": {key: value for key, value in case.items() if key not in {"run_id", "config", "simulator_config", "episodes", "max_steps", "seed", "checkpoint_every"}},
            }
            runner = ExperimentRunner(simulator_config, run_id, self.manifest_dir, metadata=metadata)
            manifests.append(runner.run(episodes, max_steps, seed, checkpoint_every))
        return SweepResult(self.sweep_id, len(case_list), manifests)

    def run_file(self, path: str | Path) -> SweepResult:
        return self.run(self.load_cases(path))
