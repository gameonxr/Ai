from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

from artifact_io import write_json_atomic
from .runner import ExperimentRunner, RunManifest


@dataclass
class SweepResult:
    """Persisted result collection for one deterministic sweep."""

    sweep_id: str
    cases_requested: int
    manifests: list[RunManifest]
    resumed_cases: int = 0

    @property
    def cases_completed(self) -> int:
        return sum(manifest.status == "completed" for manifest in self.manifests)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_type": "sweep",
            "schema_version": 1,
            "sweep_id": self.sweep_id,
            "cases_requested": self.cases_requested,
            "cases_completed": self.cases_completed,
            "resumed_cases": self.resumed_cases,
            "manifests": [manifest.run_id for manifest in self.manifests],
        }

    def write_json(self, path: str | Path) -> None:
        write_json_atomic(self.to_dict(), path)


class SweepRunner:
    """Run a finite list of reproducible experiment cases in declaration order."""

    def __init__(self, sweep_id: str, manifest_dir: str | Path = "artifacts/runs", resume: bool = False):
        if not isinstance(sweep_id, str) or not sweep_id.strip():
            raise ValueError("sweep_id must be a non-empty string")
        self.sweep_id = sweep_id
        self.manifest_dir = Path(manifest_dir)
        self.resume = resume

    @staticmethod
    def load_cases(path: str | Path) -> list[dict[str, Any]]:
        cases_path = Path(path)
        try:
            payload = json.loads(cases_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise ValueError(f"Unable to read sweep cases {cases_path}: {error}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid sweep cases JSON in {cases_path}: {error.msg}") from error
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
        resumed_cases = 0
        seen_ids: set[str] = set()
        for index, case in enumerate(case_list):
            if not isinstance(case, dict):
                raise ValueError(f"sweep case {index} must be a JSON object")
            run_id = case.get("run_id", f"{self.sweep_id}-{index + 1}")
            if not isinstance(run_id, str) or not run_id.strip():
                raise ValueError(f"sweep case {index} run_id must be a non-empty string")
            if run_id in seen_ids:
                raise ValueError(f"duplicate sweep run_id: {run_id}")
            seen_ids.add(run_id)
            simulator_config = case.get("config", case.get("simulator_config", "config/simulator_config.yaml"))
            if not isinstance(simulator_config, str) or not simulator_config.strip():
                raise ValueError(f"sweep case {run_id} config must be a non-empty string")
            episodes = self._positive_int(case, "episodes", 1)
            max_steps = self._positive_int(case, "max_steps", 100)
            seed = self._optional_int(case, "seed")
            checkpoint_every = self._nonnegative_int(case, "checkpoint_every", 0)
            metadata = {
                "sweep_id": self.sweep_id,
                "sweep_index": index,
                "parameters": {key: value for key, value in case.items() if key not in {"run_id", "config", "simulator_config", "episodes", "max_steps", "seed", "checkpoint_every"}},
            }
            existing_path = self.manifest_dir / f"{run_id}.json"
            if self.resume and existing_path.exists():
                existing = self._load_manifest(existing_path)
                expected_metadata = metadata
                if existing.status == "completed":
                    if existing.config_path != simulator_config or existing.episodes_requested != episodes or existing.max_steps_requested != max_steps or existing.seed != seed or existing.metadata != expected_metadata:
                        raise ValueError(f"existing completed manifest does not match sweep case: {run_id}")
                    manifests.append(existing)
                    resumed_cases += 1
                    continue
            runner = ExperimentRunner(simulator_config, run_id, self.manifest_dir, metadata=metadata)
            manifests.append(runner.run(episodes, max_steps, seed, checkpoint_every))
        return SweepResult(self.sweep_id, len(case_list), manifests, resumed_cases)

    @staticmethod
    def _positive_int(case: dict[str, Any], key: str, default: int) -> int:
        value = case.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"sweep case {case.get('run_id', '<generated>')} {key} must be a positive integer")
        return value

    @staticmethod
    def _nonnegative_int(case: dict[str, Any], key: str, default: int) -> int:
        value = case.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"sweep case {case.get('run_id', '<generated>')} {key} must be a non-negative integer")
        return value

    @staticmethod
    def _optional_int(case: dict[str, Any], key: str) -> int | None:
        value = case.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"sweep case {case.get('run_id', '<generated>')} {key} must be an integer or null")
        return value

    @staticmethod
    def _load_manifest(path: Path) -> RunManifest:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("manifest payload must be an object")
            if payload.setdefault("artifact_type", "experiment_manifest") != "experiment_manifest":
                raise ValueError("unsupported manifest artifact type")
            schema_version = int(payload.setdefault("schema_version", 1))
            if schema_version != 1:
                raise ValueError("unsupported manifest schema version")
            payload["schema_version"] = schema_version
            required_fields = ("run_id", "status", "started_at", "finished_at", "config_path", "episodes_requested", "episodes_completed", "total_steps")
            missing_fields = [field for field in required_fields if field not in payload]
            if missing_fields:
                raise ValueError(f"manifest missing required fields: {', '.join(missing_fields)}")
            if not isinstance(payload.get("metrics", []), list):
                raise ValueError("manifest metrics must be a list")
            if not isinstance(payload.get("metadata", {}), dict):
                raise ValueError("manifest metadata must be an object")
            return RunManifest(**payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid existing manifest: {path}") from error

    def run_file(self, path: str | Path) -> SweepResult:
        return self.run(self.load_cases(path))
