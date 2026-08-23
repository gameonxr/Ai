from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable

from checkpointing import CheckpointManager
from simulator import Simulator
from training import RandomTorquePolicy, Trainer


@dataclass
class RunManifest:
    run_id: str
    status: str
    started_at: str
    finished_at: str | None
    config_path: str
    episodes_requested: int
    episodes_completed: int
    total_steps: int
    metrics: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ExperimentRunner:
    """Run repeatable experiments and persist lifecycle metadata."""

    def __init__(self, simulator_config: str = "config/simulator_config.yaml", run_id: str = "experiment", manifest_dir: str | Path = "artifacts/runs", policy_factory: Callable | None = None, metadata: dict[str, Any] | None = None):
        self.simulator_config = simulator_config
        self.run_id = run_id
        self.manifest_dir = Path(manifest_dir)
        self.policy_factory = policy_factory
        self.metadata = dict(metadata or {})

    def run(self, episodes: int = 1, max_steps: int = 100, seed: int | None = None, checkpoint_every: int = 0) -> RunManifest:
        if episodes < 1 or max_steps < 1:
            raise ValueError("episodes and max_steps must be >= 1")
        started = datetime.now(timezone.utc).isoformat()
        manifest = RunManifest(self.run_id, "running", started, None, self.simulator_config, episodes, 0, 0, metadata=self.metadata.copy())
        simulator = Simulator(self.simulator_config)
        policy = self.policy_factory(simulator) if self.policy_factory else RandomTorquePolicy(list(simulator.actuators), seed=seed)
        trainer = Trainer(simulator, policy, reward_fn=lambda observation, action: 0.0)
        try:
            for episode in range(episodes):
                _, metrics = trainer.run_episode(max_steps=max_steps, seed=None if seed is None else seed + episode)
                manifest.metrics.append({"episode": episode, "steps": metrics.steps, "total_reward": metrics.total_reward, "terminated": metrics.terminated})
                manifest.episodes_completed += 1
                manifest.total_steps += metrics.steps
                if checkpoint_every and manifest.episodes_completed % checkpoint_every == 0:
                    path = self.manifest_dir / f"{self.run_id}.checkpoint.json"
                    CheckpointManager.save(simulator, path, self.run_id, {"episode": episode})
                    manifest.checkpoint_path = str(path)
            manifest.status = "completed"
        except Exception:
            manifest.status = "failed"
            raise
        finally:
            manifest.finished_at = datetime.now(timezone.utc).isoformat()
            self._save_manifest(manifest)
            trainer.close()
        return manifest

    def _save_manifest(self, manifest: RunManifest) -> None:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        path = self.manifest_dir / f"{manifest.run_id}.json"
        path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True), encoding="utf-8")
