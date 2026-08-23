# AI Body Simulator

AI Body Simulator is a **brain-agnostic, deterministic humanoid simulation foundation**. The simulator exposes sensor observations and accepts validated actions through stable APIs; a future LLM, reinforcement-learning agent, or rule-based controller can replace the included dummy brain without changing physics, body, sensors, or actuators.

> **Core loop:** observe → think → act → physics → observe.

## Current scope

This repository implements the Phase 1 foundation described in `AI_BODY_SIMULATOR_SPEC.md`: a configurable 13-DOF humanoid body, YAML configuration loading, a physics abstraction with MuJoCo/Bullet-compatible backend boundaries, modular proprioception/IMU/vision/touch sensors, validated motor actions, deterministic stepping, pause/resume, examples, and pytest coverage. Phase 2 adds an opt-in, headless Matplotlib renderer and a stable renderer interface for debugging and visualization. The next modular phase adds a policy interface, simulator training adapter, seeded baseline policy, episode metrics, and JSONL rollout persistence. A multi-agent coordinator now synchronizes independent simulator instances while preserving the single-agent APIs. Recording beyond rollouts and external AI connections remain deferred.

## Quick start

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pytest -q
python examples/dummy_brain_example.py
python examples/render_simulation.py
```

The default configuration is `config/simulator_config.yaml`. It uses the MuJoCo backend boundary with a portable deterministic implementation for headless execution; the public backend interface is isolated so a native engine implementation can be introduced without changing the simulator or brain APIs. Rendering is disabled by default and can be enabled in YAML with `rendering.enabled: true`.

To render the current simulator state from Python, first set `rendering.enabled: true` in `config/simulator_config.yaml`, then call:

```python
sim = Simulator("config/simulator_config.yaml")
sim.reset(seed=42)
sim.render("artifacts/body.png")
sim.shutdown()
```

## Public API

```python
from brain import DummyBrain
from simulator import Simulator

sim = Simulator("config/simulator_config.yaml")
sim.set_brain(DummyBrain())
sim.reset(seed=42)
observation = sim.step()
print(observation.proprioception["joint_positions"])
sim.pause()
sim.resume()
sim.shutdown()
```

The brain receives only `core.Observation` and returns only `core.Action`. `ActionValidator` rejects unknown joints and non-finite values while clamping commands to actuator limits. Observations and actions provide `to_dict()` and `to_json()` methods for future transport layers. `config_validation/` also checks referenced body and actuator YAML semantics, including humanoid joint count, parent/child links, finite axes and ranges, initial-state joint names, control mode, and actuator limits.

## Rollout collection

The `training/` package keeps policies behind the same brain contract. A seeded baseline rollout can be collected and persisted as JSONL without exposing physics internals:

```bash
python examples/collect_rollouts.py
```

For custom experiments, implement `Policy.act(observation) -> Action`, pass it to `Trainer`, and provide a reward function. `config/training_config.yaml` records the intended episode, seed, and policy defaults. `evaluation/` runs seeded policy episodes and returns aggregate mean, standard deviation, minimum, and maximum reward metrics. The CLI can optionally persist a reproducibility-aware evaluation artifact for later reporting:

```bash
ai-sim evaluate --episodes 5 --max-steps 100 --seed 42 --json-out artifacts/evaluations/baseline.json
```

The example can be run with:

```bash
python examples/evaluate_policy.py
```

The multi-agent baseline can be run with:

```bash
python examples/multi_agent_example.py
```

`MultiAgentCoordinator` owns one isolated `SimulationEnvironment` per named agent and advances them in a deterministic synchronization order. Agents exchange only standardized observations and actions; they do not share physics internals.

## Recording and replay

`recording/` captures standardized observations and actions as a metadata-prefixed JSONL episode file. The same action sequence can be replayed through `ReplayBrain` without granting the replay layer access to physics internals:

```bash
python examples/record_replay.py
```

The recorder is intentionally separate from the simulator loop, so later storage formats or playback controls can be introduced without changing the brain or physics contracts. `datasets/` exports analysis-ready, versioned trajectory JSONL with metadata, observations, actions, rewards, termination flags, and per-step info:

```bash
python examples/export_dataset.py
```

`config/dataset_config.yaml` records the recommended dataset path and metadata fields.

## Robot adapter boundary

`robot/` defines a transport-neutral adapter contract and a safe `SimulatedRobotAdapter`. It exercises connect, observation, action, disconnect, and emergency-stop behavior against the simulator without opening live hardware connections. Run the example with:

```bash
python examples/robot_adapter_example.py
```

Live robot I/O is deliberately disabled by default and requires a future, explicitly implemented transport adapter. `config/robot_adapter_config.yaml` records this safety boundary.

## Vectorized environments

`vector_env/` provides a batched reset/step interface over independent simulator instances. The current implementation advances environments sequentially in a deterministic order, while keeping the API ready for a future parallel execution backend:

```bash
python examples/vector_env_example.py
```

Each environment has isolated state and receives one validated `Action` per batch position. `config/vector_env_config.yaml` contains the default batch size and seed.

## Sensor and actuator reliability

Proprioception and IMU sensors support optional seeded Gaussian noise and first-order low-pass filtering through `noise_std` and `filter_alpha` in YAML. Motor actuators support `response_time`, `max_velocity`, and torque limits; the simulator shapes validated commands before sending them across the physics boundary. The default configuration keeps noise disabled and uses a conservative response-time model.

## Checkpointing and resumable runs

`checkpointing/` stores versioned simulator state, actuator commands, metrics, run identifiers, and user metadata as JSON. Save and restore through the simulator API:

```python
sim.save_checkpoint("artifacts/checkpoint.json", run_id="experiment-01", metadata={"seed": 42})
sim.restore_checkpoint("artifacts/checkpoint.json")
```

The complete workflow is demonstrated by:

```bash
python examples/checkpoint_example.py
```

`config/checkpoint_config.yaml` records the recommended checkpoint path and cadence. Checkpoints are validated by version and joint set before restoration.

## Experiment orchestration

`experiments/` provides a configurable runner that creates repeatable seeded episodes, writes a machine-readable run manifest, and optionally saves checkpoints at an episode cadence. The runner API rejects non-integer or out-of-range episode, step, seed, and checkpoint values before creating simulator state. Each manifest is an `artifact_type: experiment_manifest` payload with `schema_version: 1` and records requested episode count, maximum steps, seed, configuration path, lifecycle status, and per-episode metrics so reruns can be checked for provenance:

```bash
python examples/run_experiment.py
```

The default settings live in `config/experiment_config.yaml`. Generated manifests and checkpoints are stored under `artifacts/`, which is intentionally excluded from version control. Multiple deterministic cases can be run in declaration order from a JSON file; each case may set `run_id`, `config`, `episodes`, `max_steps`, `seed`, `checkpoint_every`, and arbitrary parameter metadata:

```json
[
  {"run_id": "baseline-seed-1", "seed": 1, "episodes": 2, "max_steps": 100, "learning_rate": 0.001},
  {"run_id": "baseline-seed-2", "seed": 2, "episodes": 2, "max_steps": 100, "learning_rate": 0.001}
]
```

Sweep inputs are validated before execution: IDs and config paths must be non-empty strings, episode and step counts must be positive integers, seeds must be integers or `null`, and checkpoint cadence must be non-negative.

Run the sweep with:

```bash
ai-sim sweep --cases config/sweep_cases.json --sweep-id baseline-seeds --manifest-dir artifacts/runs --json-out artifacts/sweeps/baseline-seeds.json
ai-sim sweep --cases config/sweep_cases.json --sweep-id baseline-seeds --manifest-dir artifacts/runs --resume
```

Each generated manifest records `sweep_id`, declaration index, and non-runtime parameter values under `metadata.parameters`, so the report command can compare the resulting runs without changing the simulator or brain contracts. With `--json-out`, the sweep command also persists a compact `artifact_type: sweep` summary listing requested/completed/resumed cases and their manifest IDs; this summary is intentionally ignored as a run manifest when reports scan the directory. Re-running the same case file with `--resume` reuses matching completed manifests and rejects a changed configuration, episode count, or parameter set instead of silently mixing results.

## Benchmarking

`benchmarks/` provides a deterministic step-through benchmark that reports wall-clock duration, simulated duration, and real-time factor:

```bash
python examples/benchmark.py
```

Configure the run in `config/benchmark_config.yaml`. The benchmark is intentionally separate from training so performance measurements do not depend on policy behavior. For reproducible performance tracking, persist a benchmark artifact and include it in a report:

```bash
ai-sim benchmark --steps 1000 --seed 42 --json-out artifacts/benchmarks/baseline.json
ai-sim report --manifest-dir artifacts --json-out artifacts/report.json --markdown-out artifacts/report.md
```

Benchmark artifacts record the configuration path, seed, step count, simulated and wall-clock duration, real-time factor, and final simulator state. The benchmark API likewise rejects non-integer or non-positive step values and non-integer seeds before simulation starts.

## Evaluation

`evaluation/` runs seeded policy evaluation and produces aggregate episode metrics. The evaluator rejects non-integer or out-of-range episode, step, and seed inputs before touching simulator state. Persist the result with `ai-sim evaluate --json-out artifacts/evaluations/baseline.json`; the artifact is report-compatible and carries its configuration, seed, reward setting, and `schema_version: 1`.

## Runtime health diagnostics

`health.py` checks the canonical configuration and required/optional runtime dependencies. Persist a snapshot when collecting reproducibility evidence for a run:

```bash
ai-sim health --json-out artifacts/health/baseline.json
```

All persisted benchmark, evaluation, health, sweep, and experiment-manifest artifacts carry `schema_version: 1` alongside their `artifact_type`. The resulting `artifact_type: health` JSON can be included automatically by `ai-sim report --manifest-dir artifacts`, which reports the number of snapshots and healthy snapshots alongside experiment, evaluation, benchmark, and sweep summaries.
Sweep summary artifacts are shown with requested, completed, and resumed case counts. Report generation tolerates malformed or non-object JSON files, skips them, and records their paths under `artifact_errors` so one damaged artifact does not hide the remaining results. Use `ai-sim report --strict` in CI or release checks when any artifact parsing error or unsupported schema version should make the command exit non-zero; default mode skips those artifacts and records their paths under `artifact_errors`.

## Unified CLI

After installing the package, the `ai-sim` command provides one operator entry point for common workflows:

```bash
ai-sim validate config/simulator_config.yaml
ai-sim health --json-out artifacts/health/baseline.json
ai-sim benchmark --steps 1000 --seed 42 --json-out artifacts/benchmarks/baseline.json
ai-sim run --run-id baseline --episodes 3 --max-steps 100
ai-sim sweep --cases config/sweep_cases.json --sweep-id baseline-seeds --manifest-dir artifacts/runs --json-out artifacts/sweeps/baseline-seeds.json
ai-sim sweep --cases config/sweep_cases.json --sweep-id baseline-seeds --manifest-dir artifacts/runs --resume
ai-sim evaluate --episodes 5 --max-steps 100 --json-out artifacts/evaluations/baseline.json
ai-sim report --manifest-dir artifacts --json-out artifacts/report.json --markdown-out artifacts/report.md
```

Each command returns structured JSON suitable for shell automation and CI logs. `evaluate --json-out` additionally writes a machine-readable artifact containing the configuration path, seed, reward setting, episode metrics, and aggregate results. `report` aggregates experiment manifests and evaluation artifacts found under the selected directory into JSON and/or Markdown summaries. The equivalent source entry point is `python cli.py` when working directly from the repository. The wheel bundles the canonical YAML resources, so `ai-sim validate` and `ai-sim benchmark` also work outside the repository directory.

## Repository layout

`simulator/` orchestrates the loop and configuration. `brain/` contains the abstract brain contract and dummy implementation. `body/` loads links and joints from YAML. `physics/` defines the backend abstraction and portable MuJoCo/Bullet boundaries. `sensors/` and `actuators/` provide modular components. `environment/` contains the basic floor/world model. `rendering/` provides the headless renderer interface and Matplotlib implementation. `training/` provides policy, trainer, and rollout interfaces. `agents/` provides the multi-agent coordinator. `recording/` provides episode recording and replay primitives. `robot/` provides the safe adapter contract. `vector_env/` provides batched independent environments. `checkpointing/` provides versioned resumable state. `observability/` provides structured events and metrics. `experiments/` provides run orchestration and manifests. `evaluation/` provides seeded policy evaluation and aggregate metrics. `reports/` aggregates persisted experiment and evaluation artifacts. `benchmarks/` provides reproducible performance reports. `datasets/` provides versioned trajectory export. `config_validation/` provides canonical configuration diagnostics. `ai_body_simulator_resources/` bundles default YAML resources for installed distributions. `cli.py` exposes the unified operator CLI. `config/`, `examples/`, and `tests/` contain configuration, usage examples, and automated verification. Sensor transforms and actuator dynamics are covered by the reliability test suite.

## Design constraints

The AI brain must not access physics internals, body structure, sensor implementations, actuator mechanics, rendering, or training systems. All communication passes through the observation and action contracts. Body and physics parameters belong in YAML rather than hardcoded simulator logic.
