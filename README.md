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

The brain receives only `core.Observation` and returns only `core.Action`. `ActionValidator` rejects unknown joints and non-finite values while clamping commands to actuator limits. Observations and actions provide `to_dict()` and `to_json()` methods for future transport layers.

## Rollout collection

The `training/` package keeps policies behind the same brain contract. A seeded baseline rollout can be collected and persisted as JSONL without exposing physics internals:

```bash
python examples/collect_rollouts.py
```

For custom experiments, implement `Policy.act(observation) -> Action`, pass it to `Trainer`, and provide a reward function. `config/training_config.yaml` records the intended episode, seed, and policy defaults. The multi-agent baseline can be run with:

```bash
python examples/multi_agent_example.py
```

`MultiAgentCoordinator` owns one isolated `SimulationEnvironment` per named agent and advances them in a deterministic synchronization order. Agents exchange only standardized observations and actions; they do not share physics internals.

## Recording and replay

`recording/` captures standardized observations and actions as a metadata-prefixed JSONL episode file. The same action sequence can be replayed through `ReplayBrain` without granting the replay layer access to physics internals:

```bash
python examples/record_replay.py
```

The recorder is intentionally separate from the simulator loop, so later storage formats or playback controls can be introduced without changing the brain or physics contracts.

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

## Repository layout

`simulator/` orchestrates the loop and configuration. `brain/` contains the abstract brain contract and dummy implementation. `body/` loads links and joints from YAML. `physics/` defines the backend abstraction and portable MuJoCo/Bullet boundaries. `sensors/` and `actuators/` provide modular components. `environment/` contains the basic floor/world model. `rendering/` provides the headless renderer interface and Matplotlib implementation. `training/` provides policy, trainer, and rollout interfaces. `agents/` provides the multi-agent coordinator. `recording/` provides episode recording and replay primitives. `robot/` provides the safe adapter contract. `vector_env/` provides batched independent environments. `checkpointing/` provides versioned resumable state. `observability/` provides structured events and metrics. `config/`, `examples/`, and `tests/` contain configuration, usage examples, and automated verification. Sensor transforms and actuator dynamics are covered by the reliability test suite.

## Design constraints

The AI brain must not access physics internals, body structure, sensor implementations, actuator mechanics, rendering, or training systems. All communication passes through the observation and action contracts. Body and physics parameters belong in YAML rather than hardcoded simulator logic.
