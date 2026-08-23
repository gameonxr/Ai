# Contributing

## Development setup

Use Python 3.10 or newer, create a virtual environment, and install the pinned project requirements:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Verification

Before opening a pull request, run the complete test suite, coverage report, examples, and whitespace check:

```bash
python -m pytest -q --cov=simulator --cov=brain --cov=core --cov=body --cov=physics --cov=sensors --cov=actuators --cov=environment --cov=rendering --cov=training --cov=agents --cov=recording --cov=robot --cov=vector_env --cov-report=term-missing
python examples/dummy_brain_example.py
python examples/render_simulation.py
python examples/collect_rollouts.py
python examples/multi_agent_example.py
python examples/record_replay.py
python examples/robot_adapter_example.py
python examples/vector_env_example.py
git diff --check
```

Keep the brain, physics, body, sensors, actuators, rendering, training, and adapter layers separated. Do not add live hardware I/O or external service credentials to tests. All new behavior should include a deterministic test and a short README or module docstring update.
