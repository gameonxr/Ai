import pytest

from benchmarks import run_benchmark


def test_benchmark_reports_expected_duration():
    result = run_benchmark("config/simulator_config.yaml", steps=10, seed=3)
    assert result.steps == 10
    assert result.simulation_seconds == pytest.approx(0.05)
    assert result.wall_seconds >= 0.0
    assert result.final_state["step_count"] == 10


def test_benchmark_validates_step_count():
    with pytest.raises(ValueError):
        run_benchmark(steps=0)
