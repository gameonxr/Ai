from actuators import MotorActuator
from sensors.transforms import GaussianNoise, LowPassFilter


def test_seeded_noise_is_reproducible():
    first = GaussianNoise(0.1, seed=5).apply([1.0, 2.0])
    second = GaussianNoise(0.1, seed=5).apply([1.0, 2.0])
    assert first == second


def test_low_pass_filter_smooths_values():
    filt = LowPassFilter(0.5)
    assert filt.apply([0.0]) == [0.0]
    assert filt.apply([1.0]) == [0.5]


def test_motor_response_respects_velocity_limit():
    motor = MotorActuator("neck", "neck", {"max_torque": 10.0, "max_velocity": 2.0, "response_time": 0.01})
    assert motor.shape_command(10.0, 0.1) == 0.2
    assert motor.shape_command(10.0, 0.1) == 0.4
