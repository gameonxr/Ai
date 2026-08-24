from sensors import ObservationBuilder
import math

import pytest

from sensors.imu import IMUSensor
from sensors.sensor_registry import build_sensors
from sensors.transforms import GaussianNoise, LowPassFilter
from sensors.vision import VisionSensor


def test_vision_sensor_validates_camera_configuration():
    for resolution in ([64], [64, 0], [64, True], "64x64"):
        with pytest.raises(ValueError, match="vision resolution"):
            VisionSensor("vision", {"resolution": resolution})  # type: ignore[arg-type]
    for fov in (0.0, 181.0, math.nan, True, "90"):
        with pytest.raises(ValueError, match="vision fov"):
            VisionSensor("vision", {"fov": fov})  # type: ignore[arg-type]
    sensor = VisionSensor("vision", {"resolution": [80, 60], "fov": 120.0})
    assert sensor.observe({})["resolution"] == (80, 60)
    assert sensor.observe({})["fov"] == 120.0


def test_sensor_base_rejects_malformed_constructor_inputs():
    for name, config, message in (("", {}, "sensor name must be a non-empty string"), ("imu", [], "sensor config must be an object"), ("imu", {"enabled": 1}, "sensor enabled must be a boolean")):
        with pytest.raises(ValueError, match=message):
            IMUSensor(name, config)  # type: ignore[arg-type]


def test_sensor_registry_rejects_malformed_configuration():
    with pytest.raises(ValueError, match="sensor configuration must be an object"):
        build_sensors([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="sensor settings for imu must be an object"):
        build_sensors({"imu": []})  # type: ignore[arg-type]


def test_sensor_transforms_reject_invalid_parameters():
    for value in (True, -0.1, math.nan, "0.1"):
        with pytest.raises(ValueError, match="standard_deviation"):
            GaussianNoise(value)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed must be an integer or null"):
        GaussianNoise(seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed must be an integer or null"):
        GaussianNoise().reset(seed=True)  # type: ignore[arg-type]
    for value in (True, 0.0, 1.1, math.nan, "1.0"):
        with pytest.raises(ValueError, match="alpha"):
            LowPassFilter(value)  # type: ignore[arg-type]


def test_observation_builder_rejects_invalid_inputs():
    builder = ObservationBuilder(build_sensors({"proprioception": {"enabled": True}}))
    for state, timestamp, message in (([], 0.0, "physics_state must be an object"), ({}, math.nan, "timestamp must be a finite number"), ({}, True, "timestamp must be a finite number")):
        with pytest.raises(ValueError, match=message):
            builder.build(state, timestamp)  # type: ignore[arg-type]


def test_only_enabled_sensors_are_exposed():
    sensors = build_sensors({"proprioception": {"enabled": True}, "vision": {"enabled": False}, "imu": {"enabled": True}})
    observation = ObservationBuilder(sensors).build({"joint_positions": {}, "joint_velocities": {}, "joint_accelerations": {}, "body_position": [0,0,1], "body_velocity": [0,0,0], "body_rotation": [0,0,0,1], "body_angular_velocity": [0,0,0], "gravity": [0,0,-9.81]}, 0.0)
    assert observation.proprioception is not None and observation.imu is not None and observation.vision is None
