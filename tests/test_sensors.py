from sensors import ObservationBuilder
import math

import pytest

from body import BodyLoader
from sensors.depth import DepthSensor
from sensors.imu import IMUSensor
from sensors.proprioception import ProprioceptionSensor
from sensors.sensor_registry import build_sensors
from sensors.transforms import GaussianNoise, LowPassFilter
from sensors.vision import VisionSensor


def test_touch_sensor_validates_contact_input():
    from sensors.touch import TouchSensor

    sensor = TouchSensor("touch")
    with pytest.raises(ValueError, match="physics_state must be an object"):
        sensor.observe([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="contacts must be a list"):
        sensor.observe({"contacts": {}})
    assert sensor.observe({}) == {"contacts": []}


@pytest.mark.parametrize("config", [{"noise_std": "0.1"}, {"noise_std": True}, {"filter_alpha": "1.0"}])
def test_imu_rejects_implicitly_coerced_transform_parameters(config):
    with pytest.raises(ValueError, match="Invalid IMU sensor configuration"):
        IMUSensor("imu", config)


@pytest.mark.parametrize("config", [{"noise_std": "0.1"}, {"noise_std": True}, {"filter_alpha": "1.0"}])
def test_proprioception_rejects_implicitly_coerced_transform_parameters(config):
    with pytest.raises(ValueError, match="Invalid proprioception sensor configuration"):
        ProprioceptionSensor("proprioception", config)


def test_sensor_specific_transform_errors_are_contextual():
    with pytest.raises(ValueError, match="Invalid IMU sensor configuration"):
        IMUSensor("imu", {"noise_std": "not-a-number"})
    with pytest.raises(ValueError, match="Invalid proprioception sensor configuration"):
        ProprioceptionSensor("proprioception", {"filter_alpha": 2.0})


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


def test_vision_sensor_rasterizes_configured_body_projection():
    body = BodyLoader.load("config/body_humanoid.yaml")
    sensor = VisionSensor("vision", {"resolution": [32, 24]}, body=body)
    state = {"joint_positions": {name: 0.0 for name in body.joints}, "world": {"floor_enabled": True, "objects": [{"type": "floor", "size": (4.0, 6.0)}]}}
    reading = sensor.observe(state)
    assert reading["available"] is True
    assert reading["rgb"].shape == (24, 32, 3)
    assert reading["rgb"].dtype.name == "uint8"
    assert reading["non_background_pixels"] > 0
    assert reading["source"] == "headless_body_projection"


def test_vision_sensor_without_body_reports_unavailable_frame():
    reading = VisionSensor("vision").observe({"joint_positions": {}})
    assert reading["available"] is False
    assert reading["non_background_pixels"] == 0


def test_depth_sensor_rasterizes_body_depth_map():
    body = BodyLoader.load("config/body_humanoid.yaml")
    sensor = DepthSensor("depth", {"resolution": [32, 24], "near": 0.1, "far": 10.0}, body=body)
    state = {"joint_positions": {name: 0.0 for name in body.joints}, "world": {"floor_enabled": True, "objects": [{"type": "floor", "size": (4.0, 6.0)}]}}
    reading = sensor.observe(state)
    assert reading["available"] is True
    assert reading["depth"].shape == (24, 32)
    assert reading["depth"].dtype.name == "float32"
    assert reading["valid_pixels"] > 0
    assert reading["depth"].max() == 10.0
    assert reading["depth"].min() < 10.0


@pytest.mark.parametrize("config", [{"near": 0.0}, {"far": 0.0}, {"near": 2.0, "far": 1.0}, {"near": "0.1"}])
def test_depth_sensor_rejects_invalid_range(config):
    with pytest.raises(ValueError, match="depth near and far"):
        DepthSensor("depth", config)


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
