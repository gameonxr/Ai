from sensors import ObservationBuilder
import math

import numpy as np
import pytest

from body import BodyLoader
from sensors.depth import DepthSensor
from sensors.fusion import build_visual_fusion
from sensors.segmentation import SegmentationSensor
from sensors.perception import PerceptionSensor
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
    assert reading["frame_id"] == 0.0
    assert reading["camera"]["projection"] == "orthographic"


@pytest.mark.parametrize("config", [{"view_scale": 0.0}, {"view_scale": -1.0}, {"view_scale": True}, {"view_scale": "1.0"}])
def test_vision_sensor_rejects_invalid_view_scale(config):
    with pytest.raises(ValueError, match="vision view_scale"):
        VisionSensor("vision", config)


def test_vision_view_scale_changes_frame_coverage():
    body = BodyLoader.load("config/body_humanoid.yaml")
    state = {"joint_positions": {name: 0.0 for name in body.joints}}
    close = VisionSensor("vision", {"resolution": [64, 64], "view_scale": 0.5}, body=body).observe(state)
    wide = VisionSensor("vision", {"resolution": [64, 64], "view_scale": 2.0}, body=body).observe(state)
    assert close["camera"]["view_scale"] == 0.5
    assert wide["camera"]["view_scale"] == 2.0
    assert close["non_background_pixels"] > wide["non_background_pixels"]


def test_vision_sensor_without_body_reports_unavailable_frame():
    reading = VisionSensor("vision").observe({"joint_positions": {}})
    assert reading["available"] is False
    assert reading["non_background_pixels"] == 0


def test_depth_sensor_rejects_invalid_view_scale():
    with pytest.raises(ValueError, match="depth view_scale"):
        DepthSensor("depth", {"view_scale": 0.0})


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
    assert reading["frame_id"] == 0.0
    assert reading["camera"]["near"] == 0.1


@pytest.mark.parametrize("config", [{"near": 0.0}, {"far": 0.0}, {"near": 2.0, "far": 1.0}, {"near": "0.1"}])
def test_depth_sensor_rejects_invalid_range(config):
    with pytest.raises(ValueError, match="depth near and far"):
        DepthSensor("depth", config)


def test_segmentation_sensor_rejects_invalid_view_scale():
    with pytest.raises(ValueError, match="segmentation view_scale"):
        SegmentationSensor("segmentation", {"view_scale": 0.0})


def test_segmentation_sensor_returns_stable_body_labels():
    body = BodyLoader.load("config/body_humanoid.yaml")
    sensor = SegmentationSensor("segmentation", {"resolution": [32, 24]}, body=body)
    state = {"joint_positions": {name: 0.0 for name in body.joints}, "world": {"floor_enabled": True, "objects": [{"type": "floor", "size": (4.0, 6.0)}]}}
    reading = sensor.observe(state)
    assert reading["available"] is True
    assert reading["segmentation"].shape == (24, 32)
    assert reading["segmentation"].dtype.name == "int32"
    assert reading["label_map"]["background"] == 0
    assert reading["label_map"]["floor"] == 1
    assert reading["label_map"]["torso"] >= 2
    assert reading["valid_pixels"] > 0
    assert set(reading["segmentation"].flat) <= set(reading["label_map"].values())
    assert reading["frame_id"] == 0.0
    assert reading["camera"]["coordinate_frame"] == "body_debug"


def test_segmentation_sensor_without_body_reports_unavailable_map():
    reading = SegmentationSensor("segmentation").observe({"joint_positions": {}})
    assert reading["available"] is False
    assert reading["valid_pixels"] == 0
    assert reading["segmentation"].max() == 0


def test_perception_sensor_rejects_invalid_view_scale():
    with pytest.raises(ValueError, match="perception view_scale"):
        PerceptionSensor("perception", {"view_scale": 0.0})


def test_perception_sensor_returns_structured_body_summary():
    body = BodyLoader.load("config/body_humanoid.yaml")
    sensor = PerceptionSensor("perception", body=body)
    state = {"joint_positions": {name: 0.0 for name in body.joints}, "world": {"floor_enabled": True, "objects": [{"type": "floor", "size": (4.0, 6.0)}]}}
    reading = sensor.observe(state)
    assert reading["available"] is True
    assert reading["source"] == "headless_body_projection"
    assert reading["visible_links"] == list(body.links)
    assert reading["link_positions"]["torso"]["position"] == [0.0, 0.0, 1.0]
    assert reading["link_positions"]["torso"]["visible"] is True
    assert reading["body_bounds"]["min"][2] <= reading["body_bounds"]["max"][2]
    assert reading["world_objects"] == [{"type": "floor", "visible": True}]
    assert reading["frame_id"] == 0.0
    assert reading["camera"]["projection"] == "orthographic"


def test_visual_sensors_share_frame_id_for_sensor_fusion():
    body = BodyLoader.load("config/body_humanoid.yaml")
    state = {"time": 1.25, "joint_positions": {name: 0.0 for name in body.joints}, "world": {"floor_enabled": True, "objects": [{"type": "floor", "size": (4.0, 6.0)}]}}
    readings = [sensor.observe(state) for sensor in (VisionSensor("vision", body=body), DepthSensor("depth", body=body), SegmentationSensor("segmentation", body=body), PerceptionSensor("perception", body=body))]
    assert {reading["frame_id"] for reading in readings} == {1.25}


def test_perception_sensor_without_body_reports_unavailable_summary():
    reading = PerceptionSensor("perception").observe({"joint_positions": {}})
    assert reading["available"] is False
    assert reading["visible_links"] == []
    assert reading["body_bounds"] is None


def test_visual_fusion_builds_aligned_channel_summary():
    camera = {"projection": "orthographic", "coordinate_frame": "body_debug", "resolution": (32, 24)}
    readings = {
        "vision": {"frame_id": 1.0, "camera": camera, "available": True, "rgb": np.zeros((24, 32, 3), dtype=np.uint8), "non_background_pixels": 12},
        "depth": {"frame_id": 1.0, "camera": camera, "available": True, "depth": np.full((24, 32), 10.0, dtype=np.float32), "valid_pixels": 9},
    }
    summary = build_visual_fusion(readings)
    assert summary["aligned"] is True
    assert summary["frame_id"] == 1.0
    assert summary["available_channels"] == ["vision", "depth"]
    assert summary["channels"]["vision"]["valid_pixels"] == 12


def test_visual_fusion_reports_channel_schema_metadata():
    import numpy as np

    camera = {"projection": "orthographic", "coordinate_frame": "body_debug", "resolution": (4, 3)}
    summary = build_visual_fusion({"vision": {"frame_id": 1.0, "camera": camera, "available": True, "rgb": np.zeros((3, 4, 3), dtype=np.uint8), "non_background_pixels": 2}})
    assert summary["channels"]["vision"]["shape"] == [3, 4, 3]
    assert summary["channels"]["vision"]["dtype"] == "uint8"


@pytest.mark.parametrize("reading, message", [
    ({"frame_id": 1.0, "camera": {"projection": "orthographic", "coordinate_frame": "body_debug", "resolution": (4, 3)}, "available": True, "rgb": [[[]]]}, "RGB"),
    ({"frame_id": 1.0, "camera": {"projection": "orthographic", "coordinate_frame": "body_debug", "resolution": (4, 3)}, "available": True, "depth": __import__("numpy").zeros((3, 4), dtype="float64")}, "depth"),
])
def test_visual_fusion_rejects_invalid_channel_schema(reading, message):
    with pytest.raises(ValueError, match=message):
        build_visual_fusion({"vision" if message == "RGB" else "depth": reading})


def test_visual_fusion_rejects_mismatched_frames():
    camera = {"projection": "orthographic", "coordinate_frame": "body_debug", "resolution": (32, 24)}
    readings = {
        "vision": {"frame_id": 1.0, "camera": camera, "available": True, "rgb": np.zeros((24, 32, 3), dtype=np.uint8)},
        "depth": {"frame_id": 2.0, "camera": camera, "available": True, "depth": np.zeros((24, 32), dtype=np.float32)},
    }
    with pytest.raises(ValueError, match="frame_id values must match"):
        build_visual_fusion(readings)


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
    assert observation.proprioception is not None and observation.imu is not None and observation.vision is None and observation.segmentation is None and observation.perception is None
    assert "visual_fusion" not in observation.info
