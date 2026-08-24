from pathlib import Path

from body import BodyLoader


import pytest
import yaml


def test_joint_rejects_malformed_configuration():
    from body.joint import Joint

    cases = (
        ([], "Joint config must be a mapping"),
        ({"parent": "torso", "child": "head", "axis": ["0", 0, 1]}, "Joint axis must be a finite 3-vector"),
        ({"parent": "torso", "child": "head", "range": [1, 1]}, "Joint range must be a finite 2-vector"),
        ({"parent": "torso", "child": "head", "max_torque": 0}, "Joint max_torque must be positive"),
        ({"parent": "torso", "child": "head", "damping": -0.1}, "Joint damping must be non-negative"),
    )
    for config, message in cases:
        with pytest.raises(ValueError, match=message):
            Joint.from_config("neck", config)  # type: ignore[arg-type]


def test_link_rejects_malformed_configuration():
    from body.link import Link

    cases = (
        ([], "Link config must be a mapping"),
        ({"mass": "1.0"}, "Link mass must be a finite positive number"),
        ({"inertia": [1.0, 0.0, 1.0]}, "Link inertia must be a finite positive 3-vector"),
        ({"collision_shape": ""}, "Link collision_shape must be a non-empty string"),
    )
    for config, message in cases:
        with pytest.raises(ValueError, match=message):
            Link.from_config("torso", config)  # type: ignore[arg-type]


def test_body_has_expected_dof():
    body = BodyLoader.load("config/body_humanoid.yaml")
    assert body.dof >= 12
    assert len(body.links) >= 10
    assert body.joints["neck"].range_radians[1] > 0


def test_body_loader_reports_missing_and_invalid_files(tmp_path):
    with pytest.raises(ValueError, match="Unable to read body configuration"):
        BodyLoader.load(tmp_path / "missing.yaml")

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text("body: [broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid body YAML in .+invalid.yaml"):
        BodyLoader.load(invalid)

    non_mapping = tmp_path / "list.yaml"
    non_mapping.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        BodyLoader.load(non_mapping)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        ("body", "mass", "1.0", "Body mass must be a finite number"),
        ("body", "mass", 0, "Body mass must be positive"),
        ("body", "default_damping", -0.1, "Body default_damping must be non-negative"),
        ("initial_state", "joint_positions", {"neck": "0.1"}, "Body initial joint position neck must be a finite number"),
    ],
)
def test_body_loader_rejects_coerced_metadata(tmp_path, section, field, value, message):
    payload = yaml.safe_load(Path("config/body_humanoid.yaml").read_text(encoding="utf-8"))
    payload[section][field] = value
    path = tmp_path / "invalid-metadata.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        BodyLoader.load(path)


def test_body_loader_rejects_malformed_sections(tmp_path):
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("body: {}\nlinks: []\njoints: {}\ninitial_state: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="section links must be a mapping"):
        BodyLoader.load(malformed)
