from pathlib import Path

import pytest
import yaml

from config_validation import ConfigurationValidator


def test_canonical_simulator_config_is_valid():
    report = ConfigurationValidator().validate(Path("config/simulator_config.yaml"))
    assert report.valid, report.errors


def test_validator_reports_missing_section(tmp_path: Path):
    path = tmp_path / "missing.yaml"
    path.write_text("simulator: {timestep: 0.01}\n", encoding="utf-8")
    report = ConfigurationValidator().validate(path)
    assert not report.valid
    assert any("Missing required section" in error for error in report.errors)


def test_validator_reports_unreadable_root_path(tmp_path: Path):
    unreadable = tmp_path / "root-directory"
    unreadable.mkdir()
    report = ConfigurationValidator().validate(unreadable)
    assert not report.valid
    assert any("Unable to read configuration" in error for error in report.errors)


def test_validator_reports_non_mapping_root(tmp_path: Path):
    path = tmp_path / "list.yaml"
    path.write_text("[]\n", encoding="utf-8")
    report = ConfigurationValidator().validate(path)
    assert not report.valid
    assert report.errors == ["Root configuration must be a mapping"]


def test_validator_reports_invalid_yaml(tmp_path: Path):
    path = tmp_path / "invalid.yaml"
    path.write_text("simulator: [\n", encoding="utf-8")
    report = ConfigurationValidator().validate(path)
    assert not report.valid
    assert any("Invalid YAML" in error for error in report.errors)


def test_validator_reports_unreadable_reference(tmp_path: Path):
    reference = tmp_path / "reference-directory"
    reference.mkdir()
    path = tmp_path / "config.yaml"
    path.write_text("simulator: {}\nphysics: {config_path: reference-directory}\nbody: {}\nsensors: {}\nactuators: {}\n", encoding="utf-8")
    report = ConfigurationValidator().validate(path)
    assert not report.valid
    assert any("Unable to read physics configuration" in error for error in report.errors)


def test_validator_reports_non_mapping_reference(tmp_path: Path):
    (tmp_path / "physics.yaml").write_text("[]\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text("simulator: {}\nphysics: {config_path: physics.yaml}\nbody: {}\nsensors: {}\nactuators: {}\n", encoding="utf-8")
    report = ConfigurationValidator().validate(path)
    assert not report.valid
    assert any("must contain a mapping" in error for error in report.errors)


def test_validator_reports_malformed_reference_yaml(tmp_path: Path):
    path = tmp_path / "missing-reference.yaml"
    path.write_text("simulator: {}\nphysics: {config_path: missing.yaml}\nbody: {}\nsensors: {}\nactuators: {}\n", encoding="utf-8")
    report = ConfigurationValidator().validate(path)
    assert not report.valid
    assert any("does not exist" in error for error in report.errors)


def test_validator_reports_body_and_actuator_semantic_errors(tmp_path: Path):
    (tmp_path / "body.yaml").write_text(
        "links: {torso: {}}\njoints:\n  broken: {parent: torso, child: missing, axis: [0, 0], range: [1, 1], max_torque: 0}\ninitial_state: {joint_positions: {unknown: nan}}\n",
        encoding="utf-8",
    )
    (tmp_path / "actuator.yaml").write_text(
        "actuators: {control_mode: velocity, defaults: {max_torque: 0, max_velocity: -1, max_force: 1, damping: -0.1, response_time: nan}}\n",
        encoding="utf-8",
    )
    config = tmp_path / "semantic-invalid.yaml"
    config.write_text(
        "simulator: {timestep: 0.01}\nphysics: {}\nbody: {config_path: body.yaml}\nsensors: {}\nactuators: {config_path: actuator.yaml}\n",
        encoding="utf-8",
    )
    report = ConfigurationValidator().validate(config)
    assert not report.valid
    assert any("at least 12 joints" in error for error in report.errors)
    assert any("unknown link" in error for error in report.errors)
    assert any("axis must contain three" in error for error in report.errors)
    assert any("control_mode must be torque or position" in error for error in report.errors)
    assert any("max_torque must be positive" in error for error in report.errors)
    assert any("response_time" in error for error in report.errors)


@pytest.mark.parametrize(
    ("environment", "message"),
    [
        ([], "environment must be a mapping when provided"),
        ({"floor_enabled": "false"}, "environment.floor_enabled must be a boolean"),
        ({"floor_friction": True}, "environment.floor_friction must be a finite non-negative number"),
        ({"floor_size": [10]}, "environment.floor_size must be a finite positive 2-vector"),
    ],
)
def test_validator_reports_invalid_environment_values(tmp_path: Path, environment, message):
    config = tmp_path / "environment-invalid.yaml"
    config.write_text(
        "simulator: {}\nphysics: {}\nbody: {}\nsensors: {}\nactuators: {}\n"
        + yaml.safe_dump({"environment": environment}),
        encoding="utf-8",
    )
    report = ConfigurationValidator().validate(config)
    assert not report.valid
    assert message in report.errors


def test_validator_reports_unknown_initial_joint(tmp_path: Path):
    body = tmp_path / "body.yaml"
    body.write_text(
        "links: {torso: {}}\njoints: {neck: {parent: torso, child: torso, axis: [0, 0, 1], range: [-1, 1], max_torque: 1}}\ninitial_state: {joint_positions: {missing: 0}}\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text(
        "simulator: {}\nphysics: {}\nbody: {config_path: body.yaml}\nsensors: {}\nactuators: {}\n",
        encoding="utf-8",
    )
    report = ConfigurationValidator().validate(config)
    assert any("references unknown joint" in error for error in report.errors)
