from pathlib import Path

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


def test_validator_reports_invalid_yaml(tmp_path: Path):
    path = tmp_path / "invalid.yaml"
    path.write_text("simulator: [\n", encoding="utf-8")
    report = ConfigurationValidator().validate(path)
    assert not report.valid
    assert any("Invalid YAML" in error for error in report.errors)


def test_validator_reports_missing_reference(tmp_path: Path):
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
