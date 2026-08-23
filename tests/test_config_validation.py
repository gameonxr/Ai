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
