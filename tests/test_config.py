from pathlib import Path

import pytest

from simulator import ConfigLoader


def test_config_loader_resolves_references():
    config = ConfigLoader("config/simulator_config.yaml").load()
    assert config["body"]["loaded"]["body"]["type"] == "humanoid"
    assert config["physics"]["loaded"]["physics"]["gravity"][2] == -9.81


def test_config_loader_reports_unreadable_root_path(tmp_path: Path):
    unreadable = tmp_path / "root-directory"
    unreadable.mkdir()
    with pytest.raises(FileNotFoundError, match="Unable to read configuration"):
        ConfigLoader(unreadable).load()


def test_config_loader_rejects_malformed_root_yaml(tmp_path: Path):
    path = tmp_path / "invalid.yaml"
    path.write_text("simulator: [\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML configuration"):
        ConfigLoader(path).load()


def test_config_loader_rejects_non_mapping_root(tmp_path: Path):
    path = tmp_path / "list.yaml"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Configuration root must be a mapping"):
        ConfigLoader(path).load()


def test_config_loader_rejects_non_mapping_section(tmp_path: Path):
    path = tmp_path / "section.yaml"
    path.write_text("simulator: []\nphysics: {}\nbody: {}\nsensors: {}\nactuators: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Configuration sections must be mappings: simulator"):
        ConfigLoader(path).load()


def test_config_loader_rejects_non_numeric_timestep(tmp_path: Path):
    path = tmp_path / "timestep.yaml"
    path.write_text("simulator: {timestep: fast}\nphysics: {}\nbody: {}\nsensors: {}\nactuators: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="simulator timestep values must be numeric"):
        ConfigLoader(path).load()


def test_config_loader_merge_rejects_non_mapping_inputs():
    with pytest.raises(ValueError, match="Configuration merge base must be a mapping"):
        ConfigLoader.merge([], {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Configuration merge override must be a mapping"):
        ConfigLoader.merge({}, [])  # type: ignore[arg-type]


def test_config_loader_reports_unreadable_reference_path(tmp_path: Path):
    reference = tmp_path / "reference-directory"
    reference.mkdir()
    path = tmp_path / "config.yaml"
    path.write_text("simulator: {}\nphysics: {config_path: reference-directory}\nbody: {}\nsensors: {}\nactuators: {}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Unable to read referenced configuration"):
        ConfigLoader(path).load()


def test_config_loader_rejects_malformed_reference_yaml(tmp_path: Path):
    (tmp_path / "physics.yaml").write_text("physics: [\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text("simulator: {}\nphysics: {config_path: physics.yaml}\nbody: {}\nsensors: {}\nactuators: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid YAML referenced configuration"):
        ConfigLoader(path).load()


def test_config_loader_rejects_non_mapping_reference(tmp_path: Path):
    (tmp_path / "physics.yaml").write_text("[]\n", encoding="utf-8")
    path = tmp_path / "config.yaml"
    path.write_text("simulator: {}\nphysics: {config_path: physics.yaml}\nbody: {}\nsensors: {}\nactuators: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Referenced configuration must be a mapping"):
        ConfigLoader(path).load()
