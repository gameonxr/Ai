from body import BodyLoader


import pytest


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


def test_body_loader_rejects_malformed_sections(tmp_path):
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text("body: {}\nlinks: []\njoints: {}\ninitial_state: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="section links must be a mapping"):
        BodyLoader.load(malformed)
