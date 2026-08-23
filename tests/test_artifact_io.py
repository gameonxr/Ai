import json
from pathlib import Path

from artifact_io import write_json_atomic


def test_write_json_atomic_creates_parent_and_valid_json(tmp_path: Path):
    output = tmp_path / "nested" / "artifact.json"
    write_json_atomic({"artifact_type": "test", "value": 7}, output)
    assert json.loads(output.read_text(encoding="utf-8")) == {"artifact_type": "test", "value": 7}
    assert list(output.parent.glob(f".{output.name}.*.tmp")) == []


def test_write_json_atomic_replaces_existing_file(tmp_path: Path):
    output = tmp_path / "artifact.json"
    write_json_atomic({"value": 1}, output)
    write_json_atomic({"value": 2}, output)
    assert json.loads(output.read_text(encoding="utf-8"))["value"] == 2
