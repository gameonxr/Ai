from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def _replace_atomically(content: str, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False) as temporary:
        temporary.write(content)
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, output)
    finally:
        temporary_path.unlink(missing_ok=True)


def write_text_atomic(content: str, path: str | Path) -> None:
    """Write text atomically so readers never observe a partial file."""
    _replace_atomically(content, path)


def write_json_atomic(payload: dict[str, Any], path: str | Path) -> None:
    """Write a JSON object atomically so readers never observe a partial file."""
    _replace_atomically(json.dumps(payload, indent=2, sort_keys=True) + "\n", path)
