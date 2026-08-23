from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Renderer(ABC):
    """Stable renderer contract; rendering stays outside physics and brain APIs."""

    @abstractmethod
    def render(self, body, state: dict[str, Any], output_path: str | Path | None = None) -> Any:
        """Render the current body state and optionally save an artifact."""
        raise NotImplementedError

    def close(self) -> None:
        return None
