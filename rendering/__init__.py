from .matplotlib_renderer import MatplotlibRenderer
from .renderer import Renderer


def create_renderer(name: str = "matplotlib", config: dict | None = None) -> Renderer:
    if name.lower() in {"matplotlib", "headless", "2d"}:
        return MatplotlibRenderer(config)
    raise ValueError(f"Unsupported renderer: {name}")


__all__ = ["Renderer", "MatplotlibRenderer", "create_renderer"]
