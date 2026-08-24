from .matplotlib_3d_renderer import Matplotlib3DRenderer
from .matplotlib_renderer import MatplotlibRenderer
from .renderer import Renderer


def create_renderer(name: str = "matplotlib", config: dict | None = None) -> Renderer:
    normalized_name = name.lower()
    if normalized_name in {"matplotlib", "headless", "2d"}:
        return MatplotlibRenderer(config)
    if normalized_name in {"3d", "matplotlib3d", "headless3d"}:
        return Matplotlib3DRenderer(config)
    raise ValueError(f"Unsupported renderer: {name}")


__all__ = ["Renderer", "MatplotlibRenderer", "Matplotlib3DRenderer", "create_renderer"]
