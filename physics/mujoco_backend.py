from __future__ import annotations

from xml.sax.saxutils import escape

from .toy_backend import ToyPhysicsEngine


class MuJoCoBackend(ToyPhysicsEngine):
    """Headless MuJoCo backend boundary with a portable deterministic fallback.

    When native MuJoCo is available, ``load_body`` builds a minimal XML model from
    the configured links and joints. The public simulator contract remains unchanged.
    The analytical fallback is retained for platforms where native model creation is
    unavailable, keeping tests and examples portable.
    """

    engine_name = "mujoco"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        self.native = None
        self.model = None
        self.data = None
        try:
            import mujoco  # type: ignore
            self.native = mujoco
        except ImportError:
            self.native = None

    def load_body(self, body_definition) -> None:
        super().load_body(body_definition)
        if self.native is None:
            return
        try:
            self.model = self.native.MjModel.from_xml_string(self._build_xml(body_definition))
            self.data = self.native.MjData(self.model)
        except Exception:
            # A malformed optional native model must not break the stable Phase 1 API.
            self.model = None
            self.data = None

    def reset(self, seed=None) -> None:
        super().reset(seed)
        if self.model is not None and self.data is not None:
            self.native.mj_resetData(self.model, self.data)

    def step(self, dt=0.005) -> None:
        # The portable state remains the source of truth for deterministic API output.
        super().step(dt)
        if self.model is not None and self.data is not None:
            self.model.opt.timestep = float(dt)
            self.native.mj_step(self.model, self.data)

    def shutdown(self) -> None:
        self.data = None
        self.model = None
        super().shutdown()

    @staticmethod
    def _build_xml(body) -> str:
        geoms = []
        for name, link in body.links.items():
            props = link.properties
            shape = link.collision_shape
            if shape == "sphere":
                geom_type = f"sphere"; size = str(float(props.get("radius", 0.1)))
            elif shape == "cylinder":
                geom_type = "cylinder"; size = f"{float(props.get('radius', 0.05))} {float(props.get('length', 0.2)) / 2.0}"
            else:
                geom_type = "box"; dims = props.get("dimensions", [0.1, 0.1, 0.1]); size = " ".join(str(float(x) / 2.0) for x in dims)
            geoms.append(f'<body name="{escape(name)}"><geom type="{geom_type}" size="{size}" mass="{link.mass}"/></body>')
        return '<mujoco model="ai_body"><option gravity="0 0 -9.81"/><worldbody>' + "".join(geoms) + "</worldbody></mujoco>"
