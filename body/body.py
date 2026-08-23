from dataclasses import dataclass, field
from .joint import Joint
from .link import Link


@dataclass
class Body:
    body_type: str
    mass: float
    height: float
    default_damping: float
    links: dict[str, Link] = field(default_factory=dict)
    joints: dict[str, Joint] = field(default_factory=dict)
    initial_joint_positions: dict[str, float] = field(default_factory=dict)

    @property
    def dof(self) -> int:
        return len(self.joints)

    def reset(self) -> None:
        return None

    def definition(self) -> dict:
        return {"type": self.body_type, "mass": self.mass, "height": self.height, "links": self.links, "joints": self.joints}
