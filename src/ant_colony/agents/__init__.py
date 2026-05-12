"""Ant colony agent types — role-specific state machines.

Each agent type extends pysimengine.Agent with a state machine
(via ``update_state(world)``) and role-specific colour scheme.
"""

from .forager import ForagerAgent
from .builder import BuilderAgent
from .soldier import SoldierAgent
from .queen import QueenAgent

ANT_COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "forager": ((140, 90, 40), (200, 140, 60)),
    "builder": ((80, 55, 30), (120, 90, 50)),
    "soldier": ((140, 30, 30), (190, 60, 60)),
    "queen":   ((120, 70, 160), (160, 110, 200)),
}


def create_agent(
    role: str,
    x: float = 0.0,
    y: float = 0.0,
    direction: float = 0.0,
    **kwargs,
):
    """Factory: return the correct agent subclass for *role*."""
    cls_map = {
        "forager": ForagerAgent,
        "builder": BuilderAgent,
        "soldier": SoldierAgent,
        "queen": QueenAgent,
    }
    cls = cls_map.get(role)
    if cls is None:
        raise ValueError(f"Unknown agent role: {role!r}")
    return cls(x=x, y=y, direction=direction, role=role, **kwargs)


__all__ = [
    "ForagerAgent",
    "BuilderAgent",
    "SoldierAgent",
    "QueenAgent",
    "create_agent",
    "ANT_COLORS",
]
