"""Behavior pattern — single-responsibility behaviours that mutate agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Agent, World


class Behavior:
    """A single behaviour that reads sensor data and acts on the agent."""

    priority: int = 0

    def update(self, agent: Agent, world: World) -> None:
        raise NotImplementedError


class CompositeBehavior(Behavior):
    """Runs children in priority order."""

    def __init__(self, children: list[Behavior] | None = None):
        self.children = sorted(children or [], key=lambda b: -b.priority)

    def add(self, behavior: Behavior) -> None:
        self.children.append(behavior)
        self.children.sort(key=lambda b: -b.priority)

    def update(self, agent: Agent, world: World) -> None:
        for child in self.children:
            child.update(agent, world)
