"""WanderWithPersistence — biased random walk with directional memory."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ant_colony.pysimengine import Behavior

if TYPE_CHECKING:
    from ant_colony.pysimengine import Agent, World


class WanderWithPersistence(Behavior):
    """Biased random walk — keeps heading most of the time, occasionally turns."""

    priority: int = 5

    def __init__(self, persistence: float = 0.80, max_turn: float = 0.35):
        self.persistence = persistence
        self.max_turn = max_turn

    def update(self, agent: Agent, world: World) -> None:
        if not agent.alive:
            return

        if random.random() > self.persistence:
            agent.direction += random.uniform(-self.max_turn, self.max_turn)

        if not agent.memory.get("_moved_this_tick", False):
            agent.x += math.cos(agent.direction) * agent.speed
            agent.y += math.sin(agent.direction) * agent.speed
            agent.x = max(0.0, min(world.width - 1, agent.x))
            agent.y = max(0.0, min(world.height - 1, agent.y))
