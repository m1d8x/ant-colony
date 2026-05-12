"""AvoidObstacles — sense walls and obstacles, steer away."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ant_colony.pysimengine import Behavior

if TYPE_CHECKING:
    from ant_colony.pysimengine import Agent, World


class AvoidObstacles(Behavior):
    """Cast rays ahead and steer away from blocked cells."""

    priority: int = 100

    def __init__(self, look_ahead: float = 6.0, scan_arc: int = 5):
        self.look_ahead = look_ahead
        self.scan_arc = scan_arc

    def update(self, agent: Agent, world: World) -> None:
        if not agent.alive:
            return

        half = self.scan_arc // 2
        blocked_dirs: list[float] = []
        clear_dirs: list[float] = []

        for i in range(self.scan_arc):
            offset = math.radians(30) * (i - half) / max(half, 1)
            angle = agent.direction + offset
            px = agent.x + math.cos(angle) * self.look_ahead
            py = agent.y + math.sin(angle) * self.look_ahead

            if not world._in_bounds(px, py) or world.is_blocked(px, py):
                blocked_dirs.append(offset)
            else:
                clear_dirs.append(offset)

        if blocked_dirs and len(blocked_dirs) > len(clear_dirs):
            # More blocked than clear — hard left/right
            agent.direction += random.choice([-1, 1]) * math.radians(45)
        elif blocked_dirs:
            # Some blocked — steer toward nearest clear
            best_offset = min(clear_dirs, key=lambda o: abs(o))
            agent.direction += best_offset * 0.5

        new_x = agent.x + math.cos(agent.direction) * agent.speed
        new_y = agent.y + math.sin(agent.direction) * agent.speed

        if not world._in_bounds(new_x, new_y) or world.is_blocked(new_x, new_y):
            # Slide along obstacle
            if not world.is_blocked(new_x, agent.y) and world._in_bounds(new_x, agent.y):
                agent.x = new_x
            elif not world.is_blocked(agent.x, new_y) and world._in_bounds(agent.x, new_y):
                agent.y = new_y
            else:
                agent.direction += random.uniform(-1.0, 1.0)
            agent.memory["_moved_this_tick"] = True
            return

        agent.x = new_x
        agent.y = new_y
        agent.memory["_moved_this_tick"] = True
