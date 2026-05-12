"""FollowGradient — steer toward the strongest pheromone trail."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ant_colony.pysimengine import Behavior

if TYPE_CHECKING:
    from ant_colony.pysimengine import Agent, World


class FollowGradient(Behavior):
    """Read pheromone at three sensor points and steer up-gradient."""

    priority: int = 20

    def __init__(self, turn_rate: float = 0.30):
        self.turn_rate = turn_rate

    def update(self, agent: Agent, world: World) -> None:
        if not agent.alive:
            return

        is_forager = agent.role == "forager"
        read_fn = world.read_food_pheromone if is_forager else world.read_home_pheromone

        left_pt = agent.sensor_left()
        ctr_pt = agent.sensor_center()
        right_pt = agent.sensor_right()

        left_val = read_fn(left_pt[0], left_pt[1])
        ctr_val = read_fn(ctr_pt[0], ctr_pt[1])
        right_val = read_fn(right_pt[0], right_pt[1])

        if left_val > ctr_val and left_val > right_val:
            agent.direction -= self.turn_rate
        elif right_val > ctr_val and right_val > left_val:
            agent.direction += self.turn_rate
        else:
            agent.direction += random.uniform(-0.08, 0.08)

        agent.x += math.cos(agent.direction) * agent.speed
        agent.y += math.sin(agent.direction) * agent.speed
        agent.x = max(0.0, min(world.width - 1, agent.x))
        agent.y = max(0.0, min(world.height - 1, agent.y))
