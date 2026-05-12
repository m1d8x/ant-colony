"""DepositTrail — leave pheromone at the agent's current position."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ant_colony.pysimengine import Behavior

if TYPE_CHECKING:
    from ant_colony.pysimengine import Agent, World


class DepositTrail(Behavior):
    """Drop pheromone proportional to food carried and ant type."""

    priority: int = 40

    def __init__(self, base_deposit: float = 1.0, food_multiplier: float = 2.0):
        self.base_deposit = base_deposit
        self.food_multiplier = food_multiplier

    def update(self, agent: Agent, world: World) -> None:
        if not agent.alive:
            return

        gx, gy = int(agent.x), int(agent.y)
        if not (0 <= gx < world.width and 0 <= gy < world.height):
            return

        if agent.role == "forager":
            strength = self.base_deposit + agent.food_carrying * self.food_multiplier
            world.add_food_pheromone(gx, gy, strength)
        else:
            world.add_home_pheromone(gx, gy, self.base_deposit)
