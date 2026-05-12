"""QueenAgent — SPAWNING / IDLE. Stationary at nest, speed=0."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ant_colony.pysimengine import Agent

if TYPE_CHECKING:
    from ant_colony.pysimengine import World


SPAWN_COOLDOWN = 5
SPAWN_EFFECT_RADIUS = 15.0


class QueenAgent(Agent):
    """Queen ant — stationary spawner at the nest centre."""

    def __init__(self, x=0.0, y=0.0, direction=0.0, **kwargs):
        kwargs.setdefault("role", "queen")
        kwargs.setdefault("speed", 0.0)
        super().__init__(x=x, y=y, direction=direction, **kwargs)
        self.state = "SPAWNING"
        self._spawn_timer = 0

    def update_state(self, world):
        if not self.alive:
            return
        self.x = world.nest_x
        self.y = world.nest_y
        if self.state == "SPAWNING":
            self._spawning(world)
        elif self.state == "IDLE":
            self._idle(world)

    def _spawning(self, world):
        self._spawn_timer += 1
        if self._spawn_timer >= SPAWN_COOLDOWN:
            self.state = "IDLE"
            self._spawn_timer = 0
        pulse = abs(math.sin(self._spawn_timer / SPAWN_COOLDOWN * math.pi))
        self.memory["spawn_pulse"] = SPAWN_EFFECT_RADIUS * pulse

    def _idle(self, world):
        self.memory["spawn_pulse"] = 0.0
        if self.memory.get("_queen_spawn_signal", False):
            self.state = "SPAWNING"
            self._spawn_timer = 0
            self.memory["_queen_spawn_signal"] = False
