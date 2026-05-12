"""ForagerAgent — SEARCHING -> FOUND_FOOD -> CARRYING -> RETURNING."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ant_colony.pysimengine import Agent

if TYPE_CHECKING:
    from ant_colony.pysimengine import World


COLLECT_DISTANCE = 3.0
DEPOSIT_DISTANCE = 5.0
FOOD_PICKUP_RATE = 2.0


class ForagerAgent(Agent):
    """Forager ant — searches for food and brings it back to the nest."""

    def __init__(self, x=0.0, y=0.0, direction=0.0, **kwargs):
        kwargs.setdefault("role", "forager")
        super().__init__(x=x, y=y, direction=direction, **kwargs)
        self.state = "SEARCHING"
        self._found_food_pos = None

    def update_state(self, world):
        if not self.alive:
            return
        {
            "SEARCHING": self._search,
            "FOUND_FOOD": self._found_food,
            "CARRYING": self._carrying,
            "RETURNING": self._returning,
        }.get(self.state, lambda w: None)(world)

    def _search(self, world):
        for fx, fy, amount in world.food_sources:
            if math.hypot(self.x - fx, self.y - fy) < COLLECT_DISTANCE and amount > 0:
                self._found_food_pos = (fx, fy)
                self.state = "FOUND_FOOD"
                return

    def _found_food(self, world):
        if self._found_food_pos is None:
            self.state = "SEARCHING"
            return
        fx, fy = self._found_food_pos
        for i, (sfx, sfy, amount) in enumerate(world.food_sources):
            if abs(sfx - fx) < 0.1 and abs(sfy - fy) < 0.1 and amount > 0:
                take = min(self.food_capacity - self.food_carrying, amount, FOOD_PICKUP_RATE)
                if take > 0:
                    self.food_carrying += take
                    world.food_sources[i] = (sfx, sfy, amount - take)
                self.state = "CARRYING"
                return
        self._found_food_pos = None
        self.state = "SEARCHING"

    def _carrying(self, world):
        if math.hypot(self.x - world.nest_x, self.y - world.nest_y) < DEPOSIT_DISTANCE:
            if self.food_carrying > 0:
                world.colony_food_store += self.food_carrying
                self.food_carrying = 0.0
            self.state = "RETURNING" if self._found_food_pos else "SEARCHING"
            return
        dx = world.nest_x - self.x
        dy = world.nest_y - self.y
        target = math.atan2(dy, dx)
        diff = (target - self.direction + math.pi) % (2 * math.pi) - math.pi
        self.direction += math.copysign(min(abs(diff), 0.15), diff)

    def _returning(self, world):
        if self._found_food_pos is None:
            self.state = "SEARCHING"
            return
        fx, fy = self._found_food_pos
        if math.hypot(self.x - fx, self.y - fy) < COLLECT_DISTANCE:
            self.state = "FOUND_FOOD"
            return
        dx = fx - self.x
        dy = fy - self.y
        target = math.atan2(dy, dx)
        diff = (target - self.direction + math.pi) % (2 * math.pi) - math.pi
        self.direction += math.copysign(min(abs(diff), 0.12), diff)
