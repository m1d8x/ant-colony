"""BuilderAgent — IDLE -> GATHERING -> BUILDING state machine."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ant_colony.pysimengine import Agent

if TYPE_CHECKING:
    from ant_colony.pysimengine import World


GATHER_THRESHOLD = 80.0
GATHER_AMOUNT = 5.0
BUILD_DISTANCE = 6.0
BUILD_EFFICIENCY = 0.5
BUILD_RADIUS = 4.0
DEPOSIT_DISTANCE = 5.0


class BuilderAgent(Agent):
    """Builder ant — expands the nest using colony food supplies."""

    def __init__(self, x=0.0, y=0.0, direction=0.0, **kwargs):
        kwargs.setdefault("role", "builder")
        super().__init__(x=x, y=y, direction=direction, **kwargs)
        self.state = "IDLE"
        self._build_target = None

    def update_state(self, world):
        if not self.alive:
            return
        {
            "IDLE": self._idle,
            "GATHERING": self._gathering,
            "BUILDING": self._building,
        }.get(self.state, lambda w: None)(world)

    def _idle(self, world):
        if world.colony_food_store >= GATHER_THRESHOLD:
            self.state = "GATHERING"
            dx = world.nest_x - self.x
            dy = world.nest_y - self.y
            if abs(dx) > 1 or abs(dy) > 1:
                target = math.atan2(dy, dx)
                diff = (target - self.direction + math.pi) % (2 * math.pi) - math.pi
                self.direction += math.copysign(min(abs(diff), 0.1), diff)

    def _gathering(self, world):
        if math.hypot(self.x - world.nest_x, self.y - world.nest_y) < DEPOSIT_DISTANCE:
            take = min(GATHER_AMOUNT, world.colony_food_store)
            if take > 0:
                world.colony_food_store -= take
                self.food_carrying = take
            angle = random.uniform(0, 2 * math.pi)
            bx = max(0.0, min(world.width - 1, world.nest_x + math.cos(angle) * BUILD_DISTANCE))
            by = max(0.0, min(world.height - 1, world.nest_y + math.sin(angle) * BUILD_DISTANCE))
            self._build_target = (bx, by)
            self.state = "BUILDING"
            return
        dx = world.nest_x - self.x
        dy = world.nest_y - self.y
        target = math.atan2(dy, dx)
        diff = (target - self.direction + math.pi) % (2 * math.pi) - math.pi
        self.direction += math.copysign(min(abs(diff), 0.15), diff)

    def _building(self, world):
        if self._build_target is None or self.food_carrying <= 0:
            self.state = "IDLE"
            return
        bx, by = self._build_target
        if math.hypot(self.x - bx, self.y - by) < BUILD_RADIUS:
            deposit = self.food_carrying * BUILD_EFFICIENCY
            cx, cy = int(bx), int(by)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < world.width and 0 <= ny < world.height:
                        if world.obstacles[nx][ny] and random.random() < 0.5 * (deposit / 10.0):
                            world.obstacles[nx][ny] = False
            self.food_carrying = 0.0
            self.state = "IDLE"
            return
        dx = bx - self.x
        dy = by - self.y
        target = math.atan2(dy, dx)
        diff = (target - self.direction + math.pi) % (2 * math.pi) - math.pi
        self.direction += math.copysign(min(abs(diff), 0.12), diff)
