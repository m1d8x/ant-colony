"""SoldierAgent — PATROLLING -> COMBAT -> GUARDING state machine."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ant_colony.pysimengine import Agent

if TYPE_CHECKING:
    from ant_colony.pysimengine import World


PATROL_RADIUS = 30.0
COMBAT_THREAT_THRESHOLD = 0.3
GUARD_DURATION = 50
PATROL_TURN_INTERVAL = 40
DEPOSIT_DISTANCE = 5.0


class SoldierAgent(Agent):
    """Soldier ant — patrols, fights threats, guards the nest."""

    def __init__(self, x=0.0, y=0.0, direction=0.0, **kwargs):
        kwargs.setdefault("role", "soldier")
        super().__init__(x=x, y=y, direction=direction, **kwargs)
        self.state = "PATROLLING"
        self._guard_timer = 0
        self._patrol_turns = 0

    def update_state(self, world):
        if not self.alive:
            return
        {
            "PATROLLING": self._patrolling,
            "COMBAT": self._combat,
            "GUARDING": self._guarding,
        }.get(self.state, lambda w: None)(world)

    def _patrolling(self, world):
        if world.threat_level > COMBAT_THREAT_THRESHOLD:
            self.state = "COMBAT"
            return
        if math.hypot(self.x - world.nest_x, self.y - world.nest_y) > PATROL_RADIUS:
            dx = world.nest_x - self.x
            dy = world.nest_y - self.y
            target = math.atan2(dy, dx)
            diff = (target - self.direction + math.pi) % (2 * math.pi) - math.pi
            self.direction += math.copysign(min(abs(diff), 0.2), diff)
        self._patrol_turns += 1
        if self._patrol_turns >= PATROL_TURN_INTERVAL:
            self.direction += random.uniform(-0.5, 0.5)
            self._patrol_turns = 0

    def _combat(self, world):
        world.threat_level = max(0.0, world.threat_level - 0.02)
        if world.threat_level < COMBAT_THREAT_THRESHOLD * 0.5 and self._guard_timer > 5:
            self.state = "GUARDING"
            self._guard_timer = GUARD_DURATION
            return
        self._guard_timer += 1
        d_from_nest = math.hypot(self.x - world.nest_x, self.y - world.nest_y)
        if d_from_nest < PATROL_RADIUS * 0.5:
            dx = self.x - world.nest_x
            dy = self.y - world.nest_y
            if abs(dx) < 0.1 and abs(dy) < 0.1:
                dx, dy = math.cos(self.direction), math.sin(self.direction)
            target = math.atan2(dy, dx)
            diff = (target - self.direction + math.pi) % (2 * math.pi) - math.pi
            self.direction += math.copysign(min(abs(diff), 0.25), diff)
        else:
            self.direction += random.uniform(-0.1, 0.1)

    def _guarding(self, world):
        self._guard_timer -= 1
        if world.threat_level > COMBAT_THREAT_THRESHOLD:
            self.state = "COMBAT"
            self._guard_timer = 0
            return
        if self._guard_timer <= 0:
            self.state = "PATROLLING"
            self._patrol_turns = 0
            return
        if math.hypot(self.x - world.nest_x, self.y - world.nest_y) > DEPOSIT_DISTANCE * 2:
            dx = world.nest_x - self.x
            dy = world.nest_y - self.y
            target = math.atan2(dy, dx)
            diff = (target - self.direction + math.pi) % (2 * math.pi) - math.pi
            self.direction += math.copysign(min(abs(diff), 0.3), diff)
