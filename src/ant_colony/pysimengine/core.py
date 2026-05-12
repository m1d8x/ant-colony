"""Core simulation primitives: Agent, World, and the tick loop."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any


# ── Constants ────────────────────────────────────────────────────────────

DEFAULT_PHEROMONE_DECAY = 0.97       # multiplier per tick
PHEROMONE_DIFFUSION_RATE = 0.05      # fraction diffused to neighbours
MAX_PHEROMONE = 100.0


# ── Agent ────────────────────────────────────────────────────────────────

@dataclass
class Agent:
    """A single ant in the simulation."""

    x: float = 0.0
    y: float = 0.0
    direction: float = 0.0
    speed: float = 2.0
    role: str = "forager"
    food_carrying: float = 0.0
    food_capacity: float = 10.0
    alive: bool = True
    age: int = 0
    max_age: int = 5000
    color: tuple[int, int, int] = (200, 200, 200)   # body colour
    memory: dict[str, Any] = field(default_factory=dict)

    state: str = ""                     # FSM state (used by agent subclasses)
    sensor_distance: float = 8.0
    sensor_angle: float = math.radians(30)

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)

    def sensor_left(self) -> tuple[float, float]:
        angle = self.direction - self.sensor_angle
        return (self.x + math.cos(angle) * self.sensor_distance,
                self.y + math.sin(angle) * self.sensor_distance)

    def sensor_center(self) -> tuple[float, float]:
        return (self.x + math.cos(self.direction) * self.sensor_distance,
                self.y + math.sin(self.direction) * self.sensor_distance)

    def sensor_right(self) -> tuple[float, float]:
        angle = self.direction + self.sensor_angle
        return (self.x + math.cos(angle) * self.sensor_distance,
                self.y + math.sin(angle) * self.sensor_distance)

    def nearby_point(self, distance: float, angle_offset: float = 0.0) -> tuple[float, float]:
        angle = self.direction + angle_offset
        return (self.x + math.cos(angle) * distance,
                self.y + math.sin(angle) * distance)

    def update_state(self, world: World) -> None:
        """Run one tick of a role-specific state machine, if any.
        Default no-op — override in agent subclasses with FSM logic.
        """

# ── World ────────────────────────────────────────────────────────────────

@dataclass
class World:
    """2D grid containing pheromone maps, food sources, and obstacles."""

    width: int
    height: int

    food_pheromone: list[list[float]] = field(init=False)
    home_pheromone: list[list[float]] = field(init=False)
    obstacles: list[list[bool]] = field(init=False)

    food_sources: list[tuple[float, float, float]] = field(default_factory=list)

    nest_x: float = 0.0
    nest_y: float = 0.0

    colony_food_store: float = 100.0
    ants: list[Agent] = field(default_factory=list)
    tick_number: int = 0

    colony_manager: Any = None
    threat_level: float = 0.0

    def __post_init__(self):
        self.food_pheromone = [[0.0] * self.height for _ in range(self.width)]
        self.home_pheromone = [[0.0] * self.height for _ in range(self.width)]
        self.obstacles = [[False] * self.height for _ in range(self.width)]

    # ── pheromone helpers ────────────────────────────────────────────────

    def add_food_pheromone(self, x: int, y: int, amount: float) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.food_pheromone[x][y] = min(MAX_PHEROMONE, self.food_pheromone[x][y] + amount)

    def add_home_pheromone(self, x: int, y: int, amount: float) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            self.home_pheromone[x][y] = min(MAX_PHEROMONE, self.home_pheromone[x][y] + amount)

    def read_food_pheromone(self, x: float, y: float) -> float:
        px, py = int(x), int(y)
        if not (0 <= px < self.width - 1 and 0 <= py < self.height - 1):
            return 0.0
        fx, fy = x - px, y - py
        v00 = self.food_pheromone[px][py]
        v10 = self.food_pheromone[px + 1][py]
        v01 = self.food_pheromone[px][py + 1]
        v11 = self.food_pheromone[px + 1][py + 1]
        return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + \
               (v01 * (1 - fx) + v11 * fx) * fy

    def read_home_pheromone(self, x: float, y: float) -> float:
        px, py = int(x), int(y)
        if not (0 <= px < self.width - 1 and 0 <= py < self.height - 1):
            return 0.0
        fx, fy = x - px, y - py
        v00 = self.home_pheromone[px][py]
        v10 = self.home_pheromone[px + 1][py]
        v01 = self.home_pheromone[px][py + 1]
        v11 = self.home_pheromone[px + 1][py + 1]
        return (v00 * (1 - fx) + v10 * fx) * (1 - fy) + \
               (v01 * (1 - fx) + v11 * fx) * fy

    # ── obstacle helpers ─────────────────────────────────────────────────

    def is_blocked(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return True
        return self.obstacles[ix][iy]

    def _in_bounds(self, x: float, y: float) -> bool:
        return 0.0 <= x < self.width and 0.0 <= y < self.height

    # ── tick ─────────────────────────────────────────────────────────────

    def step_pheromones(self) -> None:
        decay = DEFAULT_PHEROMONE_DECAY
        diff = PHEROMONE_DIFFUSION_RATE

        for layer in (self.food_pheromone, self.home_pheromone):
            for x in range(self.width):
                for y in range(self.height):
                    layer[x][y] *= decay

            tmp = [row[:] for row in layer]
            for x in range(1, self.width - 1):
                for y in range(1, self.height - 1):
                    val = tmp[x][y]
                    if val < 0.01:
                        continue
                    spread = val * diff
                    layer[x - 1][y] += spread / 4
                    layer[x + 1][y] += spread / 4
                    layer[x][y - 1] += spread / 4
                    layer[x][y + 1] += spread / 4
                    layer[x][y] -= spread

    def step_ants(self) -> None:
        from ant_colony.behaviors.colony_manager import ColonyManager
        from ant_colony.behaviors.avoid_obstacles import AvoidObstacles
        from ant_colony.behaviors.deposit_trail import DepositTrail
        from ant_colony.behaviors.follow_gradient import FollowGradient
        from ant_colony.behaviors.wander import WanderWithPersistence

        if self.colony_manager is None:
            self.colony_manager = ColonyManager()
        self.colony_manager.update(None, self)

        follow = FollowGradient()
        deposit = DepositTrail()
        wander = WanderWithPersistence()
        avoid = AvoidObstacles()

        for ant in self.ants[:]:
            if not ant.alive:
                continue

            # Run role-specific state machine (forager / builder / soldier / queen)
            ant.update_state(self)

            # Queen is stationary — skip shared movement behaviors
            if ant.role == "queen":
                ant.age += 1
                if ant.age >= ant.max_age:
                    ant.alive = False
                continue

            # Clear movement flag before each tick
            ant.memory["_moved_this_tick"] = False
            avoid.update(ant, self)
            follow.update(ant, self)
            deposit.update(ant, self)
            wander.update(ant, self)
            ant.age += 1
            if ant.age >= ant.max_age:
                ant.alive = False

        self.ants = [a for a in self.ants if a.alive]

    def tick(self) -> None:
        self.tick_number += 1
        self.step_pheromones()
        self.step_ants()
