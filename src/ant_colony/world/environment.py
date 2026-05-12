"""Environment — composite world."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .nest import Nest
from .obstacles import ObstacleGrid
from .food import FoodManager, FoodType
from .terrain import TerrainMap

_ATTR_PASSABLE = 1 << 0
_ATTR_NEST = 1 << 1
_ATTR_OBSTACLE = 1 << 2
_ATTR_FOOD = 1 << 3


@dataclass
class Environment:
    """Full simulation environment."""

    width: int = 200
    height: int = 200
    seed: int = 42
    nest_start_radius: int = 3
    rock_count: int = 25
    water_count: int = 4
    bush_count: int = 12
    mushroom_count: int = 8
    crystal_count: int = 4
    terrain_scale: float = 12.0
    terrain_octaves: int = 4

    nest: Nest = field(default_factory=lambda: Nest(100, 100, 3))
    obstacles: ObstacleGrid = field(default_factory=lambda: ObstacleGrid(200, 200))
    food: FoodManager = field(default_factory=FoodManager)
    terrain: TerrainMap = field(default_factory=lambda: TerrainMap(200, 200))

    _cell_attrs: list[list[int]] = field(default_factory=list)
    tick_number: int = 0

    def __post_init__(self) -> None:
        if not self._cell_attrs and not self.food.patches:
            self.generate()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "Environment":
        return cls(
            width=config.get("width", 200),
            height=config.get("height", 200),
            seed=config.get("seed", 42),
            nest_start_radius=config.get("nest_start_size", 3),
            rock_count=config.get("rock_count", config.get("obstacle_count", 25)),
            water_count=config.get("water_count", config.get("water_body_count", 4)),
            bush_count=config.get("bush_count", 12),
            mushroom_count=config.get("mushroom_count", 8),
            crystal_count=config.get("crystal_count", 4),
            terrain_scale=config.get("terrain_scale", 12.0),
            terrain_octaves=config.get("terrain_octaves", 4),
        )

    def generate(self) -> None:
        cx, cy = self.width // 2, self.height // 2
        nest_zone = (cx, cy, self.nest_start_radius)
        self.nest = Nest(cx=cx, cy=cy, start_radius=self.nest_start_radius)
        self.obstacles = ObstacleGrid.generate(
            self.width, self.height,
            rock_count=self.rock_count, water_count=self.water_count,
            seed=self.seed, nest_zone=nest_zone,
        )
        self.food = FoodManager.generate(
            self.width, self.height,
            bush_count=self.bush_count, mushroom_count=self.mushroom_count,
            crystal_count=self.crystal_count,
            seed=self.seed + 1, nest_zone=nest_zone,
        )
        self.terrain = TerrainMap(
            self.width, self.height,
            scale=self.terrain_scale, octaves=self.terrain_octaves,
            seed=self.seed + 2,
        )
        self._build_attr_grid()

    def _build_attr_grid(self) -> None:
        w, h = self.width, self.height
        attrs = [[_ATTR_PASSABLE] * h for _ in range(w)]
        for x in range(w):
            for y in range(h):
                if self.obstacles.is_blocked(float(x), float(y)):
                    attrs[x][y] = 0
                    attrs[x][y] |= _ATTR_OBSTACLE
        for x, y in self.nest:
            if 0 <= x < w and 0 <= y < h:
                attrs[x][y] |= _ATTR_NEST
                attrs[x][y] |= _ATTR_PASSABLE
        for p in self.food:
            if p.available and 0 <= p.x < w and 0 <= p.y < h:
                attrs[p.x][p.y] |= _ATTR_FOOD
        self._cell_attrs = attrs

    def _check(self, x: int, y: int, flag: int) -> bool:
        if 0 <= x < self.width and 0 <= y < self.height:
            return bool(self._cell_attrs[x][y] & flag)
        return False

    def is_passable(self, x: float, y: float) -> bool:
        return self._check(int(x), int(y), _ATTR_PASSABLE)

    def is_nest(self, x: float, y: float) -> bool:
        return self._check(int(x), int(y), _ATTR_NEST)

    def is_obstacle(self, x: float, y: float) -> bool:
        return self._check(int(x), int(y), _ATTR_OBSTACLE)

    def has_food(self, x: float, y: float) -> bool:
        return self.food.has_food(x, y)

    def movement_cost(self, x: float, y: float) -> float:
        if not self.is_passable(x, y):
            return float("inf")
        return 1.0

    def tick(self) -> None:
        self.tick_number += 1
        self.food.tick()
        self._refresh_food_flags()

    def _refresh_food_flags(self) -> None:
        w, h = self.width, self.height
        for x in range(w):
            for y in range(h):
                self._cell_attrs[x][y] &= ~_ATTR_FOOD
        for p in self.food:
            if p.available and 0 <= p.x < w and 0 <= p.y < h:
                self._cell_attrs[p.x][p.y] |= _ATTR_FOOD

    def expand_nest_toward(self, target_x: int, target_y: int) -> bool:
        border = self.nest.expansion_positions()
        if not border:
            return False
        best = min(border, key=lambda p: abs(p[0] - target_x) + abs(p[1] - target_y))
        if self.nest.add_tile(best[0], best[1]):
            x, y = best
            self._cell_attrs[x][y] |= _ATTR_NEST
            self._cell_attrs[x][y] |= _ATTR_PASSABLE
            self._cell_attrs[x][y] &= ~_ATTR_OBSTACLE
            return True
        return False

    def collect_food(self, ant_x: float, ant_y: float, amount: float) -> float:
        patch = self.food.patch_at(ant_x, ant_y, radius=2.0)
        if patch is None:
            return 0.0
        taken = patch.collect(amount)
        if patch.depleted and 0 <= patch.x < self.width and 0 <= patch.y < self.height:
            self._cell_attrs[patch.x][patch.y] &= ~_ATTR_FOOD
        return taken

    def render_text(self) -> str:
        w, h = self.width, self.height
        fc: dict[tuple[int, int], str] = {}
        for p in self.food:
            cm = {FoodType.BUSH: "B", FoodType.MUSHROOM: "M", FoodType.CRYSTAL: "C"}
            if p.available:
                fc[(p.x, p.y)] = cm.get(p.food_type, "F")
            elif p.depleted:
                fc[(p.x, p.y)] = cm.get(p.food_type, "f").lower()
        lines = []
        for y in range(min(h, 40)):
            row = ""
            for x in range(min(w, 80)):
                if (x, y) in self.nest:
                    row += "N"
                elif self.obstacles.is_blocked(float(x), float(y)):
                    row += "#"
                elif (x, y) in fc:
                    row += fc[(x, y)]
                else:
                    row += "·"
            lines.append(row)
        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "width": self.width, "height": self.height,
            "seed": self.seed, "tick": self.tick_number,
            "nest": self.nest.summary(),
            "obstacles": self.obstacles.summary(),
            "food": self.food.summary(),
            "terrain": self.terrain.summary(),
        }

    def __repr__(self) -> str:
        return (
            f"Environment({self.width}x{self.height}, "
            f"nest={len(self.nest)}t, "
            f"obstacles={self.obstacles.blocked_count()}c, "
            f"food={self.food.total_food():.0f}a)"
        )
