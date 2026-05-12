"""Food patches — three types with different values, depletion, and respawn."""

from __future__ import annotations

import enum
import math
import random
from dataclasses import dataclass, field
from typing import Iterator


class FoodType(enum.Enum):
    BUSH = 30
    MUSHROOM = 50
    CRYSTAL = 100

    @property
    def base_value(self) -> float:
        return float(self.value)

    @property
    def label(self) -> str:
        return self.name.lower()


DEFAULT_RESPAWN_TICKS = {
    FoodType.BUSH: 150,
    FoodType.MUSHROOM: 300,
    FoodType.CRYSTAL: 600,
}


@dataclass
class FoodPatch:
    """A single food source at a grid position."""

    x: int = 0
    y: int = 0
    food_type: FoodType = FoodType.BUSH
    max_amount: float = 30.0
    current_amount: float = 30.0
    respawn_ticks: int = 150
    depletion_timer: int = 0

    def __post_init__(self) -> None:
        if self.max_amount == 30.0 and self.food_type != FoodType.BUSH:
            self.max_amount = float(self.food_type.value)
        if self.current_amount == 30.0:
            self.current_amount = self.max_amount
        if self.respawn_ticks == 150 and self.food_type != FoodType.BUSH:
            self.respawn_ticks = DEFAULT_RESPAWN_TICKS[self.food_type]

    @property
    def depleted(self) -> bool:
        return self.current_amount <= 0.0

    @property
    def available(self) -> bool:
        return self.current_amount > 0.0

    def collect(self, amount: float) -> float:
        if self.current_amount <= 0:
            return 0.0
        taken = min(amount, self.current_amount)
        self.current_amount -= taken
        if self.current_amount <= 0.0:
            self.current_amount = 0.0
            self.depletion_timer = self.respawn_ticks
        return taken

    def tick(self) -> None:
        if self.current_amount <= 0 and self.depletion_timer > 0:
            self.depletion_timer -= 1
            if self.depletion_timer <= 0:
                self.current_amount = self.max_amount
                self.depletion_timer = 0

    def __repr__(self) -> str:
        status = f"{self.current_amount:.0f}/{self.max_amount:.0f}"
        if self.depleted:
            status = f"depleted ({self.depletion_timer}t)"
        return f"FoodPatch({self.x},{self.y} {self.food_type.label} {status})"


@dataclass
class FoodManager:
    """Manages all food patches."""

    patches: list[FoodPatch] = field(default_factory=list)

    @classmethod
    def generate(
        cls,
        width: int = 200,
        height: int = 200,
        bush_count: int = 12,
        mushroom_count: int = 8,
        crystal_count: int = 4,
        seed: int | None = None,
        nest_zone: tuple[int, int, int] | None = None,
    ) -> "FoodManager":
        rng = random.Random(seed)
        patches: list[FoodPatch] = []

        def _in_nest_zone(x: int, y: int) -> bool:
            if nest_zone is None:
                return False
            cx, cy, r = nest_zone
            return abs(x - cx) <= r + 3 and abs(y - cy) <= r + 3

        configs = [
            (FoodType.BUSH, bush_count, 30, DEFAULT_RESPAWN_TICKS[FoodType.BUSH]),
            (FoodType.MUSHROOM, mushroom_count, 50, DEFAULT_RESPAWN_TICKS[FoodType.MUSHROOM]),
            (FoodType.CRYSTAL, crystal_count, 100, DEFAULT_RESPAWN_TICKS[FoodType.CRYSTAL]),
        ]

        for food_type, count, value, respawn in configs:
            placed = 0
            attempts = 0
            while placed < count and attempts < count * 50:
                attempts += 1
                x = rng.randint(2, width - 3)
                y = rng.randint(2, height - 3)
                if _in_nest_zone(x, y):
                    continue
                overlap = False
                for p in patches:
                    if abs(p.x - x) + abs(p.y - y) < 3:
                        overlap = True
                        break
                if overlap:
                    continue
                patches.append(FoodPatch(
                    x=x, y=y, food_type=food_type,
                    max_amount=float(value), current_amount=float(value),
                    respawn_ticks=respawn,
                ))
                placed += 1

        return cls(patches=patches)

    def patch_at(self, x: float, y: float, radius: float = 2.0) -> FoodPatch | None:
        best: FoodPatch | None = None
        best_dist = radius
        ix, iy = int(x), int(y)
        for p in self.patches:
            if not p.available:
                continue
            d = math.hypot(p.x - ix, p.y - iy)
            if d < best_dist:
                best_dist = d
                best = p
        return best

    def all_available(self) -> list[FoodPatch]:
        return [p for p in self.patches if p.available]

    def all_depleted(self) -> list[FoodPatch]:
        return [p for p in self.patches if p.depleted]

    def total_food(self) -> float:
        return sum(p.current_amount for p in self.patches if p.available)

    def has_food(self, x: float, y: float) -> bool:
        return self.patch_at(x, y, radius=1.5) is not None

    def __len__(self) -> int:
        return len(self.patches)

    def __iter__(self) -> Iterator[FoodPatch]:
        return iter(self.patches)

    def tick(self) -> None:
        for p in self.patches:
            p.tick()

    def summary(self) -> dict:
        return {
            "total_patches": len(self.patches),
            "available": sum(1 for p in self.patches if p.available),
            "depleted": sum(1 for p in self.patches if p.depleted),
            "total_food": self.total_food(),
            "by_type": {
                ft.label: {
                    "count": sum(1 for p in self.patches if p.food_type == ft),
                    "available": sum(1 for p in self.patches if p.food_type == ft and p.available),
                }
                for ft in FoodType
            },
        }

    def __repr__(self) -> str:
        return f"FoodManager({len(self.patches)} patches, {self.total_food():.0f} food)"
