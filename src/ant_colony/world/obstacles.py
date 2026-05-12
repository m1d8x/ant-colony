"""Procedural obstacle generation — blob-shaped rocks and water bodies."""

from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class ObstacleGrid:
    """2-D bool grid marking impassable cells."""

    width: int = 200
    height: int = 200
    obstacles: list[list[bool]] = field(default_factory=list)
    _blob_ids: list[list[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.obstacles:
            w, h = self.width, self.height
            self.obstacles = [[False] * h for _ in range(w)]
            self._blob_ids = [[0] * h for _ in range(w)]

    def is_blocked(self, x: float, y: float) -> bool:
        ix, iy = int(x), int(y)
        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return True
        return self.obstacles[ix][iy]

    def blocks_pheromone(self, x: int, y: int) -> bool:
        if not (0 <= x < self.width and 0 <= y < self.height):
            return True
        return self.obstacles[x][y]

    @classmethod
    def generate(
        cls,
        width: int = 200,
        height: int = 200,
        rock_count: int = 25,
        water_count: int = 4,
        seed: int | None = None,
        nest_zone: tuple[int, int, int] | None = None,
    ) -> "ObstacleGrid":
        rng = random.Random(seed)
        grid = [[False] * height for _ in range(width)]
        blob_ids = [[0] * height for _ in range(width)]
        next_blob = 1

        def _blocked(x: int, y: int) -> bool:
            return not (0 <= x < width and 0 <= y < height) or grid[x][y]

        def _in_nest_zone(x: int, y: int) -> bool:
            if nest_zone is None:
                return False
            cx, cy, r = nest_zone
            return abs(x - cx) <= r + 2 and abs(y - cy) <= r + 2

        def _seed_blob(
            bid: int, max_cells: int, growth_prob: float, smooth_factor: float,
        ) -> None:
            cells = [(x, y) for x in range(width) for y in range(height)
                     if blob_ids[x][y] == bid]
            rng.shuffle(cells)
            attempts = 0
            while len(cells) < max_cells and attempts < max_cells * 4:
                attempts += 1
                bx, by = rng.choice(cells)
                dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
                dx, dy = rng.choice(dirs)
                nx, ny = bx + dx, by + dy
                if _blocked(nx, ny) or _in_nest_zone(nx, ny):
                    continue
                blob_nbs = sum(
                    1 for ddx, ddy in dirs
                    if (0 <= nx + ddx < width and 0 <= ny + ddy < height
                        and blob_ids[nx + ddx][ny + ddy] == bid)
                )
                prob = growth_prob
                if blob_nbs >= 2:
                    prob *= smooth_factor
                elif blob_nbs == 0:
                    prob *= 0.3
                if rng.random() < prob:
                    grid[nx][ny] = True
                    blob_ids[nx][ny] = bid
                    cells.append((nx, ny))

        # Rocks
        for i in range(rock_count):
            bid = next_blob
            next_blob += 1
            for _ in range(20):
                sx = rng.randint(1, width - 2)
                sy = rng.randint(1, height - 2)
                if not _blocked(sx, sy) and not _in_nest_zone(sx, sy):
                    grid[sx][sy] = True
                    blob_ids[sx][sy] = bid
                    break
            else:
                continue
            _seed_blob(bid, max_cells=rng.randint(6, 18),
                       growth_prob=0.25, smooth_factor=1.0)

        # Water
        for i in range(water_count):
            bid = next_blob
            next_blob += 1
            for _ in range(30):
                sx = rng.randint(3, width - 4)
                sy = rng.randint(3, height - 4)
                if not _blocked(sx, sy) and not _in_nest_zone(sx, sy):
                    grid[sx][sy] = True
                    blob_ids[sx][sy] = bid
                    for _ in range(rng.randint(2, 5)):
                        dx, dy = rng.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                        cx, cy = sx + dx, sy + dy
                        if (0 <= cx < width and 0 <= cy < height
                                and not _blocked(cx, cy)
                                and not _in_nest_zone(cx, cy)):
                            grid[cx][cy] = True
                            blob_ids[cx][cy] = bid
                    break
            else:
                continue
            _seed_blob(bid, max_cells=rng.randint(60, 200),
                       growth_prob=0.35, smooth_factor=1.8)

        return cls(width=width, height=height,
                   obstacles=grid, _blob_ids=blob_ids)

    def diffusion_mask(self) -> list[list[bool]]:
        return [[not self.obstacles[x][y] for y in range(self.height)]
                for x in range(self.width)]

    def blocked_count(self) -> int:
        return sum(row.count(True) for row in self.obstacles)

    def summary(self) -> dict:
        return {"width": self.width, "height": self.height,
                "blocked_cells": self.blocked_count()}

    def __repr__(self) -> str:
        return f"ObstacleGrid({self.width}x{self.height}, blocked={self.blocked_count()})"
