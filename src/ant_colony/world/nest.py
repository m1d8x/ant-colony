"""Expandable nest at the centre of the world."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class Nest:
    """A collection of nest tiles that can be expanded piecewise."""

    cx: int = 100
    cy: int = 100
    start_radius: int = 3

    _tiles: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self) -> None:
        if not self._tiles:
            r = self.start_radius
            self._tiles = {
                (x, y)
                for x in range(self.cx - r, self.cx + r + 1)
                for y in range(self.cy - r, self.cy + r + 1)
            }

    def contains(self, x: int, y: int) -> bool:
        return (x, y) in self._tiles

    def contains_float(self, x: float, y: float) -> bool:
        return (int(x), int(y)) in self._tiles

    def __contains__(self, pos: tuple[int, int]) -> bool:
        return pos in self._tiles

    def __len__(self) -> int:
        return len(self._tiles)

    @property
    def tiles(self) -> set[tuple[int, int]]:
        return self._tiles

    def __iter__(self) -> Iterator[tuple[int, int]]:
        return iter(self._tiles)

    def border_tiles(self) -> set[tuple[int, int]]:
        border: set[tuple[int, int]] = set()
        for x, y in self._tiles:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    nb = (x + dx, y + dy)
                    if nb not in self._tiles:
                        border.add(nb)
        return border

    def expansion_positions(self) -> list[tuple[int, int]]:
        border = self.border_tiles()
        return sorted(border, key=lambda p: abs(p[0] - self.cx) + abs(p[1] - self.cy))

    def add_tile(self, x: int, y: int) -> bool:
        if (x, y) in self._tiles:
            return False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if (x + dx, y + dy) in self._tiles:
                    self._tiles.add((x, y))
                    return True
        return False

    def add_tile_adjacent(self, x: int, y: int) -> bool:
        return self.add_tile(x, y)

    def as_mask(self, width: int, height: int) -> list[list[bool]]:
        mask = [[False] * height for _ in range(width)]
        for x, y in self._tiles:
            if 0 <= x < width and 0 <= y < height:
                mask[x][y] = True
        return mask

    def summary(self) -> dict:
        return {"cx": self.cx, "cy": self.cy, "tile_count": len(self._tiles)}

    def __repr__(self) -> str:
        return f"Nest(cx={self.cx}, cy={self.cy}, tiles={len(self._tiles)})"
