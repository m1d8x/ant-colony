"""Pheromone heatmap overlay — additive blend showing colony scent trails.

Pheromones are stored in ``World.food_pheromone[x][y]`` and
``World.home_pheromone[x][y]`` — 2D lists of floats (0..MAX_PHEROMONE).

The overlay is rebuilt every N frames for performance.
"""
from __future__ import annotations

import pygame

from ant_colony.renderers.utils import make_surf, PX_PER_CELL

PHERO_COLORS: dict[str, tuple[int, int, int]] = {
    "food": (30, 100, 255),     # blue — trail to food
    "home": (80, 220, 80),      # green — trail home
}

UPDATE_INTERVAL = 10  # frames between full rebuilds


class PheromoneLayer:
    """Renders a pheromone heatmap overlay with additive blending."""

    def __init__(self, world_width: int, world_height: int) -> None:
        self.world_w = world_width
        self.world_h = world_height
        px_w = world_width * PX_PER_CELL
        px_h = world_height * PX_PER_CELL
        self._surface = make_surf(px_w, px_h, alpha=True)
        self._frame_counter = 0
        self._dirty = True
        self._max_alpha = 140

    def update(self, food_grid: list[list[float]],
               home_grid: list[list[float]],
               cam_px: float, cam_py: float) -> None:
        """Rebuild the overlay every N frames — only visible cells."""
        self._frame_counter += 1
        if self._frame_counter < UPDATE_INTERVAL and not self._dirty:
            return
        self._frame_counter = 0
        self._dirty = False

        surf = self._surface
        surf.fill((0, 0, 0, 0))

        grids = [("food", food_grid, PHERO_COLORS["food"]),
                 ("home", home_grid, PHERO_COLORS["home"])]
        max_val = 100.0

        start_wx = max(0, int(cam_px / PX_PER_CELL))
        start_wy = max(0, int(cam_py / PX_PER_CELL))
        end_wx = min(self.world_w, int((cam_px + 1280) / PX_PER_CELL) + 1)
        end_wy = min(self.world_h, int((cam_py + 800) / PX_PER_CELL) + 1)

        for wy in range(start_wy, end_wy):
            for wx in range(start_wx, end_wx):
                r, g, b, total = 0, 0, 0, 0.0
                for _, grid, color in grids:
                    val = 0.0
                    try:
                        val = grid[wx][wy]
                    except (IndexError, TypeError):
                        pass
                    if val <= 0.0:
                        continue
                    total += val
                    cr, cg, cb = color
                    r += cr * val
                    g += cg * val
                    b += cb * val
                if total <= 0.0:
                    continue
                r = int(r / total)
                g = int(g / total)
                b = int(b / total)
                alpha = min(int(self._max_alpha * (total / max_val)),
                            self._max_alpha)
                px = wx * PX_PER_CELL
                py = wy * PX_PER_CELL
                pygame.draw.rect(surf, (r, g, b, alpha),
                                 (px, py, PX_PER_CELL, PX_PER_CELL))

    @property
    def surface(self) -> pygame.Surface:
        return self._surface

    def mark_dirty(self) -> None:
        self._dirty = True
