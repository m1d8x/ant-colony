"""
PHGrid — numpy-backed multi-layer pheromone grid with diffusion, evaporation,
obstacle masking, and gradient sensing for ant steering.

Each layer represents a pheromone type (food, home, build, danger).
"""

from __future__ import annotations

import enum
import math

import numpy as np


class PheromoneType(enum.IntEnum):
    """Pheromone channel identifiers.  Also carry a display colour."""

    AT_HOME = 0
    TO_FOOD = 1
    TO_BUILD = 2
    DANGER = 3

    @property
    def color(self) -> tuple[int, int, int]:
        return _PHEROMONE_COLORS[self]


_PHEROMONE_COLORS: dict[PheromoneType, tuple[int, int, int]] = {
    PheromoneType.AT_HOME: (0, 0, 255),
    PheromoneType.TO_FOOD: (0, 255, 0),
    PheromoneType.TO_BUILD: (255, 165, 0),
    PheromoneType.DANGER: (255, 0, 0),
}


class PHGrid:
    """Multi-layer pheromone grid backed by a numpy float32 array.

    Grid dimensions are in *cell* coordinates (not world pixels).
    ``cell_size`` controls how many world-space units map to one cell.
    """

    def __init__(
        self,
        width: int,
        height: int,
        n_types: int = 4,
        cell_size: int = 4,
    ):
        self.width = width
        self.height = height
        self.n_types = n_types
        self.cell_size = cell_size
        # shape: (n_types, height, width) — row-major access per type
        self._grid = np.zeros((n_types, height, width), dtype=np.float32)
        # obstacle mask: True = blocked for diffusion (height, width)
        self._obstacle_mask = np.zeros((height, width), dtype=bool)

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def grid(self) -> np.ndarray:
        """Return the raw numpy array (n_types, height, width)."""
        return self._grid

    @property
    def shape(self) -> tuple[int, int]:
        """Return (width, height) — the logical grid dimensions."""
        return (self.width, self.height)

    @property
    def obstacle_mask(self) -> np.ndarray:
        """Boolean mask (height, width); True = blocked for diffusion."""
        return self._obstacle_mask

    # ── Coordinate helpers ────────────────────────────────────────────────

    def _world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """Map world-space coordinates to cell (column, row)."""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def _in_bounds(self, cx: int, cy: int) -> bool:
        return 0 <= cx < self.width and 0 <= cy < self.height

    # ── Deposit / Read ────────────────────────────────────────────────────

    def deposit(self, world_pos: tuple[float, float], ptype: PheromoneType, amount: float):
        """Deposit pheromone at a world-space position (maps to one cell)."""
        cx, cy = self._world_to_cell(world_pos[0], world_pos[1])
        self.deposit_cell(cx, cy, ptype, amount)

    def deposit_cell(self, cx: int, cy: int, ptype: PheromoneType, amount: float):
        """Deposit pheromone at a cell coordinate."""
        if not self._in_bounds(cx, cy):
            return
        self._grid[ptype, cy, cx] += amount

    def get(self, world_pos: tuple[float, float], ptype: PheromoneType) -> float:
        """Read pheromone value at a world-space position."""
        cx, cy = self._world_to_cell(world_pos[0], world_pos[1])
        return self.get_cell(cx, cy, ptype)

    def get_cell(self, cx: int, cy: int, ptype: PheromoneType) -> float:
        """Read pheromone value at a cell coordinate."""
        if not self._in_bounds(cx, cy):
            return 0.0
        return float(self._grid[ptype, cy, cx])

    def set_cell(self, cx: int, cy: int, ptype: PheromoneType, amount: float):
        """Set pheromone value at a cell coordinate (overwrite)."""
        if not self._in_bounds(cx, cy):
            return
        self._grid[ptype, cy, cx] = amount

    # ── Evaporation ───────────────────────────────────────────────────────

    def evaporate(self, rate: float):
        """Multiply all cells by (1 - rate).  No negatives."""
        factor = max(0.0, 1.0 - rate)
        self._grid *= factor

    # ── Diffusion (4-neighbour, obstacle-aware, mass-conserving) ──────────

    def diffuse(self):
        """One step of 4-neighbour isotropic diffusion.

        Mass is conserved.  Cells blocked by obstacle_mask don't receive
        mass; their share is redistributed to the passable neighbours of
        the source cell (bounce-back).
        """
        # Work on a copy so we read pre-diffusion values throughout
        src = self._grid.copy()  # (n_types, height, width)
        diffused = src.copy()

        # 4-neighbour offsets
        neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for t in range(self.n_types):
            layer = src[t]  # (height, width)
            out = diffused[t]
            for cy in range(self.height):
                for cx in range(self.width):
                    val = layer[cy, cx]
                    if val <= 0:
                        continue

                    # Find which neighbours are passable
                    passable = []
                    for dx, dy in neighbours:
                        nx, ny = cx + dx, cy + dy
                        if self._in_bounds(nx, ny) and not self._obstacle_mask[ny, nx]:
                            passable.append((nx, ny))

                    if not passable:
                        continue  # all neighbours blocked — mass stays

                    # Equal share to each passable neighbour
                    share = val * 0.1  # diffusion fraction
                    per_nb = share / len(passable)

                    out[cy, cx] -= share
                    for nx, ny in passable:
                        out[ny, nx] += per_nb

        self._grid[:] = diffused

    # ── Obstacle mask ─────────────────────────────────────────────────────

    def set_obstacle_mask(self, pixel_mask: np.ndarray):
        """Down-sample a pixel-space boolean mask to cell granularity.

        ``pixel_mask`` has shape (pixel_height, pixel_width).  A cell is
        marked blocked if *any* pixel within its cell_size² region is blocked.
        """
        h, w = pixel_mask.shape
        cell = self.cell_size
        rows = min(self.height, (h + cell - 1) // cell)
        cols = min(self.width, (w + cell - 1) // cell)
        for cy in range(rows):
            for cx in range(cols):
                y0 = cy * cell
                x0 = cx * cell
                y1 = min(y0 + cell, h)
                x1 = min(x0 + cell, w)
                self._obstacle_mask[cy, cx] = bool(
                    pixel_mask[y0:y1, x0:x1].any()
                )

    # ── Sampling ahead for gradient sensing ───────────────────────────────

    def sample_ahead(
        self,
        pos: tuple[float, float],
        heading: float,
        vision_radius: float,
        ptype: PheromoneType,
    ) -> tuple[float, float, float]:
        """Sample pheromone at left, center, right sensor points.

        Returns (left_value, center_value, right_value).
        Sensors are at ±35° from heading.
        """
        sensor_angle = 0.35  # ~20 degrees
        cx = pos[0] + math.cos(heading) * vision_radius
        cy = pos[1] + math.sin(heading) * vision_radius
        center = self.get((cx, cy), ptype)

        lx = pos[0] + math.cos(heading - sensor_angle) * vision_radius
        ly = pos[1] + math.sin(heading - sensor_angle) * vision_radius
        left = self.get((lx, ly), ptype)

        rx = pos[0] + math.cos(heading + sensor_angle) * vision_radius
        ry = pos[1] + math.sin(heading + sensor_angle) * vision_radius
        right = self.get((rx, ry), ptype)

        return (left, center, right)

    def steer_toward_gradient(
        self,
        pos: tuple[float, float],
        heading: float,
        vision_radius: float,
        ptype: PheromoneType,
        turn_rate: float,
    ) -> float:
        """Return angular delta to steer toward strongest pheromone gradient.

        Positive delta = turn right, negative = turn left, zero = straight.
        """
        left, center, right = self.sample_ahead(pos, heading, vision_radius, ptype)
        if left > right and left > center:
            return -turn_rate  # turn left
        elif right > left and right > center:
            return turn_rate  # turn right
        return 0.0  # go straight

    # ── Utility ───────────────────────────────────────────────────────────

    def total(self, ptype: PheromoneType | None = None) -> float:
        """Return total mass across all cells, optionally for one type."""
        if ptype is not None:
            return float(self._grid[ptype].sum())
        return float(self._grid.sum())

    def clear(self):
        """Zero out all pheromone values."""
        self._grid.fill(0.0)

    def resize(self, new_width: int, new_height: int):
        """Resize the grid, preserving data in overlapping region."""
        new_grid = np.zeros((self.n_types, new_height, new_width), dtype=np.float32)
        new_mask = np.zeros((new_height, new_width), dtype=bool)
        w = min(self.width, new_width)
        h = min(self.height, new_height)
        new_grid[:, :h, :w] = self._grid[:, :h, :w]
        new_mask[:h, :w] = self._obstacle_mask[:h, :w]
        self._grid = new_grid
        self._obstacle_mask = new_mask
        self.width = new_width
        self.height = new_height
