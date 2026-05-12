"""Utility functions for the renderer — surface helpers, colour math, caching."""
from __future__ import annotations

import math
import random
from functools import lru_cache

import pygame


# World-to-pixel scale: 1 world cell = PX_PER_CELL pixels.
PX_PER_CELL = 10


def make_surf(width: int, height: int, alpha: bool = True) -> pygame.Surface:
    """Create a surface with optional per-pixel alpha."""
    return pygame.Surface((width, height), pygame.SRCALPHA if alpha else 0)


def lerp_color(a: tuple[int, int, int], b: tuple[int, int, int],
               t: float) -> tuple[int, int, int]:
    """Linearly interpolate between two RGB colours."""
    return (
        int(a[0] + (b[0] - a[0]) * t),
        int(a[1] + (b[1] - a[1]) * t),
        int(a[2] + (b[2] - a[2]) * t),
    )


def color_with_alpha(color: tuple[int, int, int],
                     alpha: int) -> tuple[int, int, int, int]:
    """Add alpha channel to an RGB colour."""
    return (color[0], color[1], color[2], alpha)


def round_rect(surface: pygame.Surface, rect: pygame.Rect,
               color: tuple[int, ...], radius: int = 6,
               border: int = 0,
               border_color: tuple[int, ...] | None = None) -> None:
    """Draw a rounded rectangle with optional border."""
    r = rect
    if border > 0 and border_color:
        inner = r.inflate(-border * 2, -border * 2)
        pygame.draw.rect(surface, border_color, r, border_radius=radius)
        pygame.draw.rect(surface, color, inner,
                         border_radius=max(0, radius - border))
    else:
        pygame.draw.rect(surface, color, r, border_radius=radius)


def glow_surface(radius: int, color: tuple[int, int, int],
                 max_alpha: int = 180, falloff: float = 1.5) -> pygame.Surface:
    """Create a circular glow texture with gaussian-like falloff."""
    key = (radius, color, max_alpha, falloff)
    return _glow_inner(key)


@lru_cache(maxsize=128)
def _glow_inner(key: tuple) -> pygame.Surface:
    radius, color, max_alpha, falloff = key
    size = radius * 2
    surf = make_surf(size, size)
    cx = cy = radius
    for y in range(size):
        for x in range(size):
            dx, dy = x - cx, y - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > radius:
                continue
            t = 1.0 - (dist / radius)
            a = int(max_alpha * (t ** falloff))
            if a > 0:
                surf.set_at((x, y), (color[0], color[1], color[2], a))
    return surf


def w2p(value: float) -> int:
    """Convert a world-cell coordinate to a pixel coordinate."""
    return int(value * PX_PER_CELL)


def w2pf(value: float) -> float:
    """Convert a world-cell coordinate to a pixel coordinate (float)."""
    return value * PX_PER_CELL


def p2w(value: int) -> float:
    """Convert a pixel coordinate back to a world-cell coordinate."""
    return value / PX_PER_CELL


def perlin_noise(width: int, height: int, scale: float = 0.02,
                 octaves: int = 4, seed: int | None = None,
                 normalize: bool = True) -> list[list[float]]:
    """Generate Perlin-like noise using layered value noise (pure Python)."""
    rng = random.Random(seed)

    def _smooth_noise(x: float, y: float, period: int) -> float:
        sx = x / period
        sy = y / period
        ix, iy = int(sx), int(sy)
        fx, fy = sx - ix, sy - iy
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)

        def _g(ox: int, oy: int) -> float:
            h = hash((ix + ox, iy + oy, seed if seed else 0))
            return (h & 65535) / 32768.0 - 1.0

        v00, v10 = _g(0, 0), _g(1, 0)
        v01, v11 = _g(0, 1), _g(1, 1)
        top = v00 + (v10 - v00) * fx
        bot = v01 + (v11 - v01) * fx
        return top + (bot - top) * fy

    grid: list[list[float]] = [[0.0] * width for _ in range(height)]
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0

    for _ in range(octaves):
        period = max(2, int(1 / (scale * frequency)))
        for y in range(height):
            for x in range(width):
                grid[y][x] += _smooth_noise(x, y, period) * amplitude
        max_val += amplitude
        amplitude *= 0.5
        frequency *= 2.0

    if normalize and max_val > 0:
        for y in range(height):
            for x in range(width):
                grid[y][x] = grid[y][x] / max_val * 0.5 + 0.5

    return grid


__all__ = [
    "make_surf", "lerp_color", "color_with_alpha",
    "round_rect", "glow_surface",
    "w2p", "w2pf", "p2w", "perlin_noise",
    "PX_PER_CELL",
]
