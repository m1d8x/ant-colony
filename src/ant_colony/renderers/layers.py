"""World render layers — ground, obstacles, nest, food, ants.

Each layer function accepts a ``pygame.Surface`` (the viewport), a
``pysimengine.World`` instance, and camera offsets in pixels.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache

import pygame

from ant_colony.renderers.utils import (
    make_surf, lerp_color, w2p, w2pf, p2w, perlin_noise, PX_PER_CELL,
)
from ant_colony.renderers.sprites import ant_surface, ant_surface_with_food, ANT_SPRITE_SIZE
from ant_colony.pysimengine.core import World

WINDOW_W = 1280
WINDOW_H = 800


# ── Ground Layer ──────────────────────────────────────────────────────

GROUND_COLORS = ((194, 178, 128), (162, 145, 98), (140, 118, 72))


@lru_cache(maxsize=2)
def _build_ground_texture(world_w: int, world_h: int,
                          seed: int = 42) -> pygame.Surface:
    """Pre-render the full-world ground texture using Perlin noise
    (1 pixel per world cell; scaled to display resolution at render time).
    """
    noise = perlin_noise(world_w, world_h, scale=0.03, octaves=4,
                         seed=seed)
    surf = make_surf(world_w, world_h, alpha=False)
    pix = pygame.PixelArray(surf)
    c1, c2, c3 = GROUND_COLORS
    try:
        for y in range(world_h):
            for x in range(world_w):
                n = noise[y][x]
                if n < 0.33:
                    c = lerp_color(c2, c3, n / 0.33)
                elif n < 0.66:
                    t = (n - 0.33) / 0.33
                    c = lerp_color(c2, c1, t)
                else:
                    t = (n - 0.66) / 0.34
                    c = lerp_color(c1, (220, 210, 180), t)
                pix[x, y] = (c[0] << 16) | (c[1] << 8) | c[2]
    finally:
        del pix
    return surf


def render_ground(surf: pygame.Surface, world: World,
                  cam_px: float, cam_py: float) -> None:
    """Blit the visible ground scaled to pixel dimensions."""
    ground = _build_ground_texture(world.width, world.height)
    px_w = world.width * PX_PER_CELL
    px_h = world.height * PX_PER_CELL
    scaled = pygame.transform.scale(ground, (px_w, px_h))
    surf.blit(scaled, (-cam_px, -cam_py))


# ── Obstacle Layer ────────────────────────────────────────────────────

ROCK_BASE = (100, 95, 85)
ROCK_HL = (140, 130, 115)
ROCK_SHADOW = (60, 55, 50)
WATER_COLOR = (40, 100, 180)
WATER_ALPHA = 130


@lru_cache(maxsize=256)
def _build_rock_surface(seed: int) -> pygame.Surface:
    """Pre-render a single rock tile (1 world cell)."""
    rng = random.Random(seed)
    s = make_surf(PX_PER_CELL + 4, PX_PER_CELL + 4, alpha=True)
    cx, cy = (PX_PER_CELL + 4) // 2, (PX_PER_CELL + 4) // 2
    r = PX_PER_CELL // 2 - 1
    n_pts = 6 + rng.randint(0, 3)
    pts = []
    for i in range(n_pts):
        angle = (i / n_pts) * math.pi * 2
        radius = r * (0.6 + rng.random() * 0.4)
        pts.append((cx + int(math.cos(angle) * radius),
                    cy + int(math.sin(angle) * radius)))
    shadow = [(x + 2, y + 2) for x, y in pts]
    pygame.draw.polygon(s, ROCK_SHADOW + (80,), shadow)
    pygame.draw.polygon(s, ROCK_BASE, pts)
    hl = pts[:n_pts // 2 + 1]
    if len(hl) >= 3:
        pygame.draw.polygon(s, ROCK_HL + (50,), hl)
    pygame.draw.polygon(s, ROCK_BASE + (180,), pts, 1)
    return s


def render_obstacles(surf: pygame.Surface, world: World,
                     cam_px: float, cam_py: float) -> None:
    """Draw obstacle cells as rock tiles."""
    margin = 2
    start_x = max(0, int(p2w(cam_px)) - margin)
    start_y = max(0, int(p2w(cam_py)) - margin)
    end_x = min(world.width, int(p2w(cam_px + WINDOW_W)) + margin)
    end_y = min(world.height, int(p2w(cam_py + WINDOW_H)) + margin)
    ps = PX_PER_CELL

    for wy in range(start_y, end_y):
        for wx in range(start_x, end_x):
            if not world.obstacles[wx][wy]:
                continue
            sx = wx * ps - cam_px
            sy = wy * ps - cam_py
            seed = hash((wx, wy, 42)) & 0xFFFF
            rock_surf = _build_rock_surface(seed)
            surf.blit(rock_surf, (sx, sy))


# ── Nest Layer ────────────────────────────────────────────────────────

NEST_ENTRANCE = (130, 90, 60)
NEST_DARK = (30, 22, 15)


def render_nest(surf: pygame.Surface, world: World,
                cam_px: float, cam_py: float) -> None:
    """Draw nest entrance at (nest_x, nest_y) as an earthy mound."""
    nx = w2pf(world.nest_x) - cam_px
    ny = w2pf(world.nest_y) - cam_py
    ps = PX_PER_CELL

    # Mound shadow
    pygame.draw.ellipse(surf, (50, 35, 20, 60),
                        (nx - ps * 1.5 + 3, ny - ps * 0.75 + 3,
                         ps * 3, ps * 1.5))
    # Mound
    pygame.draw.ellipse(surf, NEST_ENTRANCE,
                        (nx - ps * 1.5, ny - ps * 0.75,
                         ps * 3, ps * 1.5))
    # Entrance hole
    hole_r = ps * 0.35
    pygame.draw.ellipse(surf, NEST_DARK,
                        (nx - hole_r, ny - hole_r * 0.5,
                         hole_r * 2, hole_r))


# ── Food Layer ────────────────────────────────────────────────────────

FOOD_BUSH = (60, 180, 40)
FOOD_BERRY = (220, 50, 50)


@lru_cache(maxsize=32)
def _build_food_surface(size: int) -> pygame.Surface:
    """Pre-render a food bush sprite."""
    s = make_surf(size + 8, size + 8, alpha=True)
    cx, cy = (size + 8) // 2, (size + 8) // 2
    r = size // 2
    for i in range(6):
        angle = i * 1.1
        ox = int(math.cos(angle) * r * 0.4)
        oy = int(math.sin(angle) * r * 0.3)
        br = int(r * (0.5 + (i % 3) * 0.2))
        c = lerp_color(FOOD_BUSH, (80, 210, 60), 0.3 if i % 2 else 0.0)
        pygame.draw.circle(s, c, (cx + ox, cy + oy), max(3, br))
    for _ in range(4):
        bx = cx + int(r * 0.4 * math.cos(_ * 1.7))
        by = cy + int(r * 0.3 * math.sin(_ * 1.3))
        pygame.draw.circle(s, FOOD_BERRY, (bx + 1, by + 1), 2)
    return s


def render_food(surf: pygame.Surface, world: World,
                cam_px: float, cam_py: float) -> None:
    """Draw food sources as bright bushes that shrink as consumed."""
    ps = PX_PER_CELL
    margin = 60
    for fx, fy, amount in world.food_sources:
        sx = w2pf(fx) - cam_px
        sy = w2pf(fy) - cam_py
        if sx < -margin or sx > WINDOW_W + margin:
            continue
        if sy < -margin or sy > WINDOW_H + margin:
            continue
        if amount <= 0:
            continue
        size = int(ps * max(0.4, min(1.0, amount / 50)))
        bush = _build_food_surface(size)
        surf.blit(bush, (sx - bush.get_width() // 2,
                         sy - bush.get_height() // 2))


# ── Ant Layer ─────────────────────────────────────────────────────────

def render_ants(surf: pygame.Surface, world: World,
                cam_px: float, cam_py: float) -> None:
    """Draw ant sprites by role, rotated by direction, with food particles."""
    margin = 30
    for ant in world.ants:
        if not ant.alive:
            continue
        sx = w2pf(ant.x) - cam_px
        sy = w2pf(ant.y) - cam_py
        if sx < -margin or sx > WINDOW_W + margin:
            continue
        if sy < -margin or sy > WINDOW_H + margin:
            continue

        base = ant_surface(ant.role, ANT_SPRITE_SIZE)
        rotated = pygame.transform.rotate(base,
                                          math.degrees(ant.direction))
        rot_rect = rotated.get_rect(center=(sx, sy))
        surf.blit(rotated, rot_rect.topleft)

        if ant.food_carrying > 0:
            _ant, p_surf = ant_surface_with_food(ant.role)
            p_rot = pygame.transform.rotate(p_surf,
                                            math.degrees(ant.direction))
            offset_x = -int(math.cos(ant.direction) * 8)
            offset_y = -int(math.sin(ant.direction) * 8)
            pr = p_rot.get_rect(center=(sx + offset_x, sy + offset_y))
            surf.blit(p_rot, pr.topleft)
