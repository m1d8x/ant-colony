"""Ant sprite generation — directional sprites per role with food particles.

Role colours (body, highlight):
  forager = brown / orange
  builder = dark brown / brown
  soldier = dark red / red
  queen   = purple / light purple
"""
from __future__ import annotations

import math
from functools import lru_cache

import pygame

from ant_colony.renderers.utils import make_surf


ANT_COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "forager": ((140, 90, 40), (200, 140, 60)),     # brown / orange
    "builder": ((80, 55, 30), (120, 90, 50)),        # dark brown / brown
    "soldier": ((140, 30, 30), (190, 60, 60)),        # dark red / red
    "queen":   ((120, 70, 160), (160, 110, 200)),     # purple / light purple
}

ANT_SPRITE_SIZE = 14


def ant_surface(role: str, size: int = ANT_SPRITE_SIZE) -> pygame.Surface:
    """Return a pre-rendered ant sprite facing right (angle = 0).

    The caller rotates to match the ant's direction.
    """
    return _build_ant((role, size))


def ant_surface_with_food(role: str,
                          size: int = ANT_SPRITE_SIZE
                          ) -> tuple[pygame.Surface, pygame.Surface]:
    """Return (ant_surf, food_particle_surf) for a food-carrying ant."""
    ant = _build_ant((role, size))
    ps = make_surf(size + 14, size + 14, alpha=True)
    px, py = size // 2 - 1, size // 2 + 5
    for i in range(4):
        ox = int(math.sin(i * 2.5) * 5)
        oy = int(math.cos(i * 1.8) * 5) - 3
        r = 3 - i // 2
        a = 200 - i * 40
        pygame.draw.circle(ps, (220, 80, 50, a), (px + ox, py + oy), r)
    return ant, ps


@lru_cache(maxsize=16)
def _build_ant(key: tuple[str, int]) -> pygame.Surface:
    role, size = key
    body_color, leg_color = ANT_COLORS.get(role, ANT_COLORS["forager"])
    s = make_surf(size + 6, size + 6, alpha=True)
    cx, cy = (size + 6) // 2, (size + 6) // 2
    hl = int(size * 0.7)      # half-length of body
    r = max(2, size // 5)     # half-width of body

    # Body segments
    # Abdomen (rear)
    pygame.draw.ellipse(s, body_color,
                        (cx - hl // 3, cy - r, hl // 2, r * 2))
    # Thorax (middle) — slightly darker
    darker = (max(0, body_color[0] - 25),
              max(0, body_color[1] - 25),
              max(0, body_color[2] - 25))
    pygame.draw.ellipse(s, darker,
                        (cx - hl // 6, cy - r + 1, hl // 3, r * 2 - 2))
    # Head
    head_r = max(2, r - 1)
    pygame.draw.circle(s, body_color, (cx + hl // 3, cy), head_r)

    # Eyes
    eye_color = (200, 180, 120) if role != "soldier" else (220, 50, 50)
    eye_off = hl // 3 + head_r // 2
    pygame.draw.circle(s, eye_color, (cx + eye_off, cy - 2), 1)
    pygame.draw.circle(s, eye_color, (cx + eye_off, cy + 2), 1)

    # Legs
    leg_angles = [-0.6, 0.0, 0.6]
    for i, la in enumerate(leg_angles):
        for side in (-1, 1):
            lx = cx - hl // 8 + i * hl // 5
            ly = cy + side * r
            ex = lx + int(math.cos(la) * (r + 2)) * side
            ey = ly + int(math.sin(la) * (r + 2))
            pygame.draw.line(s, leg_color, (lx, ly), (ex, ey), 1)

    # Antennae
    for side in (-1, 1):
        start = (cx + hl // 3, cy)
        end = (cx + hl // 3 + 4, cy + side * 4)
        pygame.draw.line(s, leg_color, start, end, 1)

    # Role-specific markings
    if role == "soldier":
        for side in (-1, 1):
            mx = cx + hl // 3 + head_r
            my = cy + side * head_r
            pygame.draw.line(s, (180, 40, 40),
                             (mx - 1, my), (mx + 2, my + side * 3), 2)
    elif role == "queen":
        pygame.draw.ellipse(s, (160, 130, 200, 120),
                            (cx - hl // 3 - 1, cy - r - 1,
                             hl // 2 + 2, r * 2 + 2), 1)
    elif role == "builder":
        for side in (-1, 1):
            mx = cx + hl // 3 + head_r
            my = cy + side * 1
            pygame.draw.circle(s, (100, 80, 50), (mx, my), 1)
    elif role == "forager":
        pygame.draw.ellipse(s, (160, 110, 50, 80),
                            (cx - hl // 3 - 1, cy - r - 1,
                             hl // 2 + 2, r * 2 + 2), 1)

    return s
