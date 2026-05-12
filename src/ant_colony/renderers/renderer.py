"""Main renderer — orchestrates all layers, camera, frame pacing, and HUD.

Usage::

    renderer = Renderer()
    renderer.setup()
    while renderer.running:
        renderer.begin_frame()
        world.tick()
        renderer.render(world)
        renderer.end_frame()
    renderer.cleanup()
"""
from __future__ import annotations

import math
from functools import lru_cache

import pygame

from ant_colony.renderers.utils import make_surf, PX_PER_CELL
from ant_colony.renderers.layers import (
    render_ground, render_obstacles, render_nest, render_food,
    WINDOW_W, WINDOW_H,
)
from ant_colony.renderers.pheromone import PheromoneLayer
from ant_colony.renderers.hud import HUD
from ant_colony.renderers.sprites import ant_surface, ant_surface_with_food, ANT_SPRITE_SIZE
from ant_colony.pysimengine.core import World

CAMERA_SMOOTH = 0.08
CAMERA_EDGE_SCROLL = 30       # px from window edge to trigger scroll
CAMERA_SCROLL_SPEED = 8       # px per frame

# ── Cached ant rotation ──────────────────────────────────────────────
# For 500+ ants at 60 fps we cache rotated sprites by (role, degrees).
# 4 roles × ~72 distinct angles (every 5°) = 288 surfaces — negligible
# memory, huge perf win over per-frame pygame.transform.rotate().

_DEGREE_BUCKET = 5  # quantise to 5° increments for cache efficiency


def _cached_ant(role: str, deg: int) -> pygame.Surface:
    """Return a cached rotated ant sprite at *deg* degrees counter-clockwise."""
    bucket = (deg // _DEGREE_BUCKET) * _DEGREE_BUCKET
    return _build_rotated_ant((role, bucket % 360))


def _cached_ant_food(role: str, deg: int) -> pygame.Surface:
    """Return a cached rotated food-particle sprite."""
    bucket = (deg // _DEGREE_BUCKET) * _DEGREE_BUCKET
    return _build_rotated_food((role, bucket % 360))


@lru_cache(maxsize=512)
def _build_rotated_ant(key: tuple[str, int]) -> pygame.Surface:
    role, deg = key
    base = ant_surface(role, ANT_SPRITE_SIZE)
    return pygame.transform.rotate(base, deg)


@lru_cache(maxsize=512)
def _build_rotated_food(key: tuple[str, int]) -> pygame.Surface:
    role, deg = key
    _, food_surf = ant_surface_with_food(role)
    return pygame.transform.rotate(food_surf, deg)


# ── Vignette overlay ─────────────────────────────────────────────────

_VIGNETTE_CACHE: dict[tuple[int, int], pygame.Surface] = {}


def _build_vignette(w: int, h: int) -> pygame.Surface:
    """Dark-rimmed vignette overlay for a premium cinematic look."""
    key = (w, h)
    if key in _VIGNETTE_CACHE:
        return _VIGNETTE_CACHE[key]
    surf = make_surf(w, h)
    r_inner = min(w, h) * 0.35  # inner radius fraction
    r_outer = min(w, h) * 0.55
    cx, cy = w // 2, h // 2
    for py in range(h):
        for px in range(w):
            dx, dy = px - cx, py - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist <= r_inner:
                continue
            t = min(1.0, (dist - r_inner) / (r_outer - r_inner))
            a = int(t * t * 90)  # quadratic falloff, max 90 alpha
            if a > 0:
                surf.set_at((px, py), (0, 0, 0, a))
    _VIGNETTE_CACHE[key] = surf
    return surf


# ── Renderer ─────────────────────────────────────────────────────────


class Renderer:
    """Top-level renderer orchestrator for the ant colony simulation."""

    def __init__(self, width: int = 1280, height: int = 800,
                 fullscreen: bool = False) -> None:
        self.width = width
        self.height = height
        self.fullscreen = fullscreen

        self._window: pygame.Surface | None = None
        self._clock: pygame.Clock | None = None
        self._running = False

        # Camera offset in pixels (top-left of viewport in world pixel space)
        self.cam_x: float = 0.0
        self.cam_y: float = 0.0
        self._cam_target_x: float = 0.0
        self._cam_target_y: float = 0.0

        # Layers (created lazily in setup / set_world)
        self.hud: HUD | None = None
        self._pheromone: PheromoneLayer | None = None
        self._world_surf: pygame.Surface | None = None
        self._world_w: int = 0
        self._world_h: int = 0
        self._vignette: pygame.Surface | None = None

    def setup(self) -> None:
        """Initialise pygame and create the window."""
        pygame.init()
        flags = pygame.DOUBLEBUF
        if self.fullscreen:
            flags |= pygame.FULLSCREEN
        self._window = pygame.display.set_mode((self.width, self.height), flags)
        pygame.display.set_caption("Ant Colony Simulation")
        self._clock = pygame.time.Clock()
        self._running = True
        self.hud = HUD(window_width=self.width)
        self._vignette = _build_vignette(self.width, self.height)

    def set_world(self, world: World) -> None:
        """Configure renderer for a given world's dimensions."""
        self._world_w = world.width
        self._world_h = world.height
        self._world_surf = make_surf(self.width, self.height, alpha=False)

        # Centre camera on nest initially
        nx = world.nest_x * PX_PER_CELL
        ny = world.nest_y * PX_PER_CELL
        self.cam_x = nx - self.width // 2
        self.cam_y = ny - self.height // 2
        self._cam_target_x = self.cam_x
        self._cam_target_y = self.cam_y

        self._pheromone = PheromoneLayer(world.width, world.height)

    def begin_frame(self) -> None:
        """Start a new frame: poll events, compute dt."""
        if self._clock is None or self.hud is None:
            return
        self._clock.tick(60)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                    return
                elif event.key == pygame.K_SPACE:
                    self.hud.paused = not self.hud.paused
                elif event.key == pygame.K_TAB:
                    self.hud.show_pheromones = not self.hud.show_pheromones
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mx, my = event.pos
                    if my < self.hud.height:
                        self.hud.handle_click(mx, my)

        # Mouse position for HUD hover effects
        self.hud.update_mouse(pygame.mouse.get_pos())

        # Edge-scroll camera (only when not paused)
        if not self.hud.paused:
            mx, my = pygame.mouse.get_pos()
            if mx < CAMERA_EDGE_SCROLL:
                self._cam_target_x -= CAMERA_SCROLL_SPEED
            elif mx > self.width - CAMERA_EDGE_SCROLL:
                self._cam_target_x += CAMERA_SCROLL_SPEED
            if my < CAMERA_EDGE_SCROLL:
                self._cam_target_y -= CAMERA_SCROLL_SPEED
            elif my > self.height - CAMERA_EDGE_SCROLL:
                self._cam_target_y += CAMERA_SCROLL_SPEED

        # Clamp camera to world bounds
        px_w = self._world_w * PX_PER_CELL
        px_h = self._world_h * PX_PER_CELL
        self._cam_target_x = max(0, min(self._cam_target_x,
                                        px_w - self.width))
        self._cam_target_y = max(0, min(self._cam_target_y,
                                        px_h - self.height))

        # Smooth camera
        self.cam_x += (self._cam_target_x - self.cam_x) * CAMERA_SMOOTH
        self.cam_y += (self._cam_target_y - self.cam_y) * CAMERA_SMOOTH

    def render(self, world: World) -> None:
        """Render all layers for the current world state.

        Layer order (bottom → top):
          1. Ground (Perlin noise)
          2. Obstacles (rocks, water pools)
          3. Nest (entrance, rooms)
          4. Food (bushes)
          5. Pheromones (additive blend overlay, camera-offset)
          6. Ants (directional sprites with rotation cache)
          7. Vignette (cinematic dark rim)
          8. HUD (always on top)
        """
        if self._window is None or self._world_surf is None:
            return
        if self._pheromone is None:
            return

        surf = self._world_surf

        # Clear the world surface with ground colour first
        surf.fill((194, 178, 128))

        # 1. Ground
        render_ground(surf, world, self.cam_x, self.cam_y)

        # 2. Obstacles
        render_obstacles(surf, world, self.cam_x, self.cam_y)

        # 3. Nest
        render_nest(surf, world, self.cam_x, self.cam_y)

        # 4. Food
        render_food(surf, world, self.cam_x, self.cam_y)

        # 5. Pheromones (additive blend, camera-offset)
        if self.hud.show_pheromones:
            self._pheromone.update(
                world.food_pheromone, world.home_pheromone,
                self.cam_x, self.cam_y,
            )
            # IMPORTANT: blit at camera offset so the overlay scrolls
            # with the world — the pheromone surface is world-sized.
            surf.blit(self._pheromone.surface,
                      (-self.cam_x, -self.cam_y),
                      special_flags=pygame.BLEND_ADD)

        # 6. Ants (cached rotation for 500+ ant performance)
        self._render_ants_cached(surf, world)

        # Draw world surface to window
        self._window.blit(surf, (0, 0))

        # 7. Vignette (cinematic dark rim overlay)
        if self._vignette:
            self._window.blit(self._vignette, (0, 0))

        # 8. HUD (always on top)
        hud_surf = self.hud.render(
            colony_food=world.colony_food_store,
            population=len(world.ants),
            ants_alive=sum(1 for a in world.ants if a.alive),
            tick=world.tick_number,
            paused=self.hud.paused,
            speed=self.hud.speed,
            show_phero=self.hud.show_pheromones,
        )
        self._window.blit(hud_surf, (0, 0))

    def _render_ants_cached(self, surf: pygame.Surface,
                            world: World) -> None:
        """Draw ant sprites with cached rotated surfaces.

        Performance: caches (role, quantised_angle) → rotated surface
        so 500+ ants don't cause 500+ pygame.transform.rotate() calls
        per frame.
        """
        margin = 30
        for ant in world.ants:
            if not ant.alive:
                continue
            sx = ant.x * PX_PER_CELL - self.cam_x
            sy = ant.y * PX_PER_CELL - self.cam_y
            if sx < -margin or sx > self.width + margin:
                continue
            if sy < -margin or sy > self.height + margin:
                continue

            deg = int(math.degrees(ant.direction)) % 360
            rotated = _cached_ant(ant.role, deg)
            rot_rect = rotated.get_rect(center=(sx, sy))
            surf.blit(rotated, rot_rect.topleft)

            if ant.food_carrying > 0:
                p_rot = _cached_ant_food(ant.role, deg)
                offset_x = -int(math.cos(ant.direction) * 8)
                offset_y = -int(math.sin(ant.direction) * 8)
                pr = p_rot.get_rect(center=(sx + offset_x, sy + offset_y))
                surf.blit(p_rot, pr.topleft)

    def end_frame(self) -> None:
        """Flip the display buffer."""
        if self._window:
            pygame.display.flip()

    @property
    def running(self) -> bool:
        return self._running

    def cleanup(self) -> None:
        """Shut down pygame."""
        self._running = False
        pygame.quit()
