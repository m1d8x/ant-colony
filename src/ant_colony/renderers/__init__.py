"""
Renderers for ant colony simulations.

Provides pygame (interactive) and headless (batch) rendering backends
that understand the pysimengine.World / pysimengine.Agent data model.
"""

from __future__ import annotations

import math
from typing import Any


# ═════════════════════════════════════════════════════════════════════════════
#  Base renderer
# ═════════════════════════════════════════════════════════════════════════════


class BaseRenderer:
    """Abstract renderer interface."""

    def render(self, world, step: int):
        raise NotImplementedError

    def handle_events(self) -> bool:
        return True

    def close(self):
        pass


# ═════════════════════════════════════════════════════════════════════════════
#  Colour palette
# ═════════════════════════════════════════════════════════════════════════════

_COLOR_BG = (15, 15, 20)
_COLOR_OBSTACLE = (60, 45, 35)
_COLOR_NEST = (100, 70, 30)
_COLOR_NEST_RING = (140, 110, 60)
_COLOR_FOOD = (50, 200, 50)
_COLOR_FOOD_DEPLETED = (60, 60, 60)
_COLOR_DEAD = (80, 40, 40)
_COLOR_HUD = (200, 200, 200)
_COLOR_PHEROMONE_FOOD = (0, 200, 0)
_COLOR_PHEROMONE_HOME = (0, 100, 200)

# Role-to-colour mapping (body, accent)
_ROLE_COLORS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "forager": ((140, 90, 40), (200, 140, 60)),
    "builder": ((80, 55, 30), (120, 90, 50)),
    "soldier": ((140, 30, 30), (190, 60, 60)),
    "queen":   ((120, 70, 160), (160, 110, 200)),
}


def _agent_color(agent) -> tuple[int, int, int]:
    """Get body colour for an agent based on role, with fallback."""
    colors = _ROLE_COLORS.get(agent.role)
    if colors is not None:
        return colors[0]
    return agent.color if hasattr(agent, "color") else (200, 200, 200)


def _agent_accent(agent) -> tuple[int, int, int]:
    """Get accent colour for an agent (brighter variant)."""
    colors = _ROLE_COLORS.get(agent.role)
    if colors is not None:
        return colors[1]
    return (255, 255, 255)


# ═════════════════════════════════════════════════════════════════════════════
#  PyGame Renderer
# ═════════════════════════════════════════════════════════════════════════════


class PyGameRenderer(BaseRenderer):
    """Pygame-based interactive renderer.

    Renders obstacles, food sources, pheromone heatmaps, nest, ants,
    and a heads-up display.
    """

    def __init__(self, world, config: dict[str, Any]):
        self.world = world
        self.config = config

        self.window_w = int(config.get("width", 1200))
        self.window_h = int(config.get("height", 800))
        self.title = config.get("title", "Ant Colony Simulation")
        self.fps = config.get("fps", 60)

        # Compute scale so the world grid fits the window
        self.scale = min(
            self.window_w / max(world.width, 1),
            self.window_h / max(world.height, 1),
        )

        self._screen = None
        self._clock = None
        self._font = None
        self._running = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def _init_pygame(self):
        import pygame

        pygame.init()
        self._screen = pygame.display.set_mode(
            (self.window_w, self.window_h),
        )
        pygame.display.set_caption(self.title)
        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14)
        self._running = True

    def handle_events(self) -> bool:
        if self._screen is None:
            self._init_pygame()

        import pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self._running = False
                elif event.key == pygame.K_SPACE:
                    pass  # pause toggled externally
        return self._running

    def close(self):
        import pygame

        self._running = False
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass

    # ── Rendering ────────────────────────────────────────────────────────

    def render(self, world, step: int):
        if self._screen is None:
            self._init_pygame()

        import pygame

        screen = self._screen
        s = self.scale
        screen.fill(_COLOR_BG)

        # ── Obstacles ──────────────────────────────────────────────────
        for x in range(world.width):
            for y in range(world.height):
                if world.obstacles[x][y]:
                    rect = (
                        int(x * s),
                        int(y * s),
                        max(1, int(math.ceil(s))),
                        max(1, int(math.ceil(s))),
                    )
                    pygame.draw.rect(screen, _COLOR_OBSTACLE, rect)

        # ── Food sources ───────────────────────────────────────────────
        for fx, fy, amount in world.food_sources:
            px = int(fx * s)
            py = int(fy * s)
            radius = max(2, int(3 * s))
            if amount > 0:
                pygame.draw.circle(screen, _COLOR_FOOD, (px, py), radius)
                # Inner fill proportional to remaining amount
                fill_r = max(1, int(radius * min(1.0, amount / 200.0)))
                inner = (100, 255, 100)
                pygame.draw.circle(screen, inner, (px, py), fill_r)
            else:
                pygame.draw.circle(screen, _COLOR_FOOD_DEPLETED, (px, py), radius, 1)

        # ── Pheromone heatmap ──────────────────────────────────────────
        # food_pheromone — green tint
        self._draw_pheromone_layer(
            screen, world.food_pheromone, s, _COLOR_PHEROMONE_FOOD
        )
        # home_pheromone — blue tint
        self._draw_pheromone_layer(
            screen, world.home_pheromone, s, _COLOR_PHEROMONE_HOME
        )

        # ── Nest ───────────────────────────────────────────────────────
        nx = int(world.nest_x * s)
        ny = int(world.nest_y * s)
        nest_r = max(4, int(6 * s))
        pygame.draw.circle(screen, _COLOR_NEST_RING, (nx, ny), nest_r, 2)
        pygame.draw.circle(screen, _COLOR_NEST, (nx, ny), max(2, int(3 * s)))

        # ── Dead bodies ────────────────────────────────────────────────
        for agent in world.ants:
            if agent.alive:
                continue
            dx = int(agent.x * s)
            dy = int(agent.y * s)
            pygame.draw.circle(screen, _COLOR_DEAD, (dx, dy), max(1, int(1.5 * s)))

        # ── Ants (alive) ───────────────────────────────────────────────
        for agent in world.ants:
            if not agent.alive:
                continue
            ax = int(agent.x * s)
            ay = int(agent.y * s)
            r = max(2, int(2 * s))
            color = _agent_color(agent)
            pygame.draw.circle(screen, color, (ax, ay), r)

            # Direction indicator
            if agent.speed > 0.1:
                dx = int(math.cos(agent.direction) * 4 * s)
                dy = int(math.sin(agent.direction) * 4 * s)
                pygame.draw.line(
                    screen, (255, 255, 255), (ax, ay), (ax + dx, ay + dy), 1
                )

            # Food-carrying indicator
            if agent.food_carrying > 0:
                accent = _agent_accent(agent)
                pygame.draw.circle(screen, accent, (ax, ay), r + 2, 1)

        # ── HUD ────────────────────────────────────────────────────────
        if self._font is not None:
            alive = sum(1 for a in world.ants if a.alive)
            total = len(world.ants)
            food_piles = sum(1 for _, _, a in world.food_sources if a > 0)
            roles: dict[str, int] = {}
            for a in world.ants:
                if a.alive:
                    roles[a.role] = roles.get(a.role, 0) + 1
            role_str = " ".join(f"{k}={v}" for k, v in sorted(roles.items()))
            hud = (
                f"Step: {step}  |  Ants: {alive}/{total}  |  "
                f"Food: {food_piles}  |  Store: {world.colony_food_store:.0f}  |  "
                f"[{role_str}]"
            )
            surf = self._font.render(hud, True, _COLOR_HUD)
            screen.blit(surf, (8, 8))

        pygame.display.flip()
        if self._clock is not None:
            self._clock.tick(self.fps)

    @staticmethod
    def _draw_pheromone_layer(
        screen,
        grid: list[list[float]],
        scale: float,
        color: tuple[int, int, int],
    ):
        """Draw a single pheromone grid as a translucent overlay."""
        import pygame

        h = len(grid[0]) if grid else 0
        w = len(grid)
        if w == 0 or h == 0:
            return
        cr, cg, cb = color
        for x in range(w):
            for y in range(h):
                v = grid[x][y]
                if v > 0.01:
                    alpha = min(120, max(1, int(v * 120)))
                    surf = pygame.Surface(
                        (max(1, int(math.ceil(scale))),
                         max(1, int(math.ceil(scale)))),
                        pygame.SRCALPHA,
                    )
                    surf.fill((cr, cg, cb, alpha))
                    screen.blit(surf, (int(x * scale), int(y * scale)))


# ═════════════════════════════════════════════════════════════════════════════
#  Headless Renderer
# ═════════════════════════════════════════════════════════════════════════════


class HeadlessRenderer(BaseRenderer):
    """Non-visual renderer for batch/headless simulations.

    Logs colony stats to stdout at a configurable interval.
    """

    def __init__(self, world, config: dict[str, Any]):
        self.world = world
        self.config = config
        self._log_interval = config.get("log_interval", 50)
        self._last_log_step = -1

    def render(self, world, step: int):
        if step - self._last_log_step >= self._log_interval or step == 0:
            self._last_log_step = step
            alive = sum(1 for a in world.ants if a.alive)
            total = len(world.ants)
            food_count = sum(1 for _, _, a in world.food_sources if a > 0)
            roles: dict[str, int] = {}
            for a in world.ants:
                if a.alive:
                    roles[a.role] = roles.get(a.role, 0) + 1
            role_str = " ".join(f"{k}={v}" for k, v in sorted(roles.items()))
            print(
                f"[step {step:6d}]  ants: {alive:3d}/{total:3d}  "
                f"food piles: {food_count:2d}  "
                f"store: {world.colony_food_store:7.1f}  "
                f"[{role_str}]",
                flush=True,
            )

    def close(self):
        print("[headless] simulation complete.", flush=True)
