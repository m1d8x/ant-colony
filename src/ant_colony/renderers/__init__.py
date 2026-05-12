"""
Renderers for ant colony simulations.

Provides pygame (interactive) and headless (batch) rendering backends
that understand the pysimengine.World / pysimengine.Agent data model.

Enhanced visuals:
- Procedural terrain background with Perlin-noise elevation colours
- Smooth per-pixel pheromone rendering via numpy / surfarray
- Anti-aliased agent and food rendering via pygame.gfxdraw
- Heads-up display with live FPS, stats, and role breakdown
- Semi-transparent legend panel (bottom-right) with colour key
"""

from __future__ import annotations

import math
import random
import time
from typing import Any


# =============================================================================
#  Perlin noise — used once at init for the terrain background
# =============================================================================

def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6 - 15) + 10)


def _lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


def _grad(hash_val: int, x: float, y: float) -> float:
    h = hash_val & 3
    u = x if h < 2 else (-x if h == 2 else y)
    v = y if h == 0 else (-y if h == 1 else x)
    return u + v


class _PerlinNoise:
    """Simple 2-D Perlin noise. Deterministic for a given seed."""

    def __init__(self, seed: int = 0):
        p = list(range(256))
        rng = random.Random(seed)
        rng.shuffle(p)
        self.perm = p + p

    def noise2d(self, x: float, y: float) -> float:
        X = int(math.floor(x)) & 255
        Y = int(math.floor(y)) & 255
        xf = x - math.floor(x)
        yf = y - math.floor(y)
        u = _fade(xf)
        v = _fade(yf)
        aa = self.perm[self.perm[X] + Y]
        ab = self.perm[self.perm[X] + Y + 1]
        ba = self.perm[self.perm[X + 1] + Y]
        bb = self.perm[self.perm[X + 1] + Y + 1]
        x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1, yf), u)
        x2 = _lerp(_grad(ab, xf, yf - 1), _grad(bb, xf - 1, yf - 1), u)
        return _lerp(x1, x2, v)


# =============================================================================
#  Colour palette
# =============================================================================

# Terrain elevation → colour ramp (0.0 = lowland, 1.0 = peak)
_TERRAIN_RAMP: list[tuple[int, int, int]] = [
    (45, 70, 110),    # deep water
    (64, 100, 148),   # shallow water / marsh
    (130, 169, 107),  # grass / plains
    (156, 194, 97),   # light grass
    (140, 140, 85),   # dry grass
    (128, 110, 75),   # bare earth
    (169, 169, 169),  # highland
    (200, 200, 210),  # peak / snow
]

# Gameplay element colours
_COL_OBSTACLE_ROCK  = (85, 75, 60)
_COL_OBSTACLE_WATER = (45, 85, 160)
_COL_NEST           = (139, 95, 35)
_COL_NEST_GLOW      = (194, 155, 70)
_COL_FOOD_BUSH      = (60, 210, 70)
_COL_FOOD_MUSHROOM  = (170, 126, 70)
_COL_FOOD_CRYSTAL   = (90, 220, 235)
_COL_FOOD_DEPLETED  = (70, 70, 70)
_COL_HUD            = (230, 230, 235)
_COL_HUD_DIM        = (140, 140, 150)
_COL_PHEROMONE_FOOD = (0, 210, 90)
_COL_PHEROMONE_HOME = (70, 140, 255)

# Panel colours (RGBA)
_COL_PANEL_BG      = (8, 10, 14, 200)
_COL_PANEL_BORDER   = (60, 70, 90, 180)
_COL_PANEL_TITLE    = (210, 210, 230)
_COL_PAUSE_OVERLAY  = (0, 0, 0, 100)

# Role → (body_colour, accent_colour)
_ROLE_COLOURS: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "forager": ((180, 120, 50), (240, 180, 80)),
    "builder": ((100, 70, 35),  (160, 120, 70)),
    "soldier": ((190, 45, 45),  (250, 80, 80)),
    "queen":   ((145, 85, 200), (195, 135, 255)),
}

# Food type → display colour
_FOOD_COLOURS = {
    "bush":     _COL_FOOD_BUSH,
    "mushroom": _COL_FOOD_MUSHROOM,
    "crystal":  _COL_FOOD_CRYSTAL,
}

# Legend item definitions
_LEGEND_ITEMS = [
    ("Agent Roles",       "title",    None),
    ("  forager",         "dot",      _ROLE_COLOURS["forager"][0]),
    ("  builder",         "dot",      _ROLE_COLOURS["builder"][0]),
    ("  soldier",         "dot",      _ROLE_COLOURS["soldier"][0]),
    ("  queen",           "dot",      _ROLE_COLOURS["queen"][0]),
    (None,                "spacer",   None),
    ("Food Types",        "title",    None),
    ("  bush",            "dot",      _COL_FOOD_BUSH),
    ("  mushroom",        "dot",      _COL_FOOD_MUSHROOM),
    ("  crystal",         "dot",      _COL_FOOD_CRYSTAL),
    ("  depleted",        "ring",     _COL_FOOD_DEPLETED),
    (None,                "spacer",   None),
    ("Pheromone Trails",  "title",    None),
    ("  to food",         "bar",      _COL_PHEROMONE_FOOD),
    ("  to home",         "bar",      _COL_PHEROMONE_HOME),
    (None,                "spacer",   None),
    ("Obstacles",         "title",    None),
    ("  rock",            "rect",     _COL_OBSTACLE_ROCK),
    ("  water",           "rect",     _COL_OBSTACLE_WATER),
]


# =============================================================================
#  Base renderer
# =============================================================================

class BaseRenderer:
    """Abstract renderer interface."""

    def render(self, world, step: int):
        raise NotImplementedError

    def handle_events(self) -> bool:
        return True

    def close(self):
        pass


# =============================================================================
#  PyGame renderer
# =============================================================================

class PyGameRenderer(BaseRenderer):
    """Enhanced pygame renderer with terrain, smooth pheromones, HUD & legend."""

    def __init__(self, world, config: dict[str, Any]):
        self.world = world
        self.config = config

        self.window_w = int(config.get("width", 1200))
        self.window_h = int(config.get("height", 800))
        self.title = config.get("title", "Ant Colony Simulation")
        self.fps = int(config.get("fps", 60))

        self._screen = None
        self._clock = None
        self._font = None
        self._font_small = None
        self._font_large = None
        self._is_freetype = False
        self._running = False
        self._paused = False
        self._current_w = self.window_w
        self._current_h = self.window_h

        # Pre-rendered surfaces (created lazily at _init_pygame)
        self._bg_raw = None             # terrain + obstacles + nest (world res)
        self._legend_surface = None     # legend panel (fixed pixel size)

        # Lookup maps (built at init, from config params)
        self._food_type_map: dict[tuple[int, int], str] = {}
        self._obstacle_type_map: dict[tuple[int, int], str] = {}

        # FPS tracking
        self._fps_samples: list[float] = []
        self._current_fps = 0

    # ------------------------------------------------------------------
    #  Pygame initialisation (lazy — called from first render / event)
    # ------------------------------------------------------------------

    def _init_pygame(self):
        import pygame
        pygame.init()

        self._screen = pygame.display.set_mode(
            (self.window_w, self.window_h), pygame.RESIZABLE,
        )
        pygame.display.set_caption(self.title)
        self._clock = pygame.time.Clock()

        # Fonts — try freetype first, fall back to SysFont
        self._is_freetype = True
        try:
            import pygame.freetype
            pygame.freetype.init()
            self._font_large = pygame.freetype.SysFont("monospace", 15, bold=True)
            self._font = pygame.freetype.SysFont("monospace", 12)
            self._font_small = pygame.freetype.SysFont("monospace", 10)
        except ImportError:
            self._is_freetype = False
            self._font_large = pygame.font.SysFont("monospace", 15, bold=True)
            self._font = pygame.font.SysFont("monospace", 12)
            self._font_small = pygame.font.SysFont("monospace", 10)

        self._running = True

        # Build lookup maps and pre-rendered surfaces
        self._build_food_type_map()
        self._build_obstacle_type_map()
        self._pre_render_background()
        self._build_legend_surface()

    # ------------------------------------------------------------------
    #  Pre-rendering helpers
    # ------------------------------------------------------------------

    def _build_food_type_map(self):
        """Re-generate FoodManager to map (x,y) -> food type label."""
        from ant_colony.world.food import FoodManager
        w, h = self.world.width, self.world.height
        cx, cy = w // 2, h // 2
        nest_r = self.config.get("nest_start_size", 3)
        seed = self.config.get("seed", 42) + 1
        fm = FoodManager.generate(
            w, h,
            bush_count=int(self.config.get("bush_count", 12)),
            mushroom_count=int(self.config.get("mushroom_count", 8)),
            crystal_count=int(self.config.get("crystal_count", 4)),
            seed=seed,
            nest_zone=(cx, cy, nest_r),
        )
        self._food_type_map = {
            (p.x, p.y): p.food_type.label for p in fm.patches
        }

    def _build_obstacle_type_map(self):
        """Re-generate ObstacleGrid to map (x,y) -> 'rock' or 'water'."""
        from ant_colony.world.obstacles import ObstacleGrid
        w, h = self.world.width, self.world.height
        cx, cy = w // 2, h // 2
        nest_r = self.config.get("nest_start_size", 3)
        rock_count = int(
            self.config.get("rock_count",
                            self.config.get("obstacle_count", 25))
        )
        water_count = int(
            self.config.get("water_count",
                            self.config.get("water_body_count", 4))
        )
        og = ObstacleGrid.generate(
            w, h,
            rock_count=rock_count,
            water_count=water_count,
            seed=self.config.get("seed", 42),
            nest_zone=(cx, cy, nest_r),
        )
        self._obstacle_type_map = {}
        for x in range(w):
            for y in range(h):
                if og.obstacles[x][y]:
                    bid = og._blob_ids[x][y]
                    self._obstacle_type_map[(x, y)] = (
                        "water" if bid > rock_count else "rock"
                    )

    def _pre_render_background(self):
        """Build a single surface: terrain + obstacles + nest @ world res."""
        import pygame

        # Try numpy path; if unavailable fall back to PixelArray
        try:
            import numpy as np
            import pygame.surfarray
        except ImportError:
            self._bg_raw = self._pre_render_background_pixelarray()
            return

        w, h = self.world.width, self.world.height

        # --- elevation from config params ---
        noise_seed = self.config.get("seed", 42) + 2
        tscale = self.config.get("terrain_scale", 12.0)
        toct = int(self.config.get("terrain_octaves", 4))
        tpers = self.config.get("terrain_persistence", 0.5)

        noise = _PerlinNoise(noise_seed)
        elev = np.zeros((w, h), dtype=np.float64)
        for x in range(w):
            for y in range(h):
                nx, ny = x / tscale, y / tscale
                v, a, f = 0.0, 1.0, 1.0
                for _ in range(toct):
                    v += noise.noise2d(nx * f, ny * f) * a
                    a *= tpers
                    f *= 2.0
                elev[x, y] = v

        # normalise → [0, 1]
        mn, mx = elev.min(), elev.max()
        span = max(mx - mn, 1e-6)
        nv = (elev - mn) / span

        # ramp interpolation (vectorised)
        ramp = np.array(_TERRAIN_RAMP, dtype=np.float64)  # (N, 3)
        rn = len(_TERRAIN_RAMP)
        idx_f = nv * (rn - 1)                        # (w, h)
        idx_lo = np.floor(idx_f).astype(np.int32).clip(0, rn - 2)
        idx_hi = idx_lo + 1
        frac = (idx_f - idx_lo)[:, :, None]          # (w, h, 1)

        rgb = (ramp[idx_lo] + frac * (ramp[idx_hi] - ramp[idx_lo]))
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)  # (w, h, 3)

        # --- obstacle override (numpy masking) ---
        obs = np.array(self.world.obstacles, dtype=bool)  # (w, h)
        if self._obstacle_type_map:
            for (x, y), typ in self._obstacle_type_map.items():
                if typ == "water":
                    rgb[x, y] = _COL_OBSTACLE_WATER
                else:
                    rgb[x, y] = _COL_OBSTACLE_ROCK
        else:
            rgb[obs] = _COL_OBSTACLE_ROCK

        self._bg_raw = pygame.surfarray.make_surface(rgb)

        # --- draw nest on top ---
        nx, ny = int(self.world.nest_x), int(self.world.nest_y)
        nr = self.config.get("nest_start_size", 3)
        self._draw_circle(self._bg_raw, nx, ny, max(2, nr), _COL_NEST)
        self._draw_circle(
            self._bg_raw, nx, ny, max(3, nr + 1), _COL_NEST_GLOW, 1,
        )

    def _pre_render_background_pixelarray(self):
        """Fallback when numpy is unavailable."""
        import pygame
        w, h = self.world.width, self.world.height
        surf = pygame.Surface((w, h))
        px = pygame.PixelArray(surf)

        noise_seed = self.config.get("seed", 42) + 2
        tscale = self.config.get("terrain_scale", 12.0)
        toct = int(self.config.get("terrain_octaves", 4))
        tpers = self.config.get("terrain_persistence", 0.5)

        noise = _PerlinNoise(noise_seed)
        elev = [[0.0] * h for _ in range(w)]
        mn, mx = float("inf"), float("-inf")
        for x in range(w):
            for y in range(h):
                nx, ny = x / tscale, y / tscale
                v, a, f = 0.0, 1.0, 1.0
                for _ in range(toct):
                    v += noise.noise2d(nx * f, ny * f) * a
                    a *= tpers
                    f *= 2.0
                elev[x][y] = v
                if v < mn:
                    mn = v
                if v > mx:
                    mx = v
        span = max(mx - mn, 1e-6)
        rn = len(_TERRAIN_RAMP)

        for x in range(w):
            for y in range(h):
                if self.world.obstacles[x][y]:
                    typ = self._obstacle_type_map.get((x, y), "rock")
                    px[x][y] = (
                        _COL_OBSTACLE_WATER if typ == "water"
                        else _COL_OBSTACLE_ROCK
                    )
                else:
                    nv = (elev[x][y] - mn) / span
                    idx = nv * (rn - 1)
                    i = int(idx)
                    frac = idx - i
                    if i >= rn - 1:
                        px[x][y] = _TERRAIN_RAMP[-1]
                    else:
                        c0, c1 = _TERRAIN_RAMP[i], _TERRAIN_RAMP[i + 1]
                        rgb = tuple(
                            int(c0[c] + frac * (c1[c] - c0[c]))
                            for c in range(3)
                        )
                        px[x][y] = rgb
        del px

        # nest
        nx, ny = int(self.world.nest_x), int(self.world.nest_y)
        nr = self.config.get("nest_start_size", 3)
        self._draw_circle(surf, nx, ny, max(2, nr), _COL_NEST)
        self._draw_circle(surf, nx, ny, max(3, nr + 1), _COL_NEST_GLOW, 1)
        return surf

    # ------------------------------------------------------------------
    #  Drawing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _draw_circle(surface, cx, cy, radius, color, width=0):
        """Anti-aliased circle if gfxdraw available, otherwise plain."""
        import pygame
        try:
            import pygame.gfxdraw
            if width <= 0:
                pygame.gfxdraw.filled_circle(surface, cx, cy, radius, color)
                pygame.gfxdraw.aacircle(surface, cx, cy, radius, color)
            else:
                pygame.gfxdraw.aacircle(surface, cx, cy, radius, color)
        except ImportError:
            pygame.draw.circle(surface, color, (cx, cy), radius, width)

    def _render_text(self, screen, text, x, y, font, color):
        """Render a string with either freetype or SysFont."""
        if self._is_freetype:
            font.render_to(screen, (x, y), text, color)
        else:
            surf = font.render(text, True, color)
            screen.blit(surf, (x, y))

    def _text_width(self, text, font):
        if self._is_freetype:
            return font.get_rect(text).width
        return font.size(text)[0]

    # ------------------------------------------------------------------
    #  Per-frame rendering
    # ------------------------------------------------------------------

    def _draw_pheromone_layer(self, screen, grid, scale, ox, oy, color):
        """Render a single pheromone grid via numpy → surfarray (one blit)."""
        import pygame

        w = len(grid)
        hh = len(grid[0]) if grid else 0
        if w == 0 or hh == 0:
            return

        cr, cg, cb = color

        # --- fast path: numpy ------------------------------------------------
        try:
            import numpy as np
            import pygame.surfarray

            arr = np.array(grid, dtype=np.float32)
            peak = arr.max()
            if peak < 0.01:
                return

            alpha = np.clip(arr / peak * 180, 0, 180).astype(np.uint8)
            alpha[arr < 0.01] = 0

            rgba = np.zeros((w, hh, 4), dtype=np.uint8)
            rgba[:, :, 0] = cr
            rgba[:, :, 1] = cg
            rgba[:, :, 2] = cb
            rgba[:, :, 3] = alpha

            surf = pygame.surfarray.make_surface(rgba)
        except ImportError:
            # --- fallback: per-cell surfaces (original logic) ----------------
            surf = pygame.Surface((w, hh), pygame.SRCALPHA)
            surf.fill((0, 0, 0, 0))
            for x in range(w):
                for y in range(hh):
                    v = grid[x][y]
                    if v > 0.01:
                        alpha = min(180, max(1, int(v * 180)))
                        px_surf = pygame.Surface((1, 1), pygame.SRCALPHA)
                        px_surf.fill((cr, cg, cb, alpha))
                        surf.blit(px_surf, (int(x * scale), int(y * scale)))

        # scale & blit
        dw, dh = int(w * scale), int(hh * scale)
        if dw > 0 and dh > 0:
            scaled = pygame.transform.scale(surf, (dw, dh))
            screen.blit(scaled, (ox, oy))

    def _draw_food(self, screen, scale, ox, oy):
        """Draw food patches directly at current scale."""
        for fx, fy, amount in self.world.food_sources:
            typ = self._food_type_map.get((int(fx), int(fy)), "bush")
            col = _FOOD_COLOURS.get(typ, _COL_FOOD_BUSH)
            sx = ox + int(fx * scale)
            sy = oy + int(fy * scale)
            radius = max(2, int((2 + amount / 30.0 * 6) * min(scale, 5)))
            bright = (
                min(255, col[0] + 40),
                min(255, col[1] + 40),
                min(255, col[2] + 40),
            )

            import pygame
            try:
                import pygame.gfxdraw
                pygame.gfxdraw.filled_circle(screen, sx, sy, radius, col)
                pygame.gfxdraw.aacircle(screen, sx, sy, radius, bright)
            except ImportError:
                pygame.draw.circle(screen, col, (sx, sy), radius)

    def _draw_agents(self, screen, scale, ox, oy):
        """Draw alive agents with role colours and direction indicators."""
        import pygame
        for agent in self.world.ants:
            if not agent.alive:
                continue

            ax = ox + int(agent.x * scale)
            ay = oy + int(agent.y * scale)
            r = max(2, int(3 * scale))

            colours = _ROLE_COLOURS.get(agent.role)
            if colours is None:
                body, accent = (200, 200, 200), (255, 255, 255)
            else:
                body, accent = colours

            self._draw_circle(screen, ax, ay, r, body)

            # direction line
            if agent.speed > 0.1:
                dx = math.cos(agent.direction) * r * 1.8
                dy = math.sin(agent.direction) * r * 1.8
                lw = max(1, int(1.5 * scale))
                pygame.draw.line(
                    screen, accent, (ax, ay), (ax + dx, ay + dy), lw,
                )

            # food-carrying ring
            if agent.food_carrying > 0:
                ring_r = r + max(1, int(1 * scale))
                self._draw_circle(
                    screen, ax, ay, ring_r, _COL_FOOD_BUSH, 1,
                )

    def _draw_hud(self, screen, step, world):
        """Top-left stats bar with semi-transparent background."""
        import pygame

        alive = sum(1 for a in world.ants if a.alive)
        total = len(world.ants)
        food_piles = len(world.food_sources)
        roles: dict[str, int] = {}
        for a in world.ants:
            if a.alive:
                roles[a.role] = roles.get(a.role, 0) + 1
        role_str = "  ".join(f"{k}: {v}" for k, v in sorted(roles.items()))

        line1 = (
            f"Step: {step}  |  FPS: {self._current_fps}  |  "
            f"Ants: {alive}/{total}  |  Food: {food_piles}  |  "
            f"Store: {world.colony_food_store:.0f}"
        )

        # panel sizing
        pad = 14
        text_w = max(self._text_width(line1, self._font),
                     self._text_width(role_str, self._font_small))
        panel_w = text_w + pad * 2
        line_h = 17
        panel_h = line_h * 2 + pad * 2

        panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        panel.fill(_COL_PANEL_BG)
        pygame.draw.rect(panel, _COL_PANEL_BORDER, panel.get_rect(), 1)
        screen.blit(panel, (10, 10))

        self._render_text(screen, line1, 10 + pad, 10 + pad,
                          self._font, _COL_HUD)
        self._render_text(screen, role_str, 10 + pad, 10 + pad + line_h,
                          self._font_small, _COL_HUD_DIM)

        # pausing indicator
        if self._paused:
            pw = 80
            ph = 28
            pause_surf = pygame.Surface((pw, ph), pygame.SRCALPHA)
            pause_surf.fill((0, 0, 0, 160))
            pygame.draw.rect(pause_surf, _COL_PANEL_BORDER,
                             pause_surf.get_rect(), 1)
            screen.blit(pause_surf, (10, 10 + panel_h + 4))
            self._render_text(
                screen, "  PAUSED", 10, 10 + panel_h + 4,
                self._font_large, (255, 255, 255),
            )

    def _build_legend_surface(self):
        """Pre-construct the legend panel (fixed pixel size)."""
        import pygame

        panel_w = 170
        line_h = 16
        title_h = 19
        gap = 6
        pad = 10

        # compute total height
        total_h = pad * 2
        for _, kind, _ in _LEGEND_ITEMS:
            if kind == "spacer":
                total_h += gap
            elif kind == "title":
                total_h += title_h + 2
            else:
                total_h += line_h + 1
        total_h = max(total_h, 260)

        surf = pygame.Surface((panel_w, total_h), pygame.SRCALPHA)
        surf.fill(_COL_PANEL_BG)
        pygame.draw.rect(surf, _COL_PANEL_BORDER, surf.get_rect(), 1)

        y = pad

        for label, kind, data in _LEGEND_ITEMS:
            if kind == "spacer":
                y += gap
                continue

            if kind == "title":
                self._render_text(
                    surf, " " + label, pad, y,
                    self._font_large, _COL_PANEL_TITLE,
                )
                y += title_h + 2
                continue

            if kind == "dot":
                # small filled circle indicator
                ind = pygame.Surface((10, 10), pygame.SRCALPHA)
                self._draw_circle(ind, 5, 5, 4, data)
                surf.blit(ind, (pad + 2, y + 1))
            elif kind == "ring":
                ind = pygame.Surface((10, 10), pygame.SRCALPHA)
                self._draw_circle(ind, 5, 5, 4, data, 1)
                surf.blit(ind, (pad + 2, y + 1))
            elif kind == "bar":
                ind = pygame.Surface((28, 8), pygame.SRCALPHA)
                ind.fill((*data, 140))
                surf.blit(ind, (pad + 2, y + 4))
            elif kind == "rect":
                ind = pygame.Surface((10, 10), pygame.SRCALPHA)
                pygame.draw.rect(ind, data, (0, 0, 10, 10))
                surf.blit(ind, (pad + 2, y + 1))

            text_col = _COL_HUD if kind != "ring" else _COL_HUD_DIM
            self._render_text(
                surf, label, pad + 16, y, self._font, text_col,
            )
            y += line_h + 1

        self._legend_surface = surf

    def _draw_legend(self, screen):
        """Blit the pre-built legend panel at bottom-right."""
        if self._legend_surface is None:
            return
        lx = self._current_w - self._legend_surface.get_width() - 12
        ly = self._current_h - self._legend_surface.get_height() - 12
        screen.blit(self._legend_surface, (lx, ly))

    def _track_fps(self):
        now = time.time()
        self._fps_samples.append(now)
        cutoff = now - 1.0
        self._fps_samples = [t for t in self._fps_samples if t > cutoff]
        self._current_fps = len(self._fps_samples)

    # ------------------------------------------------------------------
    #  Public interface
    # ------------------------------------------------------------------

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
                    self._paused = not self._paused
            elif event.type == pygame.VIDEORESIZE:
                self._current_w = event.w
                self._current_h = event.h
                self._screen = pygame.display.set_mode(
                    (event.w, event.h), pygame.RESIZABLE,
                )
        return self._running

    def render(self, world, step: int):
        if self._screen is None:
            self._init_pygame()

        import pygame

        screen = self._screen
        s = min(
            self._current_w / max(world.width, 1),
            self._current_h / max(world.height, 1),
        )
        ox = (self._current_w - int(world.width * s)) // 2
        oy = (self._current_h - int(world.height * s)) // 2

        # 1. background (pre-rendered terrain + obstacles + nest, scaled)
        if self._bg_raw is not None:
            bw = int(world.width * s)
            bh = int(world.height * s)
            if bw > 0 and bh > 0:
                screen.blit(
                    pygame.transform.scale(self._bg_raw, (bw, bh)),
                    (ox, oy),
                )

        # 2. pheromone layers (smooth numpy surfarray surfaces)
        self._draw_pheromone_layer(
            screen, world.food_pheromone, s, ox, oy, _COL_PHEROMONE_FOOD,
        )
        self._draw_pheromone_layer(
            screen, world.home_pheromone, s, ox, oy, _COL_PHEROMONE_HOME,
        )

        # 3. food patches (direct draw, type-aware colours)
        self._draw_food(screen, s, ox, oy)

        # 4. agents (gfxdraw circles with direction lines)
        self._draw_agents(screen, s, ox, oy)

        # 5. live pausing state propagation
        try:
            sim = getattr(self, "_sim_ref", None)
            if sim is not None:
                sim._paused = self._paused
        except Exception:
            pass

        # 6. HUD (top-left)
        self._draw_hud(screen, step, world)

        # 7. legend panel (bottom-right)
        self._draw_legend(screen)

        pygame.display.flip()
        self._clock.tick(self.fps)
        self._track_fps()

    def close(self):
        import pygame
        self._running = False
        try:
            pygame.display.quit()
            pygame.quit()
        except Exception:
            pass


# =============================================================================
#  Headless renderer (unchanged)
# =============================================================================

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
        pass
