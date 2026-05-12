"""HUD overlay — colony stats, pause button, speed control, pheromone toggle."""
from __future__ import annotations

import pygame

from ant_colony.renderers.utils import make_surf, round_rect


class HUD:
    """Heads-up display rendered on top of the game viewport."""

    HEIGHT = 48
    BG = (30, 28, 26)
    TEXT_COLOR = (220, 215, 200)
    ACCENT = (200, 160, 80)
    BTN_COLOR = (55, 50, 45)
    BTN_HOVER = (75, 68, 60)
    BTN_ACTIVE = (100, 85, 60)

    def __init__(self, window_width: int = 1280) -> None:
        self.win_w = window_width
        self.font = pygame.font.SysFont("monospace", 16, bold=False)
        self.font_bold = pygame.font.SysFont("monospace", 16, bold=True)

        # State
        self.paused = False
        self.speed = 1          # 1, 2, or 4
        self.show_pheromones = True

        self._surface = make_surf(self.win_w, self.HEIGHT, alpha=False)
        self._dirty = True
        self._btn_pause: pygame.Rect | None = None
        self._btn_speed: pygame.Rect | None = None
        self._btn_phero: pygame.Rect | None = None
        self._mx: int = 0
        self._my: int = 0

    @property
    def height(self) -> int:
        return self.HEIGHT

    def handle_click(self, x: int, y: int) -> str | None:
        """Return action string if a button was clicked."""
        if self._btn_pause and self._btn_pause.collidepoint(x, y):
            self.paused = not self.paused
            self._dirty = True
            return "pause"
        if self._btn_speed and self._btn_speed.collidepoint(x, y):
            speeds = [1, 2, 4]
            idx = speeds.index(self.speed)
            self.speed = speeds[(idx + 1) % 3]
            self._dirty = True
            return "speed"
        if self._btn_phero and self._btn_phero.collidepoint(x, y):
            self.show_pheromones = not self.show_pheromones
            self._dirty = True
            return "pheromone_toggle"
        return None

    def update_mouse(self, pos: tuple[int, int]) -> None:
        self._mx, self._my = pos

    def render(self, colony_food: float, population: int,
               ants_alive: int, tick: int, **kw) -> pygame.Surface:
        """Render the HUD surface, caching between state changes."""
        # Check if any state changed
        if (kw.get("paused", self.paused) != self.paused or
                kw.get("speed", self.speed) != self.speed or
                kw.get("show_phero", self.show_pheromones) != self.show_pheromones):
            self.paused = kw.get("paused", self.paused)
            self.speed = kw.get("speed", self.speed)
            self.show_pheromones = kw.get("show_phero", self.show_pheromones)
            self._dirty = True

        if not self._dirty:
            return self._surface

        surf = self._surface
        surf.fill(self.BG)
        mx, my = self._mx, self._my

        # ── Left: colony stats ─────────────────────────────────────
        stats = [
            (f"Pop: {ants_alive}/{population}", (100, 200, 100)),
            (f"Food: {colony_food:.0f}", (230, 200, 80)),
            (f"Tick: {tick}", (150, 150, 150)),
        ]
        sx = 12
        for text, color in stats:
            lbl = self.font.render(text, True, color)
            surf.blit(lbl, (sx, (self.HEIGHT - lbl.get_height()) // 2))
            sx += lbl.get_width() + 18

        # ── Right: buttons ─────────────────────────────────────────
        btn_w, btn_h = 80, self.HEIGHT - 10
        btn_y = 5

        # Pause button
        bx = self.win_w - btn_w - 8
        self._btn_pause = pygame.Rect(bx, btn_y, btn_w, btn_h)
        p_color = self.BTN_ACTIVE if self.paused else (
            self.BTN_HOVER if self._btn_pause.collidepoint(mx, my)
            else self.BTN_COLOR)
        round_rect(surf, self._btn_pause, p_color, radius=6)
        p_label = "▶ PLAY" if self.paused else "⏸ PAUSE"
        p_text = self.font_bold.render(
            p_label, True,
            self.ACCENT if self.paused else self.TEXT_COLOR)
        surf.blit(p_text,
                  (bx + (btn_w - p_text.get_width()) // 2,
                   btn_y + (btn_h - p_text.get_height()) // 2))

        # Speed button
        bx -= btn_w + 6
        self._btn_speed = pygame.Rect(bx, btn_y, btn_w, btn_h)
        s_color = (self.BTN_HOVER if self._btn_speed.collidepoint(mx, my)
                   else self.BTN_COLOR)
        round_rect(surf, self._btn_speed, s_color, radius=6)
        speed_label = f"{self.speed}x"
        speed_text = self.font_bold.render(speed_label, True, self.ACCENT)
        surf.blit(speed_text,
                  (bx + (btn_w - speed_text.get_width()) // 2,
                   btn_y + (btn_h - speed_text.get_height()) // 2))

        # Pheromone toggle
        bx -= btn_w + 6
        self._btn_phero = pygame.Rect(bx, btn_y, btn_w, btn_h)
        ph_color = (self.BTN_ACTIVE if self.show_pheromones else
                    (self.BTN_HOVER if self._btn_phero.collidepoint(mx, my)
                     else self.BTN_COLOR))
        round_rect(surf, self._btn_phero, ph_color, radius=6)
        ph_label = "PH ON" if self.show_pheromones else "PH OFF"
        ph_text = self.font_bold.render(
            ph_label, True,
            self.ACCENT if self.show_pheromones else (160, 80, 80))
        surf.blit(ph_text,
                  (bx + (btn_w - ph_text.get_width()) // 2,
                   btn_y + (btn_h - ph_text.get_height()) // 2))

        self._dirty = False
        return surf
