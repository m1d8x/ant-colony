#!/usr/bin/env python3
"""Verify all renderer modules import and initialise correctly."""
import sys, os
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

sys.path.insert(0, "src")

import pygame
pygame.display.set_mode((1, 1))  # headless display
pygame.font.init()

W, H = 200, 200

# 1. Utils
from ant_colony.renderers.utils import make_surf, w2p, PX_PER_CELL
assert PX_PER_CELL == 10
assert w2p(5) == 50
print("1. Utils OK")

# 2. Sprites
from ant_colony.renderers.sprites import ant_surface
for role in ("forager", "builder", "soldier", "queen"):
    s = ant_surface(role)
    assert s.get_width() > 0
print("2. Sprites OK (4 roles)")

# 3. HUD
from ant_colony.renderers.hud import HUD
hud = HUD(window_width=640)
hsurf = hud.render(colony_food=120, population=300, ants_alive=280, tick=100)
assert hsurf.get_width() == 640
print("3. HUD OK")

# 4. Pheromone
from ant_colony.renderers.pheromone import PheromoneLayer
pl = PheromoneLayer(100, 100)
food_grid = [[0.0] * 100 for _ in range(100)]
home_grid = [[0.0] * 100 for _ in range(100)]
food_grid[30][40] = 80.0
home_grid[50][60] = 60.0
pl.update(food_grid, home_grid, 0, 0)
assert pl.surface.get_width() > 0
print("4. Pheromone OK")

# 5. Layers
from ant_colony.pysimengine import World, Agent
from ant_colony.renderers.layers import (
    render_ground, render_obstacles, render_nest, render_food, render_ants,
)

world = World(width=50, height=50)
world.nest_x = 25
world.nest_y = 25
world.obstacles[10][10] = True
world.food_sources.append((40, 40, 50))
for i in range(20):
    a = Agent(x=25 + i*0.5, y=25 + i*0.3, direction=0.5, role="forager")
    world.ants.append(a)

s = make_surf(W, H, alpha=False)
render_ground(s, world, 0, 0)
render_obstacles(s, world, 0, 0)
render_nest(s, world, 0, 0)
render_food(s, world, 0, 0)
render_ants(s, world, 0, 0)
print("5. Layers render OK")

# 6. Full Renderer
from ant_colony.renderers import Renderer
r = Renderer(width=W, height=H)
r.setup()
r.set_world(world)
r._running = False
print("6. Renderer OK")

print("\n  ALL CHECKS PASSED ✓")
pygame.quit()
