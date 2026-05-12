"""Final integration check — run with dummy driver."""
import os, sys
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
sys.path.insert(0, "src")

import pygame
pygame.display.set_mode((1, 1))

from ant_colony.pysimengine import World, Agent
from ant_colony.renderers import Renderer

world = World(width=100, height=100)
world.nest_x = 50
world.nest_y = 50
for i in range(10):
    world.ants.append(Agent(x=50+i, y=50, direction=0.0, role="forager"))

r = Renderer(width=200, height=200)
r.setup()
r.set_world(world)

for _ in range(3):
    r.begin_frame()
    world.tick()
    r.render(world)
    r.end_frame()

r.cleanup()
print("Integration OK: renderer + world ticked for 3 frames")
