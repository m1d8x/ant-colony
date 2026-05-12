#!/usr/bin/env python3
"""Ant Colony Simulation — Polished Pygame Renderer

Usage:
    python run.py              # Normal mode
    python run.py --fullscreen # Fullscreen
"""
from __future__ import annotations

import math
import random
import sys

import pygame

from ant_colony.pysimengine import World, Agent
from ant_colony.renderers import Renderer


def build_demo_world() -> World:
    """Create a 200×200 world with ants, obstacles, food, and nest."""
    world = World(width=200, height=200)

    # Nest at centre
    world.nest_x = 100.0
    world.nest_y = 100.0

    rng = random.Random(42)

    # ── Obstacles ──────────────────────────────────────────────────
    # Scatter rock clusters
    for _ in range(30):
        cx = rng.randint(10, world.width - 10)
        cy = rng.randint(10, world.height - 10)
        if abs(cx - world.nest_x) < 15 and abs(cy - world.nest_y) < 15:
            continue
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if rng.random() < 0.4:  # irregular blob shapes
                    continue
                wx = cx + dx
                wy = cy + dy
                if 0 <= wx < world.width and 0 <= wy < world.height:
                    world.obstacles[wx][wy] = True

    # ── Food sources ───────────────────────────────────────────────
    for _ in range(20):
        while True:
            fx = rng.uniform(10, world.width - 10)
            fy = rng.uniform(10, world.height - 10)
            if math.hypot(fx - world.nest_x, fy - world.nest_y) > 25:
                break
        world.food_sources.append((fx, fy, rng.uniform(30, 80)))

    # ── Ants ───────────────────────────────────────────────────────
    # Foragers (workers)
    for i in range(300):
        angle = rng.random() * math.pi * 2
        dist = rng.uniform(3, 10)
        ant = Agent(
            x=world.nest_x + math.cos(angle) * dist,
            y=world.nest_y + math.sin(angle) * dist,
            direction=angle,
            speed=rng.uniform(1.5, 2.5),
            role="forager",
            food_carrying=0.0,
            max_age=800,
        )
        ant.food_capacity = 10.0
        world.ants.append(ant)

    # Soldiers
    for i in range(100):
        angle = rng.random() * math.pi * 2
        dist = rng.uniform(5, 15)
        ant = Agent(
            x=world.nest_x + math.cos(angle) * dist,
            y=world.nest_y + math.sin(angle) * dist,
            direction=angle,
            speed=rng.uniform(1.2, 1.8),
            role="soldier",
            max_age=1200,
        )
        world.ants.append(ant)

    # Builders
    for i in range(50):
        angle = rng.random() * math.pi * 2
        dist = rng.uniform(3, 8)
        ant = Agent(
            x=world.nest_x + math.cos(angle) * dist,
            y=world.nest_y + math.sin(angle) * dist,
            direction=angle,
            speed=rng.uniform(1.0, 1.5),
            role="builder",
            max_age=1000,
        )
        world.ants.append(ant)

    # Queen (stationary, near nest centre)
    queen = Agent(
        x=world.nest_x,
        y=world.nest_y + 2,
        direction=0.0,
        speed=0.0,
        role="queen",
        max_age=5000,
    )
    world.ants.append(queen)

    return world


def main() -> None:
    fullscreen = "--fullscreen" in sys.argv

    world = build_demo_world()
    print(f"World: {world.width}×{world.height}")
    print(f"Nest: ({world.nest_x:.0f}, {world.nest_y:.0f})")
    print(f"Ants: {len(world.ants)}")
    print(f"Food sources: {len(world.food_sources)}")
    print(f"Obstacle cells: "
          f"{sum(1 for row in world.obstacles for c in row if c)}")

    renderer = Renderer(fullscreen=fullscreen)
    renderer.setup()
    renderer.set_world(world)

    try:
        while renderer.running:
            renderer.begin_frame()
            if not renderer.running:
                break

            # Advance simulation
            if not renderer.hud.paused:
                sim_speed = renderer.hud.speed
                for _ in range(sim_speed):
                    world.tick()
                    # Foragers collect food when on food source
                    for ant in world.ants:
                        if not ant.alive or ant.role == "queen":
                            continue
                        if ant.role == "forager" and ant.food_carrying < ant.food_capacity:
                            for i, (fx, fy, amount) in enumerate(world.food_sources):
                                dist = math.hypot(ant.x - fx, ant.y - fy)
                                if dist < 2.0 and amount > 0:
                                    take = min(ant.food_capacity - ant.food_carrying,
                                               amount, 2.0)
                                    ant.food_carrying += take
                                    x, y, amt = world.food_sources[i]
                                    world.food_sources[i] = (x, y, amt - take)
                                    break
                        # Deposit food when back at nest
                        dist_nest = math.hypot(ant.x - world.nest_x,
                                                ant.y - world.nest_y)
                        if dist_nest < 5.0 and ant.food_carrying > 0:
                            world.colony_food_store += ant.food_carrying
                            ant.food_carrying = 0.0
                            ant.direction += math.pi

            # Render
            renderer.render(world)
            renderer.end_frame()

    except KeyboardInterrupt:
        pass
    finally:
        renderer.cleanup()


if __name__ == "__main__":
    main()
