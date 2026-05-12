#!/usr/bin/env python3
"""Ant Colony Simulation Demo — runs for N ticks showing colony self-regulation."""

import math
import random
import sys

sys.path.insert(0, "/root/ant-colony-sim/src")

from ant_colony.pysimengine import Agent, World


def build_world() -> World:
    world = World(width=100, height=100)
    world.nest_x = 50.0
    world.nest_y = 50.0

    for _ in range(20):
        x = random.randint(15, 85)
        y = random.randint(15, 85)
        if abs(x - world.nest_x) < 10 and abs(y - world.nest_y) < 10:
            continue
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                wx, wy = x + dx, y + dy
                if 0 <= wx < world.width and 0 <= wy < world.height:
                    world.obstacles[wx][wy] = True

    for _ in range(15):
        while True:
            x = random.uniform(10, 90)
            y = random.uniform(10, 90)
            if math.hypot(x - world.nest_x, y - world.nest_y) > 20:
                break
        world.food_sources.append((x, y, random.uniform(30, 80)))

    for _ in range(15):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(3.0, 8.0)
        ant = Agent(
            x=world.nest_x + math.cos(angle) * dist,
            y=world.nest_y + math.sin(angle) * dist,
            direction=angle,
            speed=random.uniform(1.5, 2.5),
            role="forager",
        )
        world.ants.append(ant)

    return world


def print_status(world: World, stats: dict) -> None:
    roles = stats["roles"]
    forager = roles.get("forager", 0)
    soldier = roles.get("soldier", 0)
    builder = roles.get("builder", 0)
    total = stats["ants"]

    bar_w = 20
    bar = ""
    if total > 0:
        f_w = max(1, round(forager / total * bar_w))
        s_w = max(1, round(soldier / total * bar_w))
        b_w = bar_w - f_w - s_w
        if b_w < 0:
            b_w = 0
        bar = f" [{'F'*f_w}{'S'*s_w}{'B'*b_w}]"

    print(
        f"Tick {stats['tick']:>5d}  |  "
        f"Pop: {total:>3d}  "
        f"(F:{forager} S:{soldier} B:{builder}){bar}  |  "
        f"Food: {stats['food_store']:>6.1f}  "
        f"Carried: {stats['food_carried']:>5.1f}  "
        f"Rate: {stats['food_income_rate']:>5.1f}/tick"
    )


def run_simulation(ticks: int = 500) -> None:
    world = build_world()

    print("=" * 100)
    print("  ANT COLONY SIMULATION  —  Self-Regulating Emergent Behaviors")
    print("=" * 100)
    print(f"  World: {world.width}x{world.height}, Nest: ({world.nest_x:.0f},{world.nest_y:.0f})")
    print(f"  Food sources: {len(world.food_sources)}, Initial ants: {len(world.ants)}")
    print(f"  {'Tick':>5}  {'Pop':>3}  {'Roles':>25}    {'Food':>6}  {'Rate':>6}")
    print("-" * 100)

    for tick in range(1, ticks + 1):
        world.tick()

        for ant in world.ants:
            if not ant.alive:
                continue
            if ant.role == "forager":
                for i, (fx, fy, amount) in enumerate(world.food_sources):
                    dist = math.hypot(ant.x - fx, ant.y - fy)
                    if dist < 3.0 and amount > 0 and ant.food_carrying < ant.food_capacity:
                        take = min(ant.food_capacity - ant.food_carrying, amount, 2.0)
                        ant.food_carrying += take
                        world.food_sources[i] = (fx, fy, amount - take)

                dist_to_nest = math.hypot(ant.x - world.nest_x, ant.y - world.nest_y)
                if dist_to_nest < 5.0 and ant.food_carrying > 0:
                    world.colony_food_store += ant.food_carrying
                    ant.food_carrying = 0.0

        if tick % 25 == 0 or tick == 1 or tick == ticks:
            stats = world.colony_manager.colony_stats(world) if world.colony_manager else {}
            if stats:
                print_status(world, stats)

    print("-" * 100)
    final_stats = world.colony_manager.colony_stats(world) if world.colony_manager else {}
    print(f"  Final: {final_stats.get('ants', 0)} ants, "
          f"food store: {final_stats.get('food_store', 0):.1f}, "
          f"{world.tick_number} ticks")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    run_simulation(n)
