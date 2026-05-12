"""Tests for ant colony behaviors and colony manager."""

import math
import sys

sys.path.insert(0, "/root/ant-colony-sim/src")

from ant_colony.pysimengine import Agent, World, Behavior
from ant_colony.behaviors import (
    AvoidObstacles,
    DepositTrail,
    FollowGradient,
    WanderWithPersistence,
)
from ant_colony.behaviors.colony_manager import ColonyManager


# ── Agent tests ──────────────────────────────────────────────────────────

def test_agent_defaults():
    ant = Agent(x=10.0, y=20.0)
    assert ant.x == 10.0
    assert ant.y == 20.0
    assert ant.role == "forager"
    assert ant.alive is True
    assert ant.age == 0


def test_agent_sensors():
    """Sensor positions should be offset from heading."""
    ant = Agent(x=0.0, y=0.0, direction=0.0)  # facing east
    lx, ly = ant.sensor_left()
    cx, cy = ant.sensor_center()
    rx, ry = ant.sensor_right()

    # Centre is straight ahead (east)
    assert cx > 0, "centre sensor should be ahead"
    assert abs(cy) < 0.01, "centre sensor should be horizontal"

    # Left sensor at -30° offset: sin(-30°) = -0.5, so ly < cy
    # Right sensor at +30° offset: sin(30°) = 0.5, so ry > cy
    assert ly < cy, "left sensor should be above centre (lower y)"
    assert ry > cy, "right sensor should be below centre (higher y)"

    # All sensors at same distance
    for (sx, sy) in [(lx, ly), (cx, cy), (rx, ry)]:
        dist = math.hypot(sx, sy)
        assert abs(dist - ant.sensor_distance) < 0.01, f"sensor distance {dist} != {ant.sensor_distance}"


def test_agent_nearby_point():
    ant = Agent(x=0.0, y=0.0, direction=0.0)
    nx, ny = ant.nearby_point(5.0, 0.0)
    assert abs(nx - 5.0) < 0.01
    assert abs(ny) < 0.01

    nx, ny = ant.nearby_point(5.0, math.pi / 2)
    assert abs(nx) < 0.01
    assert abs(ny - 5.0) < 0.01


# ── World tests ──────────────────────────────────────────────────────────

def test_world_defaults():
    w = World(width=50, height=50)
    assert w.width == 50
    assert w.height == 50
    assert len(w.food_pheromone) == 50
    assert len(w.food_pheromone[0]) == 50
    assert w.colony_food_store == 100.0
    assert w.colony_manager is None


def test_world_pheromone_add_and_read():
    w = World(width=10, height=10)
    w.add_food_pheromone(5, 5, 10.0)
    assert w.food_pheromone[5][5] == 10.0
    val = w.read_food_pheromone(5.0, 5.0)
    assert abs(val - 10.0) < 0.01


def test_world_pheromone_decay():
    w = World(width=10, height=10)
    w.add_food_pheromone(5, 5, 100.0)
    w.step_pheromones()
    assert w.food_pheromone[5][5] < 100.0
    assert w.food_pheromone[5][5] < 100.0 * 0.97


def test_world_obstacle_detection():
    w = World(width=10, height=10)
    w.obstacles[3][3] = True
    assert w.is_blocked(3.0, 3.0) is True
    assert w.is_blocked(2.0, 3.0) is False
    assert w.is_blocked(-1.0, 5.0) is True
    assert w.is_blocked(5.0, 10.0) is True


# ── Behavior tests ───────────────────────────────────────────────────────

def test_behavior_base():
    """Base class should raise NotImplementedError."""
    b = Behavior()
    try:
        b.update(Agent(0, 0), World(10, 10))
        assert False, "should have raised"
    except NotImplementedError:
        pass


def test_follow_gradient_turns_toward_stronger():
    """Ant should turn toward the side with stronger pheromone."""
    world = World(width=50, height=50)
    ant = Agent(x=25.0, y=25.0, direction=0.0)
    world.ants.append(ant)

    target_x = int(ant.sensor_right()[0])
    target_y = int(ant.sensor_right()[1])
    world.add_food_pheromone(target_x, target_y, 50.0)

    old_dir = ant.direction
    FollowGradient().update(ant, world)
    assert ant.direction > old_dir, "ant should turn right toward pheromone"


def test_follow_gradient_center_straight():
    """When centre sensor is strongest, barely turn."""
    world = World(width=50, height=50)
    ant = Agent(x=25.0, y=25.0, direction=0.0)
    world.ants.append(ant)

    cx, cy = int(ant.sensor_center()[0]), int(ant.sensor_center()[1])
    world.add_food_pheromone(cx, cy, 50.0)

    old_dir = ant.direction
    FollowGradient().update(ant, world)
    assert abs(ant.direction - old_dir) < 0.2, "should stay mostly straight"


def test_deposit_trail_forager():
    """Forager deposits food pheromone proportional to food carried."""
    world = World(width=50, height=50)
    ant = Agent(x=25.0, y=25.0, role="forager", food_carrying=5.0)
    world.ants.append(ant)

    DepositTrail(base_deposit=1.0, food_multiplier=2.0).update(ant, world)
    assert world.food_pheromone[25][25] > 0.0
    assert abs(world.food_pheromone[25][25] - 11.0) < 0.01


def test_deposit_trail_soldier():
    """Soldier deposits home pheromone (not food pheromone)."""
    world = World(width=50, height=50)
    ant = Agent(x=25.0, y=25.0, role="soldier")
    world.ants.append(ant)

    DepositTrail().update(ant, world)
    assert world.home_pheromone[25][25] > 0.0, "soldier should deposit home pheromone"
    assert world.food_pheromone[25][25] == 0.0, "soldier should NOT deposit food pheromone"


def test_wander_persistence():
    """Ant with high persistence should mostly keep direction."""
    world = World(width=50, height=50)
    ant = Agent(x=25.0, y=25.0, direction=1.0)

    directions = []
    for _ in range(100):
        wander = WanderWithPersistence(persistence=0.95, max_turn=0.05)
        wander.update(ant, world)
        directions.append(ant.direction)

    avg_change = sum(abs(directions[i] - directions[i-1]) for i in range(1, len(directions))) / (len(directions) - 1)
    assert avg_change < 0.03, f"avg change {avg_change} too high for persistent wander"


def test_wander_mean_zero_drift():
    """Ant wander should have zero mean drift (unbiased random walk)."""
    world = World(width=50, height=50)
    ant = Agent(x=25.0, y=25.0, direction=0.0)

    directions = []
    for _ in range(100):
        WanderWithPersistence(persistence=0.5, max_turn=0.5).update(ant, world)
        directions.append(ant.direction)

    # Compute mean angular change (should be near zero with sufficient samples)
    changes = [(directions[i] - directions[i-1]) for i in range(1, len(directions))]
    # Normalize to [-pi, pi]
    changes = [(c + math.pi) % (2 * math.pi) - math.pi for c in changes]
    mean_change = sum(changes) / len(changes)
    # Mean drift should be small (well below one max_turn step)
    assert abs(mean_change) < 0.3, f"mean drift {mean_change:.3f} too large, wander is biased"


def test_avoid_obstacles_turns():
    """Ant should turn away from obstacles ahead."""
    world = World(width=20, height=20)
    ant = Agent(x=10.0, y=10.0, direction=0.0)

    # Place a wall right in front (2-5 cells ahead)
    for dx in range(2, 6):
        world.obstacles[10 + dx][10] = True

    old_dir = ant.direction
    AvoidObstacles(look_ahead=4.0).update(ant, world)
    assert ant.direction != old_dir, "ant should turn away from obstacle"


def test_avoid_obstacles_no_obstacle():
    """Without obstacles, direction should not change much."""
    world = World(width=50, height=50)
    ant = Agent(x=25.0, y=25.0, direction=0.0)

    old_dir = ant.direction
    AvoidObstacles(look_ahead=6.0).update(ant, world)
    assert ant.x > 25.0, "ant should move forward without obstacle"


# ── Colony Manager tests ─────────────────────────────────────────────────

def test_colony_spawns_ants():
    """Colony should spawn new ants when food threshold is met."""
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    world.colony_food_store = 100.0

    for _ in range(5):
        world.ants.append(Agent(x=25.0, y=25.0))

    cm = ColonyManager()
    cm.update(None, world)
    assert len(world.ants) > 5, f"Expected spawned ants, got {len(world.ants)}"


def test_colony_no_spawn_when_starving():
    """Colony should NOT spawn when food is below threshold."""
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    world.colony_food_store = 5.0

    for _ in range(3):
        world.ants.append(Agent(x=25.0, y=25.0))

    cm = ColonyManager()
    cm.update(None, world)
    assert len(world.ants) == 3, "should not spawn when starving"


def test_colony_role_reassignment():
    """Colony should shift to more foragers when starving."""
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    world.colony_food_store = 20.0  # below STARVING_THRESHOLD

    for _ in range(20):
        world.ants.append(Agent(x=25.0, y=25.0, role="builder"))

    cm = ColonyManager()
    for tick in range(15):
        world.tick_number = tick
        cm.update(None, world)

    foragers = sum(1 for a in world.ants if a.role == "forager" and a.alive)
    total_alive = sum(1 for a in world.ants if a.alive)
    assert foragers / total_alive > 0.5, f"only {foragers}/{total_alive} foragers when starving"
    print(f"  Role reassignment: {foragers}/{total_alive} foragers "
          f"({foragers/total_alive*100:.0f}%) — target ~80%")


# ── Integration tests ────────────────────────────────────────────────────

def test_world_tick_integration():
    """Full integration: world tick should run without error."""
    import random
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0

    for _ in range(10):
        angle = random.uniform(0, 2 * math.pi)
        world.ants.append(Agent(x=25.0, y=25.0, direction=angle))

    for tick in range(50):
        world.tick()

    assert world.tick_number == 50
    assert len(world.ants) > 0, "colony should survive 50 ticks"
    print(f"  Integration: {len(world.ants)} ants alive after 50 ticks, "
          f"food_store={world.colony_food_store:.1f}")


def test_colony_survives_500_ticks():
    """Colony should sustain itself over 500 ticks."""
    import random

    world = World(width=80, height=80)
    world.nest_x = 40.0
    world.nest_y = 40.0

    for _ in range(20):
        while True:
            fx = random.uniform(10, 70)
            fy = random.uniform(10, 70)
            if math.hypot(fx - 40, fy - 40) > 15:
                break
        world.food_sources.append((fx, fy, random.uniform(30, 80)))

    for _ in range(15):
        x = random.randint(20, 60)
        y = random.randint(20, 60)
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                if 0 <= x + dx < 80 and 0 <= y + dy < 80:
                    world.obstacles[x + dx][y + dy] = True

    for _ in range(10):
        angle = random.uniform(0, 2 * math.pi)
        world.ants.append(Agent(x=40.0, y=40.0, direction=angle))

    for tick in range(500):
        world.tick()

        # Forage food
        for ant in world.ants:
            if not ant.alive or ant.role != "forager":
                continue
            for i, (fx, fy, amount) in enumerate(world.food_sources):
                if math.hypot(ant.x - fx, ant.y - fy) < 3.0 and amount > 0 and ant.food_carrying < ant.food_capacity:
                    take = min(ant.food_capacity - ant.food_carrying, amount, 2.0)
                    ant.food_carrying += take
                    world.food_sources[i] = (fx, fy, amount - take)

            if math.hypot(ant.x - world.nest_x, ant.y - world.nest_y) < 5.0 and ant.food_carrying > 0:
                world.colony_food_store += ant.food_carrying
                ant.food_carrying = 0.0

    assert world.tick_number == 500
    alive = [a for a in world.ants if a.alive]
    assert len(alive) > 0, "colony died before 500 ticks"
    print(f"  Longevity: {len(alive)} ants alive after 500 ticks, "
          f"food={world.colony_food_store:.1f}, "
          f"roles={sum(1 for a in alive if a.role=='forager')}F "
          f"{sum(1 for a in alive if a.role=='soldier')}S "
          f"{sum(1 for a in alive if a.role=='builder')}B")


# ── Runner ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print(f"  \u2705 {name}")
                passed += 1
            except Exception as e:
                print(f"  \u274c {name}: {e}")
                failed += 1

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed, {passed+failed} total")
    if failed:
        sys.exit(1)
