"""Tests for ant colony agent types and state machines.

Verifies ForagerAgent, BuilderAgent, SoldierAgent, QueenAgent
state transitions, factory, and integration with World.tick().
"""

import math
import sys

sys.path.insert(0, "/root/ant-colony-sim/src")

from ant_colony.pysimengine import Agent, World
from ant_colony.agents import (
    ForagerAgent,
    BuilderAgent,
    SoldierAgent,
    QueenAgent,
    create_agent,
    ANT_COLORS,
)


# ── Creation tests ───────────────────────────────────────────────────────

def test_forager_creation():
    f = ForagerAgent(x=10.0, y=10.0)
    assert f.role == "forager"
    assert f.state == "SEARCHING"
    assert f.alive is True


def test_builder_creation():
    b = BuilderAgent(x=10.0, y=10.0)
    assert b.role == "builder"
    assert b.state == "IDLE"


def test_soldier_creation():
    s = SoldierAgent(x=10.0, y=10.0)
    assert s.role == "soldier"
    assert s.state == "PATROLLING"


def test_queen_creation():
    q = QueenAgent(x=10.0, y=10.0)
    assert q.role == "queen"
    assert q.state == "SPAWNING"
    assert q.speed == 0.0  # stationary


def test_create_agent_factory():
    f = create_agent("forager", x=5.0, y=5.0)
    assert isinstance(f, ForagerAgent)
    assert f.role == "forager"

    b = create_agent("builder", x=5.0, y=5.0)
    assert isinstance(b, BuilderAgent)

    s = create_agent("soldier", x=5.0, y=5.0)
    assert isinstance(s, SoldierAgent)

    q = create_agent("queen", x=5.0, y=5.0)
    assert isinstance(q, QueenAgent)

    try:
        create_agent("unknown_role")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_ant_colors():
    assert ANT_COLORS["forager"] == ((140, 90, 40), (200, 140, 60))
    assert ANT_COLORS["builder"] == ((80, 55, 30), (120, 90, 50))
    assert ANT_COLORS["soldier"] == ((140, 30, 30), (190, 60, 60))
    assert ANT_COLORS["queen"] == ((120, 70, 160), (160, 110, 200))


# ── Forager state machine ────────────────────────────────────────────────

def test_forager_initial_state():
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    f = ForagerAgent(x=25.0, y=25.0)
    assert f.state == "SEARCHING"


def test_forager_finds_food():
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    f = ForagerAgent(x=26.0, y=26.0)
    world.food_sources.append((26.0, 26.0, 50.0))
    f.update_state(world)
    assert f.state == "FOUND_FOOD", f"expected FOUND_FOOD, got {f.state}"


def test_forager_picks_up_food():
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    f = ForagerAgent(x=26.0, y=26.0)
    world.food_sources.append((26.0, 26.0, 50.0))
    # Two ticks: FOUND_FOOD → picks up → CARRYING
    f.update_state(world)  # SEARCHING → FOUND_FOOD
    assert f.state == "FOUND_FOOD"
    f.update_state(world)  # FOUND_FOOD → CARRYING
    assert f.state == "CARRYING", f"expected CARRYING, got {f.state}"
    assert f.food_carrying > 0, f"should carry food, got {f.food_carrying}"


def test_forager_deposits_food_at_nest():
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    f = ForagerAgent(x=26.0, y=26.0)
    world.food_sources.append((26.0, 26.0, 50.0))
    world.ants.append(f)
    # Find food
    f.update_state(world)  # SEARCHING → FOUND_FOOD
    f.update_state(world)  # FOUND_FOOD → CARRYING (picks up)
    prev_store = world.colony_food_store
    food_carried = f.food_carrying
    assert food_carried > 0
    # Move to nest
    f.x = 25.0
    f.y = 25.0
    f.update_state(world)  # CARRYING → RETURNING (deposits)
    assert f.food_carrying == 0.0, "food should be deposited"
    assert world.colony_food_store > prev_store, "colony store should increase"


# ── Builder state machine ────────────────────────────────────────────────

def test_builder_gathering_when_threshold():
    """Builder transitions IDLE→GATHERING when food store is sufficient."""
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    world.colony_food_store = 100.0  # Above GATHER_THRESHOLD (80)
    b = BuilderAgent(x=25.0, y=25.0)
    b.update_state(world)  # IDLE → GATHERING
    assert b.state == "GATHERING", f"expected GATHERING, got {b.state}"


def test_builder_stays_idle_when_low_food():
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    world.colony_food_store = 30.0  # Below GATHER_THRESHOLD
    b = BuilderAgent(x=25.0, y=25.0)
    b.update_state(world)
    assert b.state == "IDLE", f"expected IDLE, got {b.state}"


def test_builder_transitions_to_building():
    """Builder should go GATHERING→BUILDING when at nest."""
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    world.colony_food_store = 100.0
    b = BuilderAgent(x=25.0, y=25.0)
    b.update_state(world)  # IDLE → GATHERING
    assert b.state == "GATHERING"
    b.update_state(world)  # GATHERING → BUILDING (at nest, picks up food)
    assert b.state == "BUILDING", f"expected BUILDING, got {b.state}"
    assert b.food_carrying > 0, "should have collected food"


# ── Soldier state machine ────────────────────────────────────────────────

def test_soldier_patrolling():
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    s = SoldierAgent(x=25.0, y=25.0)
    assert s.state == "PATROLLING"
    s.update_state(world)
    assert s.state == "PATROLLING", "should stay PATROLLING with no threat"


def test_soldier_combat_transition():
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    s = SoldierAgent(x=25.0, y=25.0)
    world.threat_level = 0.5  # Above COMBAT_THREAT_THRESHOLD (0.3)
    s.update_state(world)
    assert s.state == "COMBAT", f"expected COMBAT, got {s.state}"


def test_soldier_guarding_after_combat():
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    s = SoldierAgent(x=25.0, y=25.0)
    world.threat_level = 0.5
    s.update_state(world)  # → COMBAT
    assert s.state == "COMBAT"
    # Reduce threat and run several ticks to transition to GUARDING
    world.threat_level = 0.0
    for _ in range(10):
        s.update_state(world)
    assert s.state == "GUARDING", f"expected GUARDING, got {s.state}"


# ── Queen state machine ──────────────────────────────────────────────────

def test_queen_snaps_to_nest():
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    q = QueenAgent(x=10.0, y=10.0)
    q.update_state(world)
    assert q.x == 25.0, "queen should snap to nest x"
    assert q.y == 25.0, "queen should snap to nest y"


def test_queen_spawn_to_idle():
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    q = QueenAgent(x=25.0, y=25.0)
    assert q.state == "SPAWNING"
    # Run enough ticks to transition to IDLE
    for _ in range(10):
        q.update_state(world)
    assert q.state == "IDLE", f"expected IDLE, got {q.state}"


# ── Integration tests ────────────────────────────────────────────────────

def test_world_tick_with_all_agent_types():
    """World.tick() should handle all 4 agent types without error."""
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    world.colony_food_store = 100.0
    world.food_sources.append((40.0, 40.0, 50.0))

    f = ForagerAgent(x=28.0, y=28.0)
    b = BuilderAgent(x=27.0, y=27.0)
    s = SoldierAgent(x=29.0, y=29.0)
    q = QueenAgent(x=25.0, y=25.0)

    for agent in [f, b, s, q]:
        world.ants.append(agent)

    for _ in range(20):
        world.tick()

    assert world.tick_number == 20
    assert all(a.alive for a in world.ants), "all should survive 20 ticks"


def test_forager_state_machine_full_cycle():
    """Forager goes SEARCHING → FOUND_FOOD → CARRYING → RETURNING."""
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    f = ForagerAgent(x=26.0, y=26.0)
    world.food_sources.append((26.0, 26.0, 50.0))

    assert f.state == "SEARCHING"
    f.update_state(world)
    assert f.state == "FOUND_FOOD"
    f.update_state(world)
    assert f.state == "CARRYING"
    # Move to nest to deposit
    f.x, f.y = 25.0, 25.0
    f.update_state(world)
    assert f.state == "RETURNING", f"expected RETURNING, got {f.state}"


def test_builder_state_machine_full_cycle():
    """Builder goes IDLE → GATHERING → BUILDING → IDLE."""
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    world.colony_food_store = 100.0
    b = BuilderAgent(x=25.0, y=25.0)

    assert b.state == "IDLE"
    b.update_state(world)
    assert b.state == "GATHERING"
    b.update_state(world)
    assert b.state == "BUILDING", f"expected BUILDING, got {b.state}"
    # At build target, deposit
    b.x, b.y = b._build_target
    b.update_state(world)
    assert b.state == "IDLE", f"expected IDLE, got {b.state}"
    assert b.food_carrying == 0.0


def test_soldier_state_machine_full_cycle():
    """Soldier goes PATROLLING → COMBAT → GUARDING → PATROLLING."""
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    s = SoldierAgent(x=25.0, y=25.0)

    assert s.state == "PATROLLING"
    world.threat_level = 0.5
    s.update_state(world)
    assert s.state == "COMBAT"
    world.threat_level = 0.0
    for _ in range(20):
        s.update_state(world)
    assert s.state == "GUARDING", f"expected GUARDING, got {s.state}"
    # Run enough guarding ticks to return to PATROLLING
    for _ in range(60):
        s.update_state(world)
    assert s.state == "PATROLLING", f"expected PATROLLING, got {s.state}"


def test_queen_snaps_to_nest_every_tick():
    """Queen position should always equal nest position."""
    world = World(width=50, height=50)
    world.nest_x = 25.0
    world.nest_y = 25.0
    q = QueenAgent(x=10.0, y=10.0)
    world.ants.append(q)
    for _ in range(10):
        world.tick()
        assert q.x == world.nest_x
        assert q.y == world.nest_y


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
