"""Tests for the world environment module — nest, obstacles, food, terrain."""

import sys
sys.path.insert(0, "/root/ant-colony-sim/src")

from ant_colony.world import Nest, ObstacleGrid, FoodPatch, FoodManager, FoodType, TerrainMap, Environment


# ── Nest ────────────────────────────────────────────────

def test_nest_default_square():
    n = Nest(cx=50, cy=50, start_radius=3)
    assert len(n) == 49  # 7x7
    assert (50, 50) in n
    assert (47, 47) in n
    assert (53, 53) in n
    assert (46, 50) not in n


def test_nest_contains():
    n = Nest(cx=100, cy=100, start_radius=2)
    assert n.contains(100, 100)
    assert n.contains_float(100.5, 100.3)
    assert not n.contains(95, 95)


def test_nest_border_tiles():
    n = Nest(cx=10, cy=10, start_radius=1)
    border = n.border_tiles()
    expected = {(8,8),(9,8),(10,8),(11,8),(12,8),(8,9),(12,9),(8,10),(12,10),(8,11),(12,11),(8,12),(9,12),(10,12),(11,12),(12,12)}
    assert border == expected


def test_nest_add_tile_adjacent():
    n = Nest(cx=10, cy=10, start_radius=1)
    assert n.add_tile_adjacent(12, 10)  # adjacent to nest tile (11, 10)
    assert (12, 10) in n
    assert not n.add_tile_adjacent(20, 20)


def test_nest_expansion_positions():
    n = Nest(cx=10, cy=10, start_radius=1)
    positions = n.expansion_positions()
    assert len(positions) > 0


def test_nest_as_mask():
    n = Nest(cx=5, cy=5, start_radius=1)
    mask = n.as_mask(20, 20)
    assert mask[5][5] is True
    assert mask[6][5] is True
    assert mask[0][0] is False

# ── Obstacles ───────────────────────────────────────────────

def test_obstacle_grid_defaults():
    grid = ObstacleGrid(width=100, height=100)
    assert grid.width == 100
    assert not grid.is_blocked(50.0, 50.0)
    assert grid.is_blocked(-1.0, 5.0)
    assert grid.is_blocked(100.0, 50.0)


def test_obstacle_generate_basic():
    grid = ObstacleGrid.generate(width=100, height=100, rock_count=10, water_count=2, seed=42, nest_zone=(50,50,5))
    assert grid.blocked_count() > 0
    for x in range(45, 56):
        for y in range(45, 56):
            assert not grid.is_blocked(float(x), float(y))


def test_obstacle_blocks_pheromone():
    grid = ObstacleGrid.generate(width=50, height=50, rock_count=5, water_count=1, seed=99)
    found = False
    for x in range(50):
        for y in range(50):
            if grid.is_blocked(float(x), float(y)):
                assert grid.blocks_pheromone(x, y)
                found = True
    assert found


def test_obstacle_reproducible():
    g1 = ObstacleGrid.generate(width=50, height=50, seed=123)
    g2 = ObstacleGrid.generate(width=50, height=50, seed=123)
    for x in range(50):
        for y in range(50):
            assert g1.is_blocked(float(x), float(y)) == g2.is_blocked(float(x), float(y))


def test_obstacle_diffusion_mask():
    grid = ObstacleGrid.generate(width=30, height=30, rock_count=5, water_count=1, seed=42)
    mask = grid.diffusion_mask()
    for x in range(30):
        for y in range(30):
            assert mask[x][y] == (not grid.obstacles[x][y])

# ── Food ────────────────────────────────────────────────

def test_food_patch_default():
    patch = FoodPatch(x=10, y=10)
    assert patch.food_type == FoodType.BUSH
    assert patch.max_amount == 30.0
    assert patch.available
    assert not patch.depleted


def test_food_patch_types():
    assert FoodPatch(x=0, y=0, food_type=FoodType.BUSH).max_amount == 30.0
    assert FoodPatch(x=0, y=0, food_type=FoodType.MUSHROOM).max_amount == 50.0
    assert FoodPatch(x=0, y=0, food_type=FoodType.CRYSTAL).max_amount == 100.0


def test_food_collect():
    patch = FoodPatch(x=5, y=5, max_amount=30.0)
    taken = patch.collect(10.0)
    assert taken == 10.0
    assert patch.current_amount == 20.0


def test_food_deplete():
    patch = FoodPatch(x=5, y=5, max_amount=10.0)
    patch.collect(10.0)
    assert patch.depleted
    assert patch.depletion_timer == patch.respawn_ticks


def test_food_respawn():
    patch = FoodPatch(x=5, y=5, max_amount=10.0, respawn_ticks=5)
    patch.collect(10.0)
    for _ in range(4):
        patch.tick()
        assert patch.depleted
    patch.tick()
    assert patch.available
    assert patch.current_amount == 10.0


def test_food_manager_generate():
    mgr = FoodManager.generate(width=100, height=100, bush_count=5, mushroom_count=3, crystal_count=2, seed=42, nest_zone=(50,50,5))
    assert len(mgr) == 10
    assert mgr.total_food() > 0


def test_food_manager_patch_at():
    mgr = FoodManager.generate(width=50, height=50, bush_count=5, mushroom_count=0, crystal_count=0, seed=42)
    p = mgr.patches[0]
    assert mgr.patch_at(float(p.x), float(p.y), radius=3.0) is p


def test_food_manager_tick_respawn():
    mgr = FoodManager.generate(width=50, height=50, bush_count=3, mushroom_count=0, crystal_count=0, seed=42)
    for p in mgr:
        p.collect(p.current_amount)
    assert mgr.total_food() == 0.0
    for _ in range(160):
        mgr.tick()
    assert mgr.total_food() > 0.0

# ── Terrain ───────────────────────────────────────────────

def test_terrain_defaults():
    t = TerrainMap(width=50, height=50, seed=42)
    assert len(t.elevation) == 50
    assert len(t.elevation[0]) == 50


def test_terrain_elevation_range():
    t = TerrainMap(width=30, height=30, seed=123)
    for row in t.elevation:
        for val in row:
            assert 0.0 <= val <= 1.0


def test_terrain_colour_range():
    t = TerrainMap(width=20, height=20, seed=42)
    for row in t.colours:
        for r, g, b in row:
            assert 0.0 <= r <= 1.0
            assert 0.0 <= g <= 1.0
            assert 0.0 <= b <= 1.0


def test_terrain_getters():
    t = TerrainMap(width=30, height=30, seed=42)
    assert 0.0 <= t.get_elevation(15.0, 15.0) <= 1.0
    r, g, b = t.get_rgb(15.0, 15.0)
    assert all(0 <= c <= 255 for c in (r, g, b))


def test_terrain_out_of_bounds():
    t = TerrainMap(width=30, height=30, seed=42)
    assert t.get_elevation(-5.0, -5.0) == 0.5
    assert t.get_colour(100.0, 100.0) == (0.5, 0.5, 0.5)


def test_terrain_reproducible():
    t1 = TerrainMap(width=30, height=30, seed=777)
    t2 = TerrainMap(width=30, height=30, seed=777)
    for x in range(30):
        for y in range(30):
            assert t1.elevation[x][y] == t2.elevation[x][y]

# ── Environment (composite) ──────────────────────────────

def test_environment_default_generation():
    env = Environment(width=100, height=100, seed=42)
    assert len(env.nest) > 0
    assert env.food.total_food() > 0


def test_environment_from_config():
    env = Environment.from_config({
        "width": 80, "height": 80, "seed": 99,
        "nest_start_size": 4,
        "obstacle_count": 15, "water_count": 2,
        "bush_count": 6, "mushroom_count": 4, "crystal_count": 2,
    })
    assert env.nest.start_radius == 4
    assert len(env.nest) == 81


def test_environment_per_cell_queries():
    env = Environment(width=50, height=50, seed=42)
    assert env.is_nest(25.0, 25.0)
    assert env.is_passable(25.0, 25.0)
    found = False
    for x in range(50):
        for y in range(50):
            if env.is_obstacle(float(x), float(y)):
                assert not env.is_passable(float(x), float(y))
                found = True
    assert found


def test_environment_movement_cost():
    env = Environment(width=30, height=30, seed=42)
    assert env.movement_cost(15.0, 15.0) == 1.0
    for x in range(30):
        for y in range(30):
            if env.is_obstacle(float(x), float(y)):
                assert env.movement_cost(float(x), float(y)) == float("inf")
                return


def test_environment_food_interaction():
    env = Environment(width=50, height=50, seed=42)
    patch = env.food.patches[0]
    taken = env.collect_food(float(patch.x), float(patch.y), 999.0)
    assert taken > 0
    assert not env.has_food(float(patch.x), float(patch.y))


def test_environment_tick():
    env = Environment(width=30, height=30, seed=42)
    assert env.tick_number == 0
    env.tick()
    assert env.tick_number == 1


def test_environment_food_respawn_after_tick():
    env = Environment(width=30, height=30, seed=42, bush_count=3, mushroom_count=0, crystal_count=0)
    for p in env.food:
        p.current_amount = 0.0
        p.depletion_timer = p.respawn_ticks
    env._refresh_food_flags()
    assert env.food.total_food() == 0.0
    for _ in range(160):
        env.tick()
    assert env.food.total_food() > 0.0


def test_environment_expand_nest():
    env = Environment(width=50, height=50, seed=42)
    initial = len(env.nest)
    env.expand_nest_toward(40, 40)
    assert len(env.nest) > initial


def test_environment_summary():
    env = Environment(width=50, height=50, seed=42)
    s = env.summary()
    assert s["width"] == 50
    for k in ("nest", "obstacles", "food", "terrain"):
        assert k in s

# ── YAML Config ──────────────────────────────────────

def test_yaml_config_roundtrip():
    import yaml
    with open("/root/ant-colony-sim/configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
    env = Environment.from_config(cfg.get("world", cfg))
    assert env.width > 0
    assert env.food.total_food() > 0

# ── Runner ──────────────────────────────────────────────

if __name__ == "__main__":
    passed = 0
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn()
                print("✓ " + name)
                passed += 1
            except Exception as e:
                print("✗ " + name + ": " + str(e))
                failed += 1
    print()
    print("=" * 50)
    print(f"  {passed} passed, {failed} failed, {passed+failed} total")
    if failed:
        sys.exit(1)