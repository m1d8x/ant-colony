"""Tests for the PHGrid pheromone system."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

sys.path.insert(0, "/root/ant-colony-sim/src")

from ant_colony.pheromones import PHGrid, PheromoneType

# ══════════════════════════════════════════════════════════════════════════════
#  Construction & type tests
# ══════════════════════════════════════════════════════════════════════════════


def test_phgrid_defaults():
    grid = PHGrid(width=50, height=40)
    assert grid.width == 50
    assert grid.height == 40
    assert grid.n_types == 4
    assert grid.cell_size == 4
    assert grid.grid.shape == (4, 40, 50)
    assert grid.grid.dtype == np.float32
    assert grid.grid.sum() == 0.0


def test_phgrid_custom_n_types():
    grid = PHGrid(width=10, height=10, n_types=2, cell_size=8)
    assert grid.n_types == 2
    assert grid.cell_size == 8
    assert grid.grid.shape == (2, 10, 10)


def test_pheromone_type_enum():
    assert PheromoneType.AT_HOME == 0
    assert PheromoneType.TO_FOOD == 1
    assert PheromoneType.TO_BUILD == 2
    assert PheromoneType.DANGER == 3


def test_pheromone_type_colors():
    assert PheromoneType.AT_HOME.color == (0, 0, 255)
    assert PheromoneType.TO_FOOD.color == (0, 255, 0)
    assert PheromoneType.TO_BUILD.color == (255, 165, 0)
    assert PheromoneType.DANGER.color == (255, 0, 0)


def test_shape_property():
    grid = PHGrid(60, 30)
    assert grid.shape == (60, 30)


# ══════════════════════════════════════════════════════════════════════════════
#  Deposit & get
# ══════════════════════════════════════════════════════════════════════════════


def test_deposit_and_get():
    grid = PHGrid(10, 10)
    grid.deposit((2.0, 3.0), PheromoneType.TO_FOOD, 5.0)
    assert grid.get((2.0, 3.0), PheromoneType.TO_FOOD) == 5.0


def test_deposit_cell():
    grid = PHGrid(10, 10)
    grid.deposit_cell(5, 5, PheromoneType.AT_HOME, 10.0)
    assert grid.get_cell(5, 5, PheromoneType.AT_HOME) == 10.0


def test_deposit_accumulates():
    grid = PHGrid(10, 10)
    grid.deposit_cell(3, 3, PheromoneType.DANGER, 1.0)
    grid.deposit_cell(3, 3, PheromoneType.DANGER, 2.0)
    assert grid.get_cell(3, 3, PheromoneType.DANGER) == 3.0


def test_get_empty():
    grid = PHGrid(10, 10)
    assert grid.get_cell(0, 0, PheromoneType.AT_HOME) == 0.0


def test_deposit_out_of_bounds():
    grid = PHGrid(10, 10)
    grid.deposit((-5.0, 3.0), PheromoneType.TO_FOOD, 99.0)
    assert grid.total() == 0.0

    grid.deposit((999.0, 999.0), PheromoneType.TO_FOOD, 99.0)
    assert grid.total() == 0.0


def test_sub_pixel_mapping():
    """Default cell_size=4, so world coords (4, 4) map to cell (1, 1)."""
    grid = PHGrid(10, 10, cell_size=4)
    grid.deposit((4.0, 4.0), PheromoneType.TO_FOOD, 10.0)
    assert grid.get_cell(1, 1, PheromoneType.TO_FOOD) == 10.0
    # Also check that (3.9, 4.0) maps to cell (0, 1) since 3.9 // 4 == 0
    assert grid._world_to_cell(3.9, 4.0) == (0, 1)


def test_set_cell():
    grid = PHGrid(10, 10)
    grid.set_cell(7, 7, PheromoneType.TO_BUILD, 42.0)
    assert grid.get_cell(7, 7, PheromoneType.TO_BUILD) == 42.0


def test_set_cell_overwrite():
    grid = PHGrid(10, 10)
    grid.set_cell(4, 4, PheromoneType.AT_HOME, 10.0)
    grid.set_cell(4, 4, PheromoneType.AT_HOME, 20.0)
    assert grid.get_cell(4, 4, PheromoneType.AT_HOME) == 20.0


# ══════════════════════════════════════════════════════════════════════════════
#  Evaporation
# ══════════════════════════════════════════════════════════════════════════════


def test_evaporate_reduces_values():
    grid = PHGrid(10, 10)
    grid.set_cell(5, 5, PheromoneType.TO_FOOD, 100.0)
    grid.evaporate(rate=0.1)
    # After evaporation at 0.1, 100 * 0.9 = 90
    assert abs(grid.get_cell(5, 5, PheromoneType.TO_FOOD) - 90.0) < 0.01


def test_evaporate_all_channels():
    grid = PHGrid(10, 10)
    for t in PheromoneType:
        grid.set_cell(5, 5, t, 50.0)
    grid.evaporate(rate=0.2)
    for t in PheromoneType:
        assert abs(grid.get_cell(5, 5, t) - 40.0) < 0.01


def test_evaporate_no_negatives():
    grid = PHGrid(10, 10)
    grid.set_cell(5, 5, PheromoneType.AT_HOME, 0.001)
    grid.evaporate(rate=0.5)
    assert grid.get_cell(5, 5, PheromoneType.AT_HOME) >= 0.0


def test_evaporate_zero_rate():
    grid = PHGrid(10, 10)
    grid.set_cell(5, 5, PheromoneType.TO_FOOD, 100.0)
    grid.evaporate(rate=0.0)
    assert grid.get_cell(5, 5, PheromoneType.TO_FOOD) == 100.0


def test_evaporate_full_decay():
    grid = PHGrid(10, 10)
    grid.set_cell(5, 5, PheromoneType.TO_FOOD, 100.0)
    grid.evaporate(rate=1.0)
    assert grid.get_cell(5, 5, PheromoneType.TO_FOOD) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Diffusion
# ══════════════════════════════════════════════════════════════════════════════


def test_diffusion_spreads_to_neighbors():
    """Deposit in center cell, diffuse — neighbours should get some mass."""
    grid = PHGrid(10, 10)
    grid.set_cell(5, 5, PheromoneType.TO_FOOD, 100.0)
    grid.diffuse()
    # Centre should have less than 100
    c = grid.get_cell(5, 5, PheromoneType.TO_FOOD)
    assert c < 100.0, f"centre should lose mass, got {c}"
    # Neighbours should have received some
    total_neighbours = sum(
        grid.get_cell(nx, ny, PheromoneType.TO_FOOD)
        for nx, ny in [(4, 5), (6, 5), (5, 4), (5, 6)]
    )
    assert total_neighbours > 0.0, "neighbours should receive mass"


def test_diffusion_conservation():
    """Total mass should be conserved (approximately) after diffusion."""
    grid = PHGrid(20, 20)
    grid.set_cell(10, 10, PheromoneType.TO_FOOD, 100.0)
    total_before = grid.total(PheromoneType.TO_FOOD)
    grid.diffuse()
    total_after = grid.total(PheromoneType.TO_FOOD)
    # Allow small floating point differences
    assert abs(total_after - total_before) < 0.01, (
        f"mass changed from {total_before:.4f} to {total_after:.4f}"
    )


def test_diffusion_equilibrates():
    """Repeated diffusion should spread mass evenly across the grid."""
    grid = PHGrid(10, 10)
    grid.set_cell(5, 5, PheromoneType.AT_HOME, 100.0)
    for _ in range(200):
        grid.diffuse()
    # After many steps, mass should be roughly uniform
    per_cell = grid.total(PheromoneType.AT_HOME) / (10 * 10)
    for y in range(10):
        for x in range(10):
            val = grid.get_cell(x, y, PheromoneType.AT_HOME)
            assert val > 0.0, f"cell ({x},{y}) has zero mass after equilibration"
            # Within 5x of average after equilibration is generous
            assert val < per_cell * 5, (
                f"cell ({x},{y}) has {val:.4f}, expected ~{per_cell:.4f}"
            )


def test_diffusion_multi_channel():
    """Diffusion on one channel should not affect others."""
    grid = PHGrid(10, 10)
    grid.set_cell(5, 5, PheromoneType.AT_HOME, 100.0)
    grid.set_cell(5, 5, PheromoneType.TO_FOOD, 200.0)
    grid.diffuse()
    # Both channels should still be independent
    assert grid.total(PheromoneType.AT_HOME) == pytest.approx(100.0, rel=1e-3)
    assert grid.total(PheromoneType.TO_FOOD) == pytest.approx(200.0, rel=1e-3)


# ══════════════════════════════════════════════════════════════════════════════
#  Obstacle mask
# ══════════════════════════════════════════════════════════════════════════════


def test_obstacle_mask_blocks_diffusion():
    """Mass should not diffuse through a blocked cell."""
    grid = PHGrid(10, 10)
    grid.set_cell(1, 5, PheromoneType.TO_FOOD, 100.0)
    # Block the cell between source and rest of grid
    grid.obstacle_mask[5, 2] = True
    grid.diffuse()
    # Cells beyond the obstacle should have less than without obstacle
    beyond = grid.get_cell(3, 5, PheromoneType.TO_FOOD)
    # Without obstacle, cell (3,5) would get some indirect diffusion
    # With obstacle blocking (2,5), mass must go around — it should be lower
    # than the neighbour cell (1,4) or (1,6)
    neighbor = max(
        grid.get_cell(1, 4, PheromoneType.TO_FOOD),
        grid.get_cell(1, 6, PheromoneType.TO_FOOD),
    )
    assert beyond <= neighbor * 1.1 + 0.01, (
        f"beyond obstacle {beyond:.4f} should be <= lateral neighbour {neighbor:.4f}"
    )


def test_obstacle_mask_empty_cells():
    """Unblocked cells should still diffuse normally."""
    grid = PHGrid(10, 10)
    grid.set_cell(5, 5, PheromoneType.AT_HOME, 100.0)
    grid.diffuse()
    total = grid.total(PheromoneType.AT_HOME)
    assert total > 0


def test_obstacle_mask_downsample():
    """set_obstacle_mask should correctly down-sample pixel coordinates."""
    grid = PHGrid(3, 3, cell_size=4)
    pixel_mask = np.zeros((12, 12), dtype=bool)
    # Block pixel (0, 0) — should make cell (0, 0) blocked
    pixel_mask[0, 0] = True
    grid.set_obstacle_mask(pixel_mask)
    assert bool(grid.obstacle_mask[0, 0]) is True
    # Cell (2, 2) should be unblocked
    assert bool(grid.obstacle_mask[2, 2]) is False


def test_obstacle_bounce_back_conservation():
    """Mass should be conserved even with obstacles blocking diffusion."""
    grid = PHGrid(5, 5)
    grid.set_cell(2, 2, PheromoneType.TO_FOOD, 100.0)
    # Block most of the grid
    grid.obstacle_mask[0:5, 0] = True  # left column
    grid.obstacle_mask[0:5, 4] = True  # right column
    total_before = grid.total(PheromoneType.TO_FOOD)
    grid.diffuse()
    total_after = grid.total(PheromoneType.TO_FOOD)
    assert abs(total_after - total_before) < 0.1, (
        f"mass changed: {total_before:.4f} -> {total_after:.4f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Gradient sensing
# ══════════════════════════════════════════════════════════════════════════════


def test_sample_ahead_center():
    """With pheromone directly ahead, center should be strongest."""
    grid = PHGrid(100, 100, cell_size=1)
    # Place pheromone right in front of a west-facing ant
    grid.deposit((55.0, 50.0), PheromoneType.TO_FOOD, 100.0)
    left, center, right = grid.sample_ahead(
        (50.0, 50.0), 0.0, vision_radius=5.0, ptype=PheromoneType.TO_FOOD,
    )
    assert center > left, f"center {center} should be > left {left}"
    assert center > right, f"center {center} should be > right {right}"


def test_sample_ahead_right():
    """With pheromone to the right, right sensor should be strongest."""
    grid = PHGrid(100, 100, cell_size=1)
    # Place pheromone slightly to the right of straight ahead
    # heading=0 (east), right sensor at +0.35 rad
    right_angle = 0.35
    rx = 50.0 + math.cos(right_angle) * 5.0
    ry = 50.0 + math.sin(right_angle) * 5.0
    grid.deposit((rx, ry), PheromoneType.TO_FOOD, 100.0)

    left, center, right = grid.sample_ahead(
        (50.0, 50.0), 0.0, vision_radius=5.0, ptype=PheromoneType.TO_FOOD,
    )
    assert right > center, f"right {right} should be > center {center}"
    assert right > left, f"right {right} should be > left {left}"


def test_sample_ahead_left():
    """With pheromone to the left, left sensor should be strongest."""
    grid = PHGrid(100, 100, cell_size=1)
    left_angle = -0.35
    lx = 50.0 + math.cos(left_angle) * 5.0
    ly = 50.0 + math.sin(left_angle) * 5.0
    grid.deposit((lx, ly), PheromoneType.TO_FOOD, 100.0)

    left, center, right = grid.sample_ahead(
        (50.0, 50.0), 0.0, vision_radius=5.0, ptype=PheromoneType.TO_FOOD,
    )
    assert left > center, f"left {left} should be > center {center}"
    assert left > right, f"left {left} should be > right {right}"


# ══════════════════════════════════════════════════════════════════════════════
#  Steering helper
# ══════════════════════════════════════════════════════════════════════════════


def test_steer_straight_when_center_strongest():
    grid = PHGrid(100, 100, cell_size=1)
    grid.deposit((55.0, 50.0), PheromoneType.AT_HOME, 100.0)
    delta = grid.steer_toward_gradient(
        (50.0, 50.0), 0.0, 5.0, PheromoneType.AT_HOME, turn_rate=0.3,
    )
    assert delta == 0.0, f"should go straight, got delta={delta}"


def test_steer_right():
    grid = PHGrid(100, 100, cell_size=1)
    right_angle = 0.35
    rx = 50.0 + math.cos(right_angle) * 5.0
    ry = 50.0 + math.sin(right_angle) * 5.0
    grid.deposit((rx, ry), PheromoneType.TO_FOOD, 100.0)
    delta = grid.steer_toward_gradient(
        (50.0, 50.0), 0.0, 5.0, PheromoneType.TO_FOOD, turn_rate=0.3,
    )
    assert delta > 0.0, f"should turn right, got delta={delta}"


def test_steer_left():
    grid = PHGrid(100, 100, cell_size=1)
    left_angle = -0.35
    lx = 50.0 + math.cos(left_angle) * 5.0
    ly = 50.0 + math.sin(left_angle) * 5.0
    grid.deposit((lx, ly), PheromoneType.TO_BUILD, 100.0)
    delta = grid.steer_toward_gradient(
        (50.0, 50.0), 0.0, 5.0, PheromoneType.TO_BUILD, turn_rate=0.3,
    )
    assert delta < 0.0, f"should turn left, got delta={delta}"


def test_steer_zero_when_no_pheromone():
    grid = PHGrid(100, 100)
    delta = grid.steer_toward_gradient(
        (50.0, 50.0), 0.0, 5.0, PheromoneType.DANGER, turn_rate=0.3,
    )
    assert delta == 0.0, "no pheromone → no turn"


# ══════════════════════════════════════════════════════════════════════════════
#  Edge cases
# ══════════════════════════════════════════════════════════════════════════════


def test_single_cell_grid():
    grid = PHGrid(1, 1)
    grid.deposit_cell(0, 0, PheromoneType.AT_HOME, 10.0)
    grid.diffuse()  # should not crash
    grid.evaporate(0.1)
    assert grid.get_cell(0, 0, PheromoneType.AT_HOME) > 0.0


def test_resize():
    grid = PHGrid(10, 10)
    grid.set_cell(3, 3, PheromoneType.TO_FOOD, 10.0)
    grid.set_cell(9, 9, PheromoneType.TO_FOOD, 20.0)
    grid.resize(20, 20)
    assert grid.width == 20
    assert grid.height == 20
    # Cell (3,3) data preserved
    assert grid.get_cell(3, 3, PheromoneType.TO_FOOD) == 10.0
    # Cell (9,9) data preserved
    assert grid.get_cell(9, 9, PheromoneType.TO_FOOD) == 20.0
    # New cells are zero
    assert grid.get_cell(15, 15, PheromoneType.TO_FOOD) == 0.0


def test_resize_smaller():
    grid = PHGrid(20, 20)
    grid.set_cell(15, 15, PheromoneType.DANGER, 50.0)
    grid.resize(10, 10)
    assert grid.width == 10
    assert grid.height == 10
    # Cell (15,15) was clipped
    assert grid.get_cell(5, 5, PheromoneType.DANGER) == 0.0
    # But overlapping data should be preserved
    grid.set_cell(3, 3, PheromoneType.AT_HOME, 30.0)
    grid.resize(5, 5)
    assert grid.get_cell(3, 3, PheromoneType.AT_HOME) == 30.0


def test_clear():
    grid = PHGrid(10, 10)
    for t in PheromoneType:
        grid.set_cell(5, 5, t, 50.0)
    grid.clear()
    assert grid.total() == 0.0
    for t in PheromoneType:
        assert grid.get_cell(5, 5, t) == 0.0


def test_large_deposit():
    grid = PHGrid(10, 10)
    grid.deposit((3.0, 3.0), PheromoneType.TO_FOOD, 1e6)
    assert grid.get_cell(0, 0, PheromoneType.TO_FOOD) == 1e6
    grid.evaporate(rate=0.5)
    assert grid.get_cell(0, 0, PheromoneType.TO_FOOD) == 5e5


def test_out_of_bounds_get_returns_zero():
    grid = PHGrid(10, 10)
    assert grid.get((-1.0, 5.0), PheromoneType.TO_FOOD) == 0.0
    assert grid.get((100.0, 100.0), PheromoneType.TO_FOOD) == 0.0
    assert grid.get_cell(-1, 5, PheromoneType.TO_FOOD) == 0.0
    assert grid.get_cell(10, 10, PheromoneType.TO_FOOD) == 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Integration: combined diffusion + evaporation cycle
# ══════════════════════════════════════════════════════════════════════════════


def test_diffusion_evaporation_cycle():
    """Simulate a full tick cycle: diffuse + evaporate."""
    grid = PHGrid(20, 20)
    grid.deposit_cell(10, 10, PheromoneType.TO_FOOD, 100.0)

    for _ in range(10):
        grid.diffuse()
        grid.evaporate(rate=0.03)

    # After 10 ticks, total should be less than initial (evaporation)
    assert grid.total(PheromoneType.TO_FOOD) < 100.0
    # Mass should be spread out
    assert grid.get_cell(10, 10, PheromoneType.TO_FOOD) < 100.0


def test_integration_multi_type_cycle():
    """Multiple pheromone types should evolve independently."""
    grid = PHGrid(20, 20)
    grid.deposit_cell(5, 5, PheromoneType.AT_HOME, 50.0)
    grid.deposit_cell(15, 15, PheromoneType.DANGER, 80.0)

    for _ in range(5):
        grid.diffuse()
        grid.evaporate(rate=0.05)

    # Both still present
    assert grid.total(PheromoneType.AT_HOME) > 0
    assert grid.total(PheromoneType.DANGER) > 0
    # No cross-contamination
    assert grid.get_cell(5, 5, PheromoneType.DANGER) == 0.0
    assert grid.get_cell(15, 15, PheromoneType.AT_HOME) == 0.0


def test_conservation_full_cycle():
    """Mass conservation check: diffusion conserves, evaporation removes known fraction."""
    grid = PHGrid(10, 10)
    grid.set_cell(5, 5, PheromoneType.TO_FOOD, 100.0)
    total_before = grid.total(PheromoneType.TO_FOOD)

    grid.diffuse()
    after_diffuse = grid.total(PheromoneType.TO_FOOD)
    assert abs(after_diffuse - total_before) < 0.01, "diffusion should conserve mass"

    # Now evaporate
    grid.evaporate(rate=0.1)
    expected = total_before * 0.9
    assert abs(grid.total(PheromoneType.TO_FOOD) - expected) < 0.01, (
        f"expected {expected:.4f}, got {grid.total(PheromoneType.TO_FOOD):.4f}"
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Runner
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    imported_pytest = False
    try:
        import pytest
        imported_pytest = True
    except ImportError:
        pass

    if imported_pytest:
        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
    else:
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
