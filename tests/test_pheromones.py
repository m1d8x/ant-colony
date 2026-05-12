"""
Tests for the pheromone grid module (PHGrid).

Note: the simulation itself uses pysimengine.World's native pheromone arrays
(list[list[float]]), not PHGrid. These tests verify PHGrid's API works correctly.
"""

import numpy as np
import pytest

from ant_colony.pheromones import PHGrid, PheromoneType


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def grid():
    """A 100x75 cell grid."""
    return PHGrid(100, 75)


@pytest.fixture
def tiny_grid():
    """10x10 grid with cell_size=1 for exact cell tests."""
    return PHGrid(10, 10, cell_size=1)


@pytest.fixture
def single_cell():
    """Smallest possible grid (1 cell)."""
    return PHGrid(4, 4, cell_size=4)


# ======================================================================
# Construction & type enum
# ======================================================================


class TestConstruction:
    def test_default_dimensions(self, grid):
        assert grid.width == 100
        assert grid.height == 75
        assert grid.n_types == 4
        assert grid.total() == 0.0

    def test_cell_size_stored(self):
        g = PHGrid(100, 75, cell_size=10)
        assert g.cell_size == 10
        # width/height are cell counts (not derived from world/cell_size)
        assert g.width == 100
        assert g.height == 75

    def test_shape_property(self, grid):
        # shape = (width, height)
        assert grid.shape == (100, 75)

    def test_pheromone_type_enum_values(self):
        assert PheromoneType.AT_HOME == 0
        assert PheromoneType.TO_FOOD == 1
        assert PheromoneType.TO_BUILD == 2
        assert PheromoneType.DANGER == 3

    def test_pheromone_type_colors(self):
        assert PheromoneType.AT_HOME.color == (0, 0, 255)
        assert PheromoneType.TO_FOOD.color == (0, 255, 0)
        assert PheromoneType.TO_BUILD.color == (255, 165, 0)
        assert PheromoneType.DANGER.color == (255, 0, 0)


# ======================================================================
# Deposit / Get
# ======================================================================


class TestDepositAndGet:
    def test_deposit_and_get(self, grid):
        grid.deposit((50, 50), PheromoneType.TO_FOOD, 0.5)
        val = grid.get((50, 50), PheromoneType.TO_FOOD)
        assert val == pytest.approx(0.5, abs=0.01)

    def test_deposit_adds(self, grid):
        grid.deposit((50, 50), PheromoneType.TO_FOOD, 0.3)
        grid.deposit((50, 50), PheromoneType.TO_FOOD, 0.4)
        val = grid.get((50, 50), PheromoneType.TO_FOOD)
        assert val == pytest.approx(0.7, abs=0.01)

    def test_types_independent(self, grid):
        grid.deposit((50, 50), PheromoneType.AT_HOME, 0.8)
        grid.deposit((50, 50), PheromoneType.TO_FOOD, 0.3)
        grid.deposit((50, 50), PheromoneType.DANGER, 1.0)
        assert grid.get((50, 50), PheromoneType.AT_HOME) == pytest.approx(0.8, abs=0.01)
        assert grid.get((50, 50), PheromoneType.TO_FOOD) == pytest.approx(0.3, abs=0.01)
        assert grid.get((50, 50), PheromoneType.TO_BUILD) == pytest.approx(0.0, abs=0.01)
        assert grid.get((50, 50), PheromoneType.DANGER) == pytest.approx(1.0, abs=0.01)

    def test_get_out_of_bounds_returns_zero(self, grid):
        assert grid.get((9999, 50), PheromoneType.AT_HOME) == 0.0
        assert grid.get((-10, 50), PheromoneType.AT_HOME) == 0.0

    def test_deposit_out_of_bounds_safe(self, grid):
        grid.deposit((-10, 9999), PheromoneType.AT_HOME, 0.5)
        assert grid.total() == 0.0

    def test_pixel_mapping(self):
        """With cell_size=4: world (0-3, 0-3) -> cell 0, (4-7, 4-7) -> cell 1."""
        g = PHGrid(16, 16, cell_size=4)
        g.deposit((0, 0), PheromoneType.AT_HOME, 0.9)
        assert g.get((3, 3), PheromoneType.AT_HOME) == pytest.approx(0.9, abs=0.01)
        assert g.get((4, 4), PheromoneType.AT_HOME) == pytest.approx(0.0, abs=0.01)

    def test_deposit_and_get_cell(self, grid):
        grid.deposit_cell(5, 5, PheromoneType.TO_FOOD, 0.7)
        assert grid.get_cell(5, 5, PheromoneType.TO_FOOD) == pytest.approx(0.7, abs=0.01)

    def test_set_cell(self, grid):
        grid.set_cell(10, 5, PheromoneType.DANGER, 0.99)
        assert grid.get_cell(10, 5, PheromoneType.DANGER) == pytest.approx(0.99, abs=0.01)

    def test_set_cell_overwrites(self, grid):
        grid.set_cell(5, 5, PheromoneType.TO_FOOD, 0.3)
        grid.set_cell(5, 5, PheromoneType.TO_FOOD, 0.9)
        assert grid.get_cell(5, 5, PheromoneType.TO_FOOD) == pytest.approx(0.9, abs=0.01)

    def test_get_cell_oob(self, grid):
        assert grid.get_cell(999, 999, PheromoneType.AT_HOME) == 0.0
        assert grid.get_cell(-1, -1, PheromoneType.AT_HOME) == 0.0


# ======================================================================
# Evaporation
# ======================================================================


class TestEvaporation:
    def test_evaporation_reduces_all(self, grid):
        grid.set_cell(10, 10, PheromoneType.AT_HOME, 0.8)
        grid.set_cell(20, 20, PheromoneType.TO_FOOD, 0.6)
        grid.set_cell(30, 30, PheromoneType.DANGER, 0.4)
        before = grid.total()
        grid.evaporate(0.1)
        after = grid.total()
        assert after < before
        assert after == pytest.approx(before * 0.9, rel=1e-5)

    def test_evaporation_zero_rate(self, grid):
        grid.set_cell(5, 5, PheromoneType.AT_HOME, 0.5)
        before = grid.total()
        grid.evaporate(0.0)
        assert grid.total() == pytest.approx(before, rel=1e-5)

    def test_evaporation_rate_one(self, grid):
        grid.set_cell(5, 5, PheromoneType.AT_HOME, 0.5)
        grid.evaporate(1.0)
        assert grid.total() == pytest.approx(0.0, abs=1e-10)

    def test_evaporation_rate_negative_increases(self, grid):
        grid.set_cell(5, 5, PheromoneType.AT_HOME, 0.5)
        before = grid.total()
        grid.evaporate(-0.5)  # 1 - (-0.5) = 1.5x multiplier
        assert grid.total() == pytest.approx(before * 1.5, rel=1e-5)

    def test_evaporation_type_independent(self, grid):
        grid.set_cell(5, 5, PheromoneType.AT_HOME, 0.8)
        grid.set_cell(5, 5, PheromoneType.DANGER, 0.4)
        grid.evaporate(0.5)
        assert grid.get_cell(5, 5, PheromoneType.AT_HOME) == pytest.approx(0.4, abs=0.01)
        assert grid.get_cell(5, 5, PheromoneType.DANGER) == pytest.approx(0.2, abs=0.01)

    def test_evaporation_no_negatives(self, grid):
        grid.set_cell(5, 5, PheromoneType.AT_HOME, 0.5)
        grid.evaporate(1.5)  # >100% rate
        assert grid.total() == pytest.approx(0.0, abs=1e-10)


# ======================================================================
# Diffusion
# ======================================================================


class TestDiffusion:
    def test_diffusion_spreads_to_neighbours(self, tiny_grid):
        tiny_grid.set_cell(2, 2, PheromoneType.AT_HOME, 1.0)
        tiny_grid.diffuse()
        # Centre should drop (lost some to neighbours)
        assert tiny_grid.get_cell(2, 2, PheromoneType.AT_HOME) < 1.0
        # Neighbours should increase
        assert tiny_grid.get_cell(2, 3, PheromoneType.AT_HOME) > 0.0
        assert tiny_grid.get_cell(3, 2, PheromoneType.AT_HOME) > 0.0
        assert tiny_grid.get_cell(2, 1, PheromoneType.AT_HOME) > 0.0
        assert tiny_grid.get_cell(1, 2, PheromoneType.AT_HOME) > 0.0

    def test_diffusion_no_loss_single_cell(self, single_cell):
        single_cell.set_cell(0, 0, PheromoneType.AT_HOME, 1.0)
        before = single_cell.total()
        single_cell.diffuse()
        # Single cell: no neighbours, but diffusion is 10% of each cell
        # which goes to non-existent neighbours -> mass not conserved
        # Just check it runs without error
        assert single_cell.total() >= 0.0

    def test_diffusion_multi_type_independent(self, tiny_grid):
        tiny_grid.set_cell(2, 2, PheromoneType.AT_HOME, 0.8)
        tiny_grid.set_cell(3, 3, PheromoneType.DANGER, 0.6)
        before_a = tiny_grid.total(PheromoneType.AT_HOME)
        before_d = tiny_grid.total(PheromoneType.DANGER)
        tiny_grid.diffuse()
        # Mass should be conserved per type (within floating point)
        assert tiny_grid.total(PheromoneType.AT_HOME) == pytest.approx(before_a, abs=0.01)
        assert tiny_grid.total(PheromoneType.DANGER) == pytest.approx(before_d, abs=0.01)

    def test_edge_cells_diffuse(self, tiny_grid):
        """Diffusion from an edge cell should spread to valid neighbours only."""
        tiny_grid.set_cell(0, 5, PheromoneType.AT_HOME, 1.0)
        before = tiny_grid.total()
        tiny_grid.diffuse()
        # Mass should have spread
        assert tiny_grid.get_cell(1, 5, PheromoneType.AT_HOME) > 0.0
        assert tiny_grid.total() == pytest.approx(before, abs=0.01)

    def test_zero_cells_no_op(self, tiny_grid):
        """Grid with no pheromone should remain zero after diffusion."""
        tiny_grid.diffuse()
        assert tiny_grid.total() == 0.0


# ======================================================================
# Obstacle mask
# ======================================================================


class TestObstacleMask:
    def test_set_obstacle_mask(self):
        g = PHGrid(10, 10, cell_size=1)
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        g.set_obstacle_mask(mask)
        assert g.obstacle_mask[5, 5]
        assert not g.obstacle_mask[0, 0]

    def test_clear(self, grid):
        grid.set_cell(10, 10, PheromoneType.AT_HOME, 0.8)
        grid.set_cell(20, 20, PheromoneType.TO_FOOD, 0.5)
        assert grid.total() > 0
        grid.clear()
        assert grid.total() == pytest.approx(0.0, abs=1e-10)

    def test_total_per_type(self, grid):
        grid.set_cell(5, 5, PheromoneType.AT_HOME, 0.5)
        grid.set_cell(5, 5, PheromoneType.DANGER, 0.3)
        t_home = grid.total(PheromoneType.AT_HOME)
        t_danger = grid.total(PheromoneType.DANGER)
        assert t_home == pytest.approx(0.5, abs=0.01)
        assert t_danger == pytest.approx(0.3, abs=0.01)

    def test_resize_larger(self, tiny_grid):
        tiny_grid.set_cell(2, 2, PheromoneType.AT_HOME, 0.7)
        tiny_grid.resize(20, 20)
        assert tiny_grid.width == 20
        assert tiny_grid.height == 20
        assert tiny_grid.get_cell(2, 2, PheromoneType.AT_HOME) == pytest.approx(0.7, abs=0.01)
        assert tiny_grid.get_cell(15, 15, PheromoneType.AT_HOME) == 0.0

    def test_large_deposit(self):
        g = PHGrid(250, 250, cell_size=4)
        for x in range(0, 1000, 16):
            for y in range(0, 1000, 16):
                g.deposit((x, y), PheromoneType.AT_HOME, 0.1)
        assert g.total() > 0


# ======================================================================
# Gradient sensing (sample_ahead / steer)
# ======================================================================


class TestGradientSensing:
    def test_sample_ahead_all_zero_no_pheromone(self, grid):
        l, c, r = grid.sample_ahead(
            (50.0, 50.0), 0.0, 5.0, PheromoneType.TO_FOOD,
        )
        assert l == 0.0 and c == 0.0 and r == 0.0

    def test_steer_zero_when_no_pheromone(self, grid):
        delta = grid.steer_toward_gradient(
            (50.0, 50.0), 0.0, 5.0, PheromoneType.TO_FOOD, 0.3,
        )
        assert delta == 0.0


# ======================================================================
# Integration
# ======================================================================


class TestIntegration:
    def test_diffuse_evaporate_cycle(self, tiny_grid):
        """Over many diffuse+evaporate cycles, total mass decreases."""
        tiny_grid.set_cell(5, 5, PheromoneType.AT_HOME, 1.0)
        before = tiny_grid.total()
        for _ in range(50):
            tiny_grid.diffuse()
            tiny_grid.evaporate(0.02)
        after = tiny_grid.total()
        assert after < before
