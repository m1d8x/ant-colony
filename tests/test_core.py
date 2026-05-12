"""
Tests for ant_colony core abstractions.
"""

from ant_colony.core import Vec2D, Agent, Behavior, World, SimConfig


class TestVec2D:
    def test_add(self):
        a = Vec2D(1, 2)
        b = Vec2D(3, 4)
        c = a + b
        assert c.x == 4 and c.y == 6

    def test_sub(self):
        a = Vec2D(5, 5)
        b = Vec2D(2, 3)
        c = a - b
        assert c.x == 3 and c.y == 2

    def test_mul(self):
        v = Vec2D(2, 3) * 4
        assert v.x == 8 and v.y == 12

    def test_div(self):
        v = Vec2D(10, 20) / 5
        assert v.x == 2 and v.y == 4

    def test_div_by_zero(self):
        v = Vec2D(5, 5) / 0
        assert v.x == 0 and v.y == 0

    def test_length(self):
        v = Vec2D(3, 4)
        assert v.length() == 5.0

    def test_normalized(self):
        v = Vec2D(3, 4).normalized()
        assert abs(v.length() - 1.0) < 0.001

    def test_normalized_zero(self):
        v = Vec2D(0, 0).normalized()
        assert v.x == 0 and v.y == 0

    def test_dot(self):
        d = Vec2D(1, 0).dot(Vec2D(0, 1))
        assert d == 0.0

    def test_distance(self):
        d = Vec2D(0, 0).distance_to(Vec2D(3, 4))
        assert d == 5.0


class TestAgent:
    def test_auto_uid(self):
        a = Agent()
        assert len(a.uid) == 8

    def test_custom_uid(self):
        a = Agent(uid="test01")
        assert a.uid == "test01"

    def test_defaults(self):
        a = Agent()
        assert a.position.x == 0.0
        assert a.position.y == 0.0
        assert a.size == 4.0
        assert a.max_speed == 2.0


class TestSimConfig:
    def test_defaults(self):
        cfg = SimConfig()
        assert cfg.width == 1200
        assert cfg.height == 800
        assert cfg.fps == 60
