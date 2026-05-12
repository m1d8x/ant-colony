"""
Core abstractions — subclass these to build ant colony simulations.

Mirrors the pysimengine.core patterns: Vec2D, Agent, Behavior, World.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Vec2D:
    """Simple 2D vector for positions and velocities."""
    x: float = 0.0
    y: float = 0.0

    def __add__(self, other: "Vec2D") -> "Vec2D":
        return Vec2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2D") -> "Vec2D":
        return Vec2D(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> "Vec2D":
        return Vec2D(self.x * scalar, self.y * scalar)

    def __truediv__(self, scalar: float) -> "Vec2D":
        if scalar == 0:
            return Vec2D(0, 0)
        return Vec2D(self.x / scalar, self.y / scalar)

    def __repr__(self) -> str:
        return f"Vec2D({self.x:.1f}, {self.y:.1f})"

    def length(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

    def normalized(self) -> "Vec2D":
        l = self.length()
        if l == 0:
            return Vec2D(0, 0)
        return self / l

    def dot(self, other: "Vec2D") -> float:
        return self.x * other.x + self.y * other.y

    def distance_to(self, other: "Vec2D") -> float:
        return (self - other).length()


@dataclass
class Agent:
    """
    Base agent. Subclass for domain-specific agents.

    Attributes:
        uid: Unique identifier (auto-generated if empty).
        position: Current position in world space.
        velocity: Current velocity vector.
        color: (R, G, B) tuple for rendering.
        size: Visual radius.
        max_speed: Speed cap.
        max_force: Steering force cap.
        vision_radius: Sensing range.
        state: Arbitrary state dict (energy, role, flags...).
    """
    uid: str = ""
    position: Vec2D = field(default_factory=Vec2D)
    velocity: Vec2D = field(default_factory=Vec2D)
    color: tuple = (200, 200, 200)
    size: float = 4.0
    max_speed: float = 2.0
    max_force: float = 0.1
    vision_radius: float = 50.0
    state: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.uid:
            import uuid
            self.uid = str(uuid.uuid4())[:8]


class Behavior(ABC):
    """
    A unit of agent logic returning a steering force.

    Implement execute() to compute a force vector for the agent.
    Override weight() for context-sensitive activation.
    """

    @abstractmethod
    def execute(self, agent: Agent, world: "World", dt: float) -> Vec2D:
        """Return a steering force vector for this tick."""
        ...

    def weight(self, agent: Agent, world: "World") -> float:
        """Returns 1.0 by default. Override for context-sensitive weighting."""
        return 1.0


class World(ABC):
    """
    The environment agents inhabit.

    Subclass and implement:
        - step(): one simulation tick
    """

    def __init__(self, width: float, height: float):
        self.width = float(width)
        self.height = float(height)
        self.agents: list[Agent] = []

    def add_agent(self, agent: Agent):
        self.agents.append(agent)

    def remove_agent(self, agent: Agent):
        if agent in self.agents:
            self.agents.remove(agent)

    def neighbors(self, agent: Agent) -> list[Agent]:
        """Return all agents within vision_radius."""
        return [
            other for other in self.agents
            if other is not agent
            and agent.position.distance_to(other.position) <= agent.vision_radius
        ]

    def apply_force(self, agent: Agent, force: Vec2D):
        """Simple Newton step — override for custom physics."""
        agent.velocity = agent.velocity + force
        speed = agent.velocity.length()
        if speed > agent.max_speed:
            agent.velocity = agent.velocity.normalized() * agent.max_speed

    def apply_physics(self, agent: Agent, dt: float = 1.0):
        """Integrate velocity → position, clamp to world bounds."""
        agent.position = agent.position + agent.velocity * dt
        agent.position.x = max(0, min(agent.position.x, self.width))
        agent.position.y = max(0, min(agent.position.y, self.height))

    @abstractmethod
    def step(self, dt: float):
        """Advance simulation by one tick."""
        ...


@dataclass
class SimConfig:
    """Simulation configuration with sensible defaults."""
    width: int = 1200
    height: int = 800
    title: str = "Ant Colony Simulation"
    fps: int = 60
    num_steps: int = 3600
    output_path: str = "output.mp4"
    output_fps: int = 30
    scale: float = 1.0
    n_colonies: int = 2
    num_agents: int = 50
