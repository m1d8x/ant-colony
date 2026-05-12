"""
AntColonySimulation — orchestrates the full simulation loop.

Brings together pysimengine.World, world.Environment (terrain / obstacles /
food), agent types (ForagerAgent / BuilderAgent / SoldierAgent / QueenAgent),
and renderers into a single runnable simulation.
"""

from __future__ import annotations

import math
import random
import time
from typing import Any

from ant_colony.pysimengine import World
from ant_colony.world import Environment
from ant_colony.agents import create_agent
from ant_colony.renderers import PyGameRenderer, HeadlessRenderer


class AntColonySimulation:
    """Main simulation coordinator.

    Responsibilities:
      - Build the World and Environment from config
      - Spawn initial ant colonies (Queen, Foragers, Builders, Soldiers)
      - Run the step loop (world.tick() → render)
      - Provide run_pygame() and run_headless() entry methods
    """

    def __init__(self, config: dict[str, Any]):
        self.config = dict(config)
        self.step_count = 0
        self._running = True
        self._paused = False
        self._renderer: PyGameRenderer | HeadlessRenderer | None = None

        # Build world from config
        self._create_world()
        self._setup_environment()
        self._spawn_colonies()

    # ── Config helpers ──────────────────────────────────────────────────

    @staticmethod
    def _cfg(config: dict, *keys: str, default: Any = None) -> Any:
        """Extract a config value with nested fallback.

        Tries each key in order, including dotted paths into nested dicts.
        Example: _cfg(config, 'world_width', 'width', default=200) checks
        config['world_width'], then config['width'], then
        config['world']['width'].
        """
        for key in keys:
            if "." in key:
                parts = key.split(".")
                d = config
                for p in parts:
                    if isinstance(d, dict) and p in d:
                        d = d[p]
                    else:
                        break
                else:
                    return d
            else:
                if key in config:
                    return config[key]
        return default

    # ── Initialisation ──────────────────────────────────────────────────

    def _create_world(self):
        """Create the pysimengine.World (pheromone grids, obstacle grid)."""
        w = int(self._cfg(self.config, "world_width", "world.width", "width", default=200))
        h = int(self._cfg(self.config, "world_height", "world.height", "height", default=200))
        self.world = World(width=w, height=h)

    def _setup_environment(self):
        """Generate terrain / obstacles / food via world.Environment and
        copy the results into the pysimengine.World data structures."""
        # Flatten the config: promote nested dicts (e.g. world: {width: 200})
        # to top-level so Environment.from_config can find them.
        flat = dict(self.config)
        for val in list(flat.values()):
            if isinstance(val, dict):
                flat.update(val)
        env = Environment.from_config(flat)
        self.environment = env

        # Copy obstacles into World's 2D bool grid
        for x in range(self.world.width):
            for y in range(self.world.height):
                self.world.obstacles[x][y] = env.is_obstacle(float(x), float(y))

        # Copy food sources into World's food_sources list
        for patch in env.food.patches:
            if patch.available:
                self.world.food_sources.append((
                    float(patch.x),
                    float(patch.y),
                    patch.current_amount,
                ))

        # Set nest position from Environment
        self.world.nest_x = float(env.nest.cx)
        self.world.nest_y = float(env.nest.cy)

    def _spawn_colonies(self):
        """Place nest and spawn initial ants for each configured colony."""
        n_colonies = self.config.get("n_colonies", 1)
        ants_per = self.config.get("num_agents", 50)
        margin = 20.0

        for cid in range(n_colonies):
            if cid == 0:
                # First colony uses the Environment-generated nest position
                nx, ny = self.world.nest_x, self.world.nest_y
            else:
                # Additional colonies are offset to avoid overlap
                nx = margin + cid * 60.0
                ny = margin + cid * 60.0
                # Ensure in bounds
                nx = min(nx, float(self.world.width) - margin)
                ny = min(ny, float(self.world.height) - margin)

            # Queen with very long lifespan (effectively immortal for simulation)
            queen = create_agent(
                role="queen", x=nx, y=ny,
                color=(120, 70, 160),
                max_age=999999,
            )
            self.world.ants.append(queen)

            # Foragers (~60%)
            n_foragers = max(3, int(ants_per * 0.60))
            for _ in range(n_foragers):
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(10, 30)
                ax = nx + math.cos(angle) * dist
                ay = ny + math.sin(angle) * dist
                ax = max(0.0, min(float(self.world.width) - 1, ax))
                ay = max(0.0, min(float(self.world.height) - 1, ay))
                ant = create_agent(
                    role="forager", x=ax, y=ay,
                    direction=random.uniform(0, 2 * math.pi),
                    speed=random.uniform(1.5, 3.0),
                )
                self.world.ants.append(ant)

            # Soldiers (~20%)
            n_soldiers = max(1, int(ants_per * 0.20))
            for _ in range(n_soldiers):
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(15, 40)
                ax = nx + math.cos(angle) * dist
                ay = ny + math.sin(angle) * dist
                ax = max(0.0, min(float(self.world.width) - 1, ax))
                ay = max(0.0, min(float(self.world.height) - 1, ay))
                ant = create_agent(
                    role="soldier", x=ax, y=ay,
                    direction=random.uniform(0, 2 * math.pi),
                    speed=random.uniform(1.0, 2.0),
                )
                self.world.ants.append(ant)

            # Builders (~20%) — expand the nest when food is plentiful
            n_builders = max(1, int(ants_per * 0.20))
            for _ in range(n_builders):
                angle = random.uniform(0, 2 * math.pi)
                dist = random.uniform(10, 25)
                ax = nx + math.cos(angle) * dist
                ay = ny + math.sin(angle) * dist
                ax = max(0.0, min(float(self.world.width) - 1, ax))
                ay = max(0.0, min(float(self.world.height) - 1, ay))
                ant = create_agent(
                    role="builder", x=ax, y=ay,
                    direction=random.uniform(0, 2 * math.pi),
                    speed=random.uniform(1.0, 2.0),
                )
                self.world.ants.append(ant)

    def create_renderer(self, mode: str = "pygame"):
        """Create and attach the appropriate renderer."""
        if mode == "pygame":
            self._renderer = PyGameRenderer(self.world, self.config)
        else:
            self._renderer = HeadlessRenderer(self.world, self.config)

    # ── Step logic ──────────────────────────────────────────────────────

    def step(self):
        """Execute one simulation tick.

        Delegates to ``world.tick()`` which handles:
          pheromone evaporation/diffusion → colony manager → agent FSMs →
          gradient-following → trail-depositing → wandering → obstacle avoidance
        """
        # Check renderer's pause state (toggled via SPACE in pygame renderer)
        if getattr(self._renderer, "_paused", False):
            return
        if self._paused:
            return

        # Let the world run its built-in tick (pheromones + behaviors)
        self.world.tick()

        # Render
        if self._renderer:
            self._renderer.render(self.world, self.world.tick_number)

        self.step_count = self.world.tick_number

    # ── Run modes ───────────────────────────────────────────────────────

    def run_pygame(self):
        """Interactive pygame simulation loop.

        Press ESC or close window to exit.  SPACE to pause.
        """
        self.create_renderer("pygame")
        self._running = True
        self._paused = False

        alive = sum(1 for a in self.world.ants if a.alive)
        print(
            f"[pygame] starting — {len(self.world.ants)} agents "
            f"({alive} alive), "
            f"{len(self.world.food_sources)} food sources",
            flush=True,
        )

        while self._running:
            if not self._renderer.handle_events():
                break
            self.step()

        self._renderer.close()
        print(f"[pygame] simulation ended after {self.step_count} steps", flush=True)

    def run_headless(self, steps: int | None = None):
        """Non-interactive simulation for batch runs.

        Args:
            steps: Number of steps to run.  Defaults to config num_steps.
        """
        max_steps = steps if steps is not None else self.config.get("num_steps", 1000)
        self.create_renderer("headless")

        alive = sum(1 for a in self.world.ants if a.alive)
        print(
            f"[headless] running {max_steps} steps — "
            f"{len(self.world.ants)} agents ({alive} alive), "
            f"{len(self.world.food_sources)} food sources",
            flush=True,
        )

        start = time.time()
        for i in range(max_steps):
            self.step()
            # Headless renderer logs every N steps; we can skip rendering
            # after a threshold to speed up pure-stat runs
            if max_steps > 500 and i > max_steps - 100:
                # Only render last 100 steps for final status
                pass
            if not self._renderer.handle_events():
                break

        elapsed = time.time() - start
        self._renderer.close()

        alive = sum(1 for a in self.world.ants if a.alive)
        total = len(self.world.ants)
        # Count by role
        roles: dict[str, int] = {}
        for a in self.world.ants:
            if a.alive:
                roles[a.role] = roles.get(a.role, 0) + 1
        role_str = ", ".join(f"{k}={v}" for k, v in sorted(roles.items()))
        print(
            f"[headless] {max_steps} steps in {elapsed:.1f}s "
            f"({max_steps / elapsed:.0f} steps/s) — "
            f"{alive}/{total} ants remaining  [{role_str}]  "
            f"food_store={self.world.colony_food_store:.0f}",
            flush=True,
        )
