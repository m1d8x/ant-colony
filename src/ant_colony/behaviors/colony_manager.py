"""ColonyManager — self-regulating colony governance.

Tracks food store, spawns ants dynamically, and reassigns roles based on
colony needs (starving → more foragers, threatened → more soldiers,
surplus → more builders).
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

from ant_colony.pysimengine import Agent

if TYPE_CHECKING:
    from ant_colony.pysimengine import World


SPAWN_FOOD_COST = 15.0
SPAWN_THRESHOLD = 30.0
MAX_POPULATION = 60
STARVING_THRESHOLD = 50.0
SURPLUS_THRESHOLD = 150.0
ROLE_CHECK_INTERVAL = 10
SPAWN_INTERVAL = 5
FORAGING_MEMORY_DECAY = 0.90


class ColonyManager:
    """Governs colony-level decisions.  Runs once per tick on the world."""

    priority: int = 200

    def __init__(self):
        self._last_spawn_tick = -SPAWN_INTERVAL  # fire on first tick
        self._last_role_tick = 0
        self._food_income_rate = 0.0
        self._prev_food_store = 100.0

    def colony_stats(self, world: World) -> dict:
        ants = [a for a in world.ants if a.alive]
        counts: dict[str, int] = {}
        for a in ants:
            counts[a.role] = counts.get(a.role, 0) + 1
        total_food_carried = sum(a.food_carrying for a in ants)
        return {
            "ants": len(ants),
            "roles": counts,
            "food_store": round(world.colony_food_store, 1),
            "food_carried": round(total_food_carried, 1),
            "food_income_rate": round(self._food_income_rate, 2),
            "tick": world.tick_number,
        }

    def _spawn_ant(self, world: World) -> Agent | None:
        if world.colony_food_store < SPAWN_THRESHOLD:
            return None
        alive = len([a for a in world.ants if a.alive])
        if alive >= MAX_POPULATION:
            return None

        world.colony_food_store -= SPAWN_FOOD_COST

        angle = random.uniform(0, 2 * math.pi)
        dist = random.uniform(2.0, 6.0)
        x = world.nest_x + math.cos(angle) * dist
        y = world.nest_y + math.sin(angle) * dist
        x = max(0.0, min(world.width - 1, x))
        y = max(0.0, min(world.height - 1, y))

        ant = Agent(x=x, y=y, direction=angle, speed=random.uniform(1.5, 2.5), role="forager")
        world.ants.append(ant)
        return ant

    def _reassign_roles(self, world: World) -> None:
        """Dynamically redistribute roles based on colony needs.

        Fix: when roles have zero population (e.g., all builders), we pull
        from the most over-represented role instead of trying to reassign
        from under-represented roles (which have nobody to give).
        """
        ants = [a for a in world.ants if a.alive]
        if not ants:
            return

        food_ok = world.colony_food_store > STARVING_THRESHOLD
        surplus = world.colony_food_store > SURPLUS_THRESHOLD
        threatened = world.threat_level > 0.3

        if not food_ok:
            target_forager = 0.80
            target_soldier = 0.15
        elif threatened:
            target_forager = 0.50
            target_soldier = 0.35
        elif surplus:
            target_forager = 0.55
            target_soldier = 0.10
        else:
            target_forager = 0.65
            target_soldier = 0.10

        target_builder = 1.0 - target_forager - target_soldier
        n = len(ants)

        current = {"forager": 0, "soldier": 0, "builder": 0}
        for a in ants:
            current[a.role] = current.get(a.role, 0) + 1

        targets = {
            "forager": max(0, round(n * target_forager)),
            "soldier": max(0, round(n * target_soldier)),
            "builder": max(0, round(n * target_builder)),
        }

        # Ensure total targets = n
        total_target = sum(targets.values())
        if total_target != n:
            diff = n - total_target
            targets["forager"] += diff  # assign remainder to foragers

        # Compute diffs: positive = needs more, negative = has too many
        diffs = {role: targets[role] - current.get(role, 0) for role in targets}

        # Roles that need to LOSE ants (positive surplus = has excess)
        surplus_roles = [r for r in targets if diffs[r] < 0]

        # Roles that need to GAIN ants (negative surplus = needs more)
        deficit_roles = [r for r in targets if diffs[r] > 0]

        if not surplus_roles or not deficit_roles:
            return  # no reassignment needed

        # Reassign: for each deficit role, pull from the most surplus role
        for deficit in deficit_roles:
            while diffs[deficit] > 0 and surplus_roles:
                # Find candidates from the most surplus role
                surplus_role = max(surplus_roles, key=lambda r: -diffs[r])
                n_needed = diffs[deficit]
                # Pick ants to reassign
                candidates = [a for a in ants if a.role == surplus_role]
                if not candidates:
                    surplus_roles.remove(surplus_role)
                    continue
                random.shuffle(candidates)
                n_to_take = min(n_needed, abs(diffs[surplus_role]), len(candidates))
                for ant in candidates[:n_to_take]:
                    ant.role = deficit
                diffs[deficit] -= n_to_take
                diffs[surplus_role] += n_to_take
                if diffs[surplus_role] >= 0:
                    surplus_roles.remove(surplus_role)

    def update(self, _agent: Agent | None, world: World) -> None:
        # Track food income (actual deposits at nest)
        delta = world.colony_food_store - self._prev_food_store
        self._food_income_rate = (
            self._food_income_rate * FORAGING_MEMORY_DECAY
            + max(0, delta) * (1 - FORAGING_MEMORY_DECAY)
        )
        self._prev_food_store = world.colony_food_store

        # Spawn
        if world.tick_number - self._last_spawn_tick >= SPAWN_INTERVAL:
            self._spawn_ant(world)
            self._last_spawn_tick = world.tick_number

        # Role reassignment
        if world.tick_number - self._last_role_tick >= ROLE_CHECK_INTERVAL:
            self._reassign_roles(world)
            self._last_role_tick = world.tick_number
