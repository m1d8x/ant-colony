"""Quick smoke test for agent creation and state machine basics."""
import sys
sys.path.insert(0, '/root/ant-colony-sim/src')

from ant_colony.pysimengine import Agent, World
from ant_colony.agents import (
    ForagerAgent, BuilderAgent, SoldierAgent, QueenAgent,
    create_agent, ANT_COLORS,
)

# Basic creation
f = ForagerAgent(x=10, y=10)
assert f.role == 'forager'
assert f.state == 'SEARCHING'

b = BuilderAgent(x=10, y=10)
assert b.role == 'builder'
assert b.state == 'IDLE'

s = SoldierAgent(x=10, y=10)
assert s.role == 'soldier'
assert s.state == 'PATROLLING'

q = QueenAgent(x=10, y=10)
assert q.role == 'queen'
assert q.state == 'SPAWNING'
assert q.speed == 0.0

# Factory
a = create_agent('forager', x=5, y=5)
assert isinstance(a, ForagerAgent)
assert a.role == 'forager'

# Colors
assert ANT_COLORS['forager'] == ((140, 90, 40), (200, 140, 60))
assert ANT_COLORS['builder'] == ((80, 55, 30), (120, 90, 50))
assert ANT_COLORS['soldier'] == ((140, 30, 30), (190, 60, 60))
assert ANT_COLORS['queen'] == ((120, 70, 160), (160, 110, 200))

# State machine transitions - forager
world = World(width=50, height=50)
world.nest_x = 25.0
world.nest_y = 25.0

f = ForagerAgent(x=25, y=25)
world.ants.append(f)

# Forager should start in SEARCHING
assert f.state == 'SEARCHING'

# Add food source nearby
world.food_sources.append((26, 26, 50.0))
f.update_state(world)
assert f.state == 'FOUND_FOOD', f"Expected FOUND_FOOD, got {f.state}"

# Pick up food
f.x = 26
f.y = 26
f.update_state(world)
assert f.state == 'CARRYING', f"Expected CARRYING, got {f.state}"
assert f.food_carrying > 0, "Should have picked up food"

# Carry to nest
f.update_state(world)
# Carry back toward nest direction

# Builder state machine
b = BuilderAgent(x=25, y=25)
world.ants.append(b)
world.colony_food_store = 100.0  # Above GATHER_THRESHOLD (80)

assert b.state == 'IDLE'
b.update_state(world)
assert b.state == 'GATHERING', f"Expected GATHERING, got {b.state}"

# Soldier state machine
s = SoldierAgent(x=25, y=25)
world.ants.append(s)
assert s.state == 'PATROLLING'

# Trigger combat via threat
world.threat_level = 0.5
s.update_state(world)
assert s.state == 'COMBAT', f"Expected COMBAT, got {s.state}"

# Queen state machine
q = QueenAgent(x=10, y=10)
world.ants.append(q)
assert q.state == 'SPAWNING'
world.nest_x = 25
world.nest_y = 25
q.update_state(world)
assert q.x == 25 and q.y == 25, "Queen should snap to nest"
assert q.state == 'SPAWNING'  # Still spawning (cooldown > 0)

# Test world tick with all agent types
world2 = World(width=50, height=50)
world2.nest_x = 25.0
world2.nest_y = 25.0
world2.colony_food_store = 100.0
world2.food_sources.append((40, 40, 50.0))

f = ForagerAgent(x=28, y=28)
b = BuilderAgent(x=27, y=27)
s = SoldierAgent(x=29, y=29)
q = QueenAgent(x=25, y=25)
for agent in [f, b, s, q]:
    world2.ants.append(agent)

for _ in range(10):
    world2.tick()

assert f.alive or True  # just check no crash
assert world2.tick_number == 10

print("All agent verification tests passed!")
