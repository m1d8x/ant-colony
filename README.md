# Ant Colony Simulation

Emergent ant colony simulation with pheromone trails, multiple agent types, and procedural world generation.

## Install

```bash
cd ant-colony-sim
pip install -e .
```

## Run

```bash
# Headless mode (batch, logs stats to stdout)
python -m ant_colony --mode headless --steps 1000

# Interactive mode with display
python -m ant_colony --mode pygame

# Custom config
python -m ant_colony --mode headless --steps 500 --config configs/default.yaml

# Change agent count / colonies via config
python -m ant_colony --mode headless --steps 200 -s 200
```

## Controls (Pygame mode)

| Key | Action |
|-----|--------|
| ESC | Quit |
| SPACE | Pause |

## Architecture

- **pysimengine/** — Core simulation engine. `World` holds pheromone grids, obstacle map, and ant list. `tick()` runs: pheromone diffusion/evaporation → colony manager → agent FSMs → behaviors.
- **agents/** — Role-specific FSMs: `ForagerAgent` (SEARCHING→FOUND_FOOD→CARRYING→RETURNING), `BuilderAgent` (IDLE→GATHERING→BUILDING), `SoldierAgent` (PATROLLING→COMBAT→GUARDING), `QueenAgent` (SPAWNING/IDLE).
- **behaviors/** — Composable behaviors: `FollowGradient` pheromone steering, `DepositTrail` pheromone laying, `WanderWithPersistence` random walk, `AvoidObstacles` collision avoidance, `ColonyManager` spawning/role reassignment.
- **world/** — Procedural world generation: `Nest` (expandable tiles), `ObstacleGrid` (blob-shaped rocks/water), `FoodManager` (bush/mushroom/crystal patches with depletion/respawn), `TerrainMap` (Perlin noise elevation).
- **renderers/** — Pygame and headless renderers with pheromone heatmaps, HUD with live stats.
- **simulation.py** — Glue layer that wires everything together: creates World + Environment, spawns agents, runs the tick loop.

## Tests

```bash
pytest tests/ -v
```

## Config

Edit `config/config.yaml` in the package directory or pass `--config path/to/config.yaml`:

```yaml
n_colonies: 2
num_agents: 50
seed: 42
world:
  width: 200
  height: 200
```
