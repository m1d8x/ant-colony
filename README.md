# Ant Colony Simulation

Emergent ant colony simulation with pheromone trails, role-based agents,
and interactive visualisation.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Headless batch run (200 steps)
python -m ant_colony --mode headless --steps 200

# Interactive pygame mode (if display is available)
python -m ant_colony --mode pygame

# Custom config
python -m ant_colony --config configs/default.yaml --steps 500
```

## CLI

| Flag           | Description                                |
|----------------|--------------------------------------------|
| `--mode`, `-m` | `headless` or `pygame` (auto-detected)     |
| `--config`, `-c` | Path to YAML config file                |
| `--steps`, `-s` | Number of simulation steps (headless)    |
| `--output`, `-o` | Output path (headless recording)        |
| `--version`, `-v` | Show version                            |

## Architecture

```
ant_colony/
├── agents/         — Role-specific ant types (forager, soldier, queen, builder)
│   ├── forager.py  — SEARCH → FOUND → CARRYING → RETURNING FSM
│   ├── soldier.py  — PATROL → COMBAT → GUARD FSM
│   ├── queen.py    — Stationary spawner
│   └── builder.py  — GATHER → BUILD FSM
├── behaviors/      — Steering behaviours (pheromone-follow, obstacle-avoid, etc.)
├── pysimengine/    — Core engine (Agent, World, Behavior base classes)
├── world/          — Environment generation (terrain, obstacles, food, nest)
├── pheromones/     — PHGrid: numpy-backed multi-layer pheromone grid
├── renderers/      — Pygame + headless rendering backends
├── simulation.py   — AntColonySimulation: main loop coordinator
└── __main__.py     — CLI entry point
```

## Tests

```bash
python -m pytest tests/ -v
```

## Configuration

Edit `configs/default.yaml` to tune:
- World dimensions (200×200 cells)
- Obstacle / food / terrain generation
- Simulation seed for reproducibility
