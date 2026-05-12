"""Pheromone system for ant colony simulation.

Provides grid-based pheromone deposits with diffusion, evaporation,
and gradient-sensing for ant steering behaviour.
"""

from ant_colony.pheromones.ph_grid import PHGrid, PheromoneType

__all__ = [
    "PHGrid",
    "PheromoneType",
]
