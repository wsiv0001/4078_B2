"""
procedural_maze.py

Adapter that wraps maze_generator.MazeGenerator so it exposes the SAME
module-level interface as custom_maze_full.py:

    build_occupancy_grid() -> np.ndarray
    START_CELL, GOAL_CELL
    cell_to_world(cell)    -> np.ndarray world position
    GRID_RESOLUTION, PLATE_SIZE
    maze_width(), maze_height()

This lets lab3.py's `MAZE` global point at a randomly generated maze
instead of the fixed physical one, without changing anything else in
lab3.py.

IMPORTANT - SIM ONLY
---------------------
The walls in a generated maze are purely part of the SIMULATOR's
collision model - they do not correspond to any physical plates or
barriers in the lab room. Running this against the REAL robot would
mean the planner is routing around obstacles that don't physically
exist (and/or driving through space the sim thinks is free but the
real room isn't laid out to match). lab3.py checks the SIM_ONLY flag
below and will refuse to start a real-robot run while this module is
selected - don't bypass that check.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# Adjust this import to wherever maze_generator.py actually lives in your
# project (e.g. `from src.maze_generator import ...` if it's under src/).
from src.sphero_env.envs.maze_generator import MazeGenerator, grid_to_world as _grid_to_world


# ============================================================
# Knobs - change these to get a different generated maze
# ============================================================

SEED = 100                    # change (or randomize) this for a new layout
OCC_SIZE = 15                # occupancy grid is OCC_SIZE x OCC_SIZE cells
GRID_RESOLUTION = 0.125      # metres per occupancy cell (sim-only, arbitrary)

# Kept only so lab3.py's `MAZE.maze_width() * MAZE.PLATE_SIZE` formula for
# world_width/world_height still works - there's no physical "plate" here,
# so this is just "world metres per logical grid unit".
PLATE_SIZE = GRID_RESOLUTION

# This maze is simulator-only - see module docstring above.
SIM_ONLY = True

# ============================================================
# Generate once at import time so START_CELL/GOAL_CELL/the grid are
# fixed for the whole run (re-import or call regenerate() for a new one)
# ============================================================

_generator = MazeGenerator(width=OCC_SIZE, height=OCC_SIZE, seed=SEED)
_grid, START_CELL, GOAL_CELL = _generator.generate()


def regenerate(seed: int) -> None:
    """Generate a new maze with a different seed, replacing the module's
    current grid/START_CELL/GOAL_CELL in place. Call this BEFORE lab3.py
    reads MAZE.build_occupancy_grid()/START_CELL/GOAL_CELL (i.e. before
    `map = MAZE.build_occupancy_grid()` runs), not after."""
    global _generator, _grid, START_CELL, GOAL_CELL, SEED
    SEED = seed
    _generator = MazeGenerator(width=OCC_SIZE, height=OCC_SIZE, seed=seed)
    _grid, START_CELL, GOAL_CELL = _generator.generate()


def build_occupancy_grid() -> np.ndarray:
    """Return the generated occupancy grid (0 = free, 1 = wall)."""
    return _grid


def cell_to_world(cell: Tuple[int, int]) -> np.ndarray:
    """Convert a logical (toy 5x5 grid) cell to a world-frame position.

    Verified to match Planner's world_to_grid/grid_to_world centering
    convention exactly (round-tripping a cell through cell_to_world and
    back lands on the same occupancy-grid cell).
    """
    x, y = _grid_to_world(cell, OCC_SIZE, OCC_SIZE, GRID_RESOLUTION)
    return np.array([x, y], dtype=np.float32)


def maze_width() -> int:
    return OCC_SIZE


def maze_height() -> int:
    return OCC_SIZE


if __name__ == "__main__":
    print("Occupancy grid:")
    print(build_occupancy_grid().astype(int))
    print("START_CELL:", START_CELL, "-> world", cell_to_world(START_CELL))
    print("GOAL_CELL:", GOAL_CELL, "-> world", cell_to_world(GOAL_CELL))