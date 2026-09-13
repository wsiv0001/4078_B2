import heapq

import numpy as np
import matplotlib.pyplot as plt

from sphero_env.envs.custom_maze_full import build_occupancy_grid
from src.dynamics import *

map = build_occupancy_grid()

### Implement a planner and controller for the Sphero robot to navigate to a goal position in the environment.
class Planner:
    def __init__(self, map, dt=DT, resolution=0.125):
        """
        The map (input) is the occupancy grid produced by the maze generator:
            map : a 2D numpy array of shape (h, w) with values in {0, 1}
                  1 = wall / obstacle cell
                  0 = free cell
        World origin (0, 0) is at the centre of the grid, so cell (i, j) maps
        to world coordinates via the environment's grid resolution.
        """
        self.map = map
        self.dt = dt
        self.resolution = resolution
        self.h, self.w = map.shape

    def world_to_grid(self, pos):
        """Convert a world (x, y) position to a (row, col) grid cell.

        The grid's centre index is (w-1)/2, (h-1)/2 - NOT w/2, h/2 - since
        the occupancy grid has odd dimensions (2*MAZE_WIDTH+1 etc.) and the
        true centre cell sits at an integer index. Using w/2 here would be
        off by half a cell and can land on the wrong-parity (wall) cell.
        """
        col = int(round(pos[0] / self.resolution + (self.w - 1) / 2))
        row = int(round(-pos[1] / self.resolution + (self.h - 1) / 2))
        return row, col

    def grid_to_world(self, cell):
        """Convert a (row, col) grid cell back to a world (x, y) position."""
        row, col = cell
        x = (col - (self.w - 1) / 2) * self.resolution
        y = -(row - (self.h - 1) / 2) * self.resolution
        return np.array([x, y], dtype=np.float32)

    def neighbours(self, cell):
        """Return walkable 4-connected neighbours of a cell."""
        row, col = cell
        candidates = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        for r, c in candidates:
            if 0 <= r < self.h and 0 <= c < self.w and self.map[r, c] == 0:
                yield (r, c)

    def corners_only(self, path):
        """Keep only the cells where the path changes direction (plus the
        final cell), so the robot travels in straight lines between turns
        instead of getting a waypoint at every single grid cell."""
        if len(path) <= 2:
            return path

        corners = [path[0]]
        prev_direction = None

        for i in range(1, len(path)):
            r0, c0 = path[i - 1]
            r1, c1 = path[i]
            direction = (r1 - r0, c1 - c0)

            if prev_direction is None:
                prev_direction = direction
            elif direction != prev_direction:
                corners.append(path[i - 1])
                prev_direction = direction

        corners.append(path[-1])
        return corners

    def plan(self, state, goal):
        """
        Simple A* over the occupancy grid.

        Inputs:
            state : current state [x, y, heading, speed] (world frame)
            goal  : goal position [x, y] (world frame)
        Returns:
            waypoints : list of absolute world-frame [x, y] positions
                        leading from the current position to the goal.
        """
        start = self.world_to_grid(state[:2])
        end = self.world_to_grid(goal)

        # Standard A* bookkeeping
        open_set = [(0, start)]
        came_from = {}
        cost_so_far = {start: 0}

        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        found = False
        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                found = True
                break

            for next_cell in self.neighbours(current):
                new_cost = cost_so_far[current] + 1
                if next_cell not in cost_so_far or new_cost < cost_so_far[next_cell]:
                    cost_so_far[next_cell] = new_cost
                    priority = new_cost + heuristic(next_cell, end)
                    heapq.heappush(open_set, (priority, next_cell))
                    came_from[next_cell] = current

        if not found:
            # No path found - just head straight for the goal
            return [np.array(goal, dtype=np.float32)]

        # Reconstruct path from end back to start
        path = [end]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()

        # Keep only the turning points, so the robot drives in straight
        # lines between waypoints instead of stopping at every grid cell.
        path = self.corners_only(path)

        # Convert grid cells to world-frame waypoints (skip the start cell)
        waypoints = [self.grid_to_world(cell) for cell in path[1:]]
        return waypoints

    # ------------------------------------------------------------------
    # Debug visualization only - NOT part of the A* algorithm above.
    # Call this manually from lab3.py after plan() if you want to sanity
    # check the path, e.g.:
    #   waypoints = planner.plan(obs, control_env.goal_pos)
    #   planner.debug_plot(obs[:2], control_env.goal_pos, waypoints)
    # ------------------------------------------------------------------
    def debug_plot(self, start, goal, waypoints, save_path=None):
        """Plot the occupancy grid with the start, goal and planned
        waypoints overlaid, so you can visually check the path makes sense."""
        half_w = (self.w / 2) * self.resolution
        half_h = (self.h / 2) * self.resolution

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            self.map,
            cmap="gray_r",
            origin="upper",
            extent=[-half_w, half_w, -half_h, half_h],
        )

        if waypoints:
            wp = np.array(waypoints)
            ax.plot(wp[:, 0], wp[:, 1], "o-", color="lime", label="waypoints")
            for i, (x, y) in enumerate(wp):
                ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(4, 4), color="lime")

        ax.plot(start[0], start[1], "b*", markersize=15, label="start")
        ax.plot(goal[0], goal[1], "r*", markersize=15, label="goal")

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend()
        ax.set_title("Planned path over occupancy grid")

        if save_path:
            fig.savefig(save_path)
            print(f"Saved plot to {save_path}")
        else:
            plt.show()

    def debug_print(self, waypoints):
        """Print the planned waypoints to the terminal, one per line."""
        print("Planned waypoints:")
        for i, wp in enumerate(waypoints):
            print(f"  {i}: ({wp[0]:.3f}, {wp[1]:.3f})")