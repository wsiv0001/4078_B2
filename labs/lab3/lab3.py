# Import necessary libraries
from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

import argparse
import numpy as np

from labs.lab3.Planner import *
from src.pid_control import *
from sphero_env.envs.custom_maze_full import build_occupancy_grid

from contextlib import ExitStack, contextmanager

LAB1_SEED = 0
MAX_STEPS = 5000
GOAL_TOLERANCE = 0.1
WAYPOINT_PAUSE_STEPS = 10  # steps to pause/settle at each waypoint before moving on
map = build_occupancy_grid()

# World-frame position of the maze's designated starting plate
# (matches START_CELL in sphero_env.envs.custom_maze_full).
KNOWN_START_WORLD = np.array([-0.5, -0.5])

### Custom dynamics function for the Sphero robot - replace this with the one you developed in Lab 1
def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi

DT = 0.1
VELOCITY_LIMIT = 0.15

# Measured/estimated physical parameters
REAL_TOP_SPEED_MPS = 4
COMMANDED_MAX_SPEED = VELOCITY_LIMIT
SIM_SPEED_SCALE = REAL_TOP_SPEED_MPS / COMMANDED_MAX_SPEED

MAX_TURN_RATE = 100
MAX_ACCEL = 0.2

from src.dynamics import *

# def dynamics(state, action):
#     x, y, heading, speed = state
#     speed_cmd, heading_cmd = action

#     #Heading: unchanged

#     heading_error = wrap_angle(heading_cmd - heading)
#     max_turn_delta = MAX_TURN_RATE * DT
#     heading_new = wrap_angle(heading + np.clip(heading_error, -max_turn_delta, max_turn_delta))
#     # heading_new = wrap_angle(heading+heading_error)

#     # Speed: ramp toward target speed, still in "commanded" units
#     max_speed_delta = MAX_ACCEL * DT
#     speed_error = speed_cmd - speed
#     speed_new = speed + np.clip(speed_error, -max_speed_delta, max_speed_delta)
#     # speed_new = speed + speed_error
#     speed_new = np.clip(speed_new, 0.0, COMMANDED_MAX_SPEED)

#     # Position: this is the only place the real-world scale enters
#     x_new = x + speed_new * SIM_SPEED_SCALE * np.sin(heading_new) * DT
#     y_new = y + speed_new * SIM_SPEED_SCALE * np.cos(heading_new) * DT

#     return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)

### If needed, add the EKF from lab 2 here too, and integrate below.

class Controller:
    def __init__(self, dt=0.1):
        self.dt = dt

    def compute_action(self, state, waypoint):
        """
        Fill in this function to implement a simple controller that computes the action based on the current state and the waypoint.
        """
        target = waypoint
        pos = state[:2]
        delta = target - pos
        distance = np.linalg.norm(delta)

        heading = np.arctan2(delta[0], delta[1], dtype=np.float32)
        speed_cmd = np.clip(pid_distance.compute(0.0, distance), -MAX_SPEED, MAX_SPEED)
        action = (speed_cmd, heading)

        return action  # Replace this with your controller's output

def make_sim_env():
    return SpheroEnv(
        dt=DT,
        max_steps=MAX_STEPS,
        vel_limit=VELOCITY_LIMIT,
        world_width=1.25,
        world_height=1.25,
        goal_pos=(0.5, 0.5),
        goal_tolerance=GOAL_TOLERANCE,
        occupancy_grid=map,
        grid_resolution=0.125,
        dynamics=dynamics,
        obs_noise_std_pos=0.05,
        process_noise_std_speed=0.005,
        process_noise_std_heading=0.01,
        obs_noise_std_vel=0.025,
        render_mode="human",
        window_size=(800, 800),
    )

def make_real_env(api):
    return Robot(
        api=api,
        dt=DT,
        max_steps=MAX_STEPS,
        vel_limit=VELOCITY_LIMIT,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=GOAL_TOLERANCE,
        render_mode="human",
        window_size=(800, 800),
    )

@contextmanager
def managed_env(sim: bool):
    if sim:
        sim_env = make_sim_env()
        sim_env.set_log_path("logs/lab3_sim.csv")
        sim_env.start_logging()
        try:
            yield sim_env
        finally:
            sim_env.stop_logging()
            sim_env.close()
    else:
        with ExitStack() as stack:
            selected_toy, _ = scan_and_connect()
            print(f"Selected: {selected_toy.name}")

            api = stack.enter_context(SpheroEduAPI(selected_toy))
            real_env = make_real_env(api)
            real_env.set_log_path("logs/lab3_real.csv")

            real_env.start_logging()
            try:
                yield real_env
            finally:
                real_env.close()
                real_env.stop_logging()

def control_loop(control_env):

    obs, _ = control_env.reset(seed=LAB1_SEED)

    if isinstance(control_env, SpheroEnv):
        # Sim: force a known start position - SpheroEnv respects this override,
        # so obs is already in the map's world frame once we read it back.
        control_env.state_true[0:3] = np.array([-0.5, -0.5, 0.0])
        control_env.state_odom[0:3] = np.array([-0.5, -0.5, 0.0])
        obs = control_env.state_true.copy()
        frame_offset = np.zeros(2)
    else:
        # Real robot: the override above doesn't affect real hardware, so
        # obs is in the robot's OWN frame (zeroed wherever it physically was
        # at reset/connect time). Assuming the robot was physically placed
        # on the maze's start plate, compute the constant offset that maps
        # its own frame onto the map's world frame.
        frame_offset = KNOWN_START_WORLD - obs[:2]
        obs = obs.copy()
        obs[:2] += frame_offset

    def to_map_frame(raw_obs):
        raw_obs = raw_obs.copy()
        raw_obs[:2] = raw_obs[:2] + frame_offset
        return raw_obs

    controller = Controller(dt=control_env.dt)
    planner = Planner(map=map, dt=control_env.dt)

    # Planner.plan returns absolute waypoints already in the map's frame.
    waypoints = planner.plan(obs, control_env.goal_pos)

    planner.debug_print(waypoints)
    planner.debug_plot(obs[:2], control_env.goal_pos, waypoints)

    steps = 0
    goal_reached = False

    for waypoint in waypoints:
        # control_env.goal_pos = waypoint
        if goal_reached:
            break

        # Keep stepping toward THIS waypoint until we actually reach it
        # (or run out of steps) before moving on to the next one.
        reached_waypoint = False
        while not reached_waypoint and steps < MAX_STEPS:
            action = controller.compute_action(obs, waypoint)
            raw_obs, _, terminated, truncated, info = control_env.step(action)
            obs = to_map_frame(raw_obs)
            steps += 1

            control_env.render()

            if (obs[0]-waypoint[0])**2 + (obs[1]-waypoint[1])**2 < control_env.goal_tolerance**2:
                reached_waypoint = True  # Move on to the next waypoint

            if (obs[0]-control_env.goal_pos[0])**2 + (obs[1]-control_env.goal_pos[1])**2 < control_env.goal_tolerance**2:
                goal_reached = True
                break  # Goal reached, stop stepping toward this waypoint

        # Pause at the waypoint for a few steps (commanding zero speed)
        # before heading toward the next one - lets the robot settle
        # instead of immediately carrying momentum into the next turn.
        if reached_waypoint and not goal_reached:
            stop_action = (0.0, obs[2])  # zero speed, hold current heading
            for _ in range(WAYPOINT_PAUSE_STEPS):
                if steps >= MAX_STEPS:
                    break
                raw_obs, _, terminated, truncated, info = control_env.step(stop_action)
                obs = to_map_frame(raw_obs)
                steps += 1
                control_env.render()

    control_env.emergency_stop()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run simulation")
    args = parser.parse_args(argv)

    with managed_env(args.sim) as control_env:
        control_loop(control_env)

if __name__ == "__main__":
    main()