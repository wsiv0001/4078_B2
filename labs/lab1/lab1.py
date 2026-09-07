# ---------------------------------- Imports --------------------------------- #
# Import necessary libraries
from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

# Standard Library Imports
import argparse
import numpy as np
from contextlib import ExitStack, contextmanager

# group_b2 imports (only the three files you have)
from src.dynamics import *
from src.pid_control import *
from src.shared_functions import *

# ----------------------- Global constants for lab1.py ----------------------- #
LAB1_SEED = 0
DT = 0.1
RENDER_MODE = True

WAYPOINTS = np.array([
    [0.0, 0.5],
    [0.5, 0.5],
    [0.5, 0.0],
    [0.0, 0.0]
], dtype=np.float32)

GOAL_TOLERANCE = 0.05
WAIT_STEPS = 10        # steps to sit at each waypoint (DT=0.1 -> 2s)
MAX_TOTAL_STEPS = 2000  # safety cap

# ---------------------------------------------------------------------------- #
#                                 ENVIRONMENTS                                 #
# ---------------------------------------------------------------------------- #
# ------------------------------------ Sim ----------------------------------- #
def make_sim_env():
    return SpheroEnv(
        dt=DT,
        max_steps=5000,
        vel_limit=MAX_SPEED,
        world_width=5.0,
        world_height=5.0,
        goal_pos=tuple(WAYPOINTS[0]),
        goal_tolerance=GOAL_TOLERANCE,
        occupancy_grid=None,
        dynamics=dynamics,
        obs_noise_std_pos=0.00,
        process_noise_std_speed=0.00,
        process_noise_std_heading=0.00,
        obs_noise_std_vel=0,
        render_mode="human",
        window_size=(800, 800),
    )

# ----------------------------------- real ----------------------------------- #
def make_real_env(api):
    return Robot(
        api=api,
        dt=DT,
        max_steps=5000,
        vel_limit=MAX_SPEED,
        world_width=5.0,
        world_height=5.0,
        goal_pos=tuple(WAYPOINTS[0]),
        goal_tolerance=GOAL_TOLERANCE,
        render_mode="human",
        window_size=(800, 800),
    )

# ---------------------------- Environment Manager --------------------------- #
SESSION_NAME = "session1_SB-DAE7"  # rename this per run - it becomes the folder under logs/lab1_logs/

@contextmanager
def managed_env(sim: bool):
    if sim:
        sim_log_path = get_log_path(SESSION_NAME, is_real=False)
        sim_env = make_sim_env()
        sim_env.set_log_path(sim_log_path)
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
            real_log_path = get_log_path(SESSION_NAME, is_real=True)
            real_env.set_log_path(real_log_path)

            real_env.start_logging()
            try:
                yield real_env
            finally:
                real_env.close()
                real_env.stop_logging()

# ---------------------------------------------------------------------------- #
#                                 CONTROL LOOP                                 #
# ---------------------------------------------------------------------------- #
def control_loop(control_env, waypoints=WAYPOINTS, wait_steps=WAIT_STEPS):

    initial_heading = np.arctan2(waypoints[0][0], waypoints[0][1], dtype=np.float32)
    obs, _ = control_env.reset(
        seed=LAB1_SEED,
        options={"initial_heading": initial_heading}
    )

    waypoint_idx = 0
    wait_counter = 0
    waypoint_reached = False
    heading = initial_heading

    for _ in range(MAX_TOTAL_STEPS):
        if waypoint_idx >= len(waypoints):
            break

        target = waypoints[waypoint_idx]
        pos = obs[:2]
        delta = target - pos
        distance = np.linalg.norm(delta)

        if not waypoint_reached and distance <= GOAL_TOLERANCE:
            waypoint_reached = True  # latch — ignore distance from here until we move on

        if not waypoint_reached:
            heading = np.arctan2(delta[0], delta[1], dtype=np.float32)
            speed_cmd = np.clip(
                pid_distance.compute(0.0, distance),
                -MAX_SPEED, MAX_SPEED)
            action = (speed_cmd, heading)
        elif wait_counter < wait_steps:
            action = (0.0, heading)
            wait_counter += 1
        else:
            print(f"Waypoint {waypoint_idx} reached!")
            waypoint_idx += 1
            wait_counter = 0
            waypoint_reached = False
            pid_distance.reset()
            continue

        obs, _, terminated, truncated, info = control_env.step(action)
        if RENDER_MODE:
            control_env.render()

    control_env.emergency_stop()

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run simulation")
    args = parser.parse_args(argv)

    with managed_env(args.sim) as control_env:
        control_loop(control_env)

if __name__ == "__main__":
    main()