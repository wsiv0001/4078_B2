# Import necessary libraries
from sphero_env.robot.connect import scan_and_connect
from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from labs.lab3.Planner import *
from src.pid_control import *
from src.shared_functions import *  # wrap_angle, get_log_path
from sphero_env.envs.custom_maze_full import build_occupancy_grid
from src.dynamics import *
from src.EKF import *

from contextlib import ExitStack, contextmanager

#============================== GLOBALS =================================

LAB1_SEED = 0
MAX_STEPS = 5000
GOAL_TOLERANCE = 0.03
WAYPOINT_PAUSE_STEPS = 10  # steps to pause/settle at each waypoint before moving on
map = build_occupancy_grid()

# Rename this per run - becomes the folder under logs/lab3_logs/
SESSION_NAME = "session4_SB-DAE7_EKF - with noise"

# World-frame position of the maze's designated starting plate
# (matches START_CELL in sphero_env.envs.custom_maze_full).
KNOWN_START_WORLD = np.array([-0.5, -0.5])

#============================== CONTROLLER AND ENVIRONMENTS =================================

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
        obs_noise_std_pos=1.1e-4,
        process_noise_std_speed=0.00,
        process_noise_std_heading=0.0,
        obs_noise_std_vel=0.0,
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
        obs_noise_std_pos=0.0,
        obs_noise_std_vel=0.0,
        render_mode="human",
        window_size=(800, 800),
    )

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

#============================== MAIN CONTROL LOOP =================================

def control_loop(control_env):

    obs, _ = control_env.reset(seed=LAB1_SEED)

    # Keep a copy of the very first reading in the robot's own raw LOCAL
    # frame (before any map-frame offset is applied) - this is the common
    # seed for the odometry trace and the open-loop dynamics rollout below,
    # so all debug traces start from the exact same point.
    local_obs0 = obs.copy()

    if isinstance(control_env, SpheroEnv):
        # Sim: force a known start position - SpheroEnv respects this override,
        # so obs is already in the map's world frame once we read it back.
        control_env.state_true[0:3] = np.array([-0.5, -0.5, 0.0])
        control_env.state_odom[0:3] = np.array([-0.5, -0.5, 0.0])
        obs = control_env.state_true.copy()
        local_obs0 = obs.copy()
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

    def to_local_frame(state):
        """Inverse of to_map_frame, position-only - use this whenever handing
        a map-frame state to something (e.g. the visualiser) that expects the
        robot's own raw local frame instead."""
        state = state.copy()
        state[:2] = state[:2] - frame_offset
        return state

    controller = Controller(dt=control_env.dt)
    planner = Planner(map=map, dt=control_env.dt)

    # --- EKF setup ---
    # obs at this point is already in the map frame (see above), so the EKF
    # runs entirely in the map frame too - no extra transform needed.
    ekf = EKF(dt=control_env.dt)
    ekf.update(obs)  # seed the filter's position from the first measurement
    ekf.state_est[2] = obs[2]  # seed heading/speed directly (not estimated on step 1)
    ekf.state_est[3] = obs[3]
    est_state = ekf.state_est.copy()

    if hasattr(control_env, "vis") and control_env.vis is not None:
        # Visualiser draws ground-truth/odometry in the robot's raw local
        # frame, not the shifted map frame - convert before displaying so
        # the magenta belief lines up with those traces instead of sitting
        # offset by frame_offset.
        control_env.vis.set_belief(to_local_frame(ekf.state_est), ekf.P)

    # Planner.plan returns absolute waypoints already in the map's frame.
    # Plan from the filtered estimate, same as control will use it.
    waypoints = planner.plan(est_state, control_env.goal_pos)

    planner.debug_print(waypoints)
    # Save instead of plt.show(): avoids blocking on an interactive Tk
    # window (which also caused stray "main thread is not in main loop"
    # warnings on exit) and the "Not Responding" freeze seen earlier.
    run_type = "sim" if isinstance(control_env, SpheroEnv) else "real"
    plot_dir = os.path.join("logs", "lab3_logs", SESSION_NAME, run_type)
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "planned_path.png")
    planner.debug_plot(est_state[:2], control_env.goal_pos, waypoints, save_path=plot_path)

    # --- Waypoint-following state (ported from lab1's control_loop) ---
    waypoint_idx = -1  # -1 = boot-settle phase, before waypoint following begins
    wait_counter = 0
    waypoint_reached = False  # latch: once True, ignore distance until we advance
    heading = est_state[2]

    # Number of steps to hold zero-speed for during the boot-settle phase,
    # replacing the earlier time.sleep(3) in managed_env. Doing this as
    # actual zero-speed control_env.step() calls (rather than an external
    # sleep) actively keeps pinging the robot while its onboard dynamics
    # finishes booting, and - just as importantly - it means the EKF, the
    # open-loop dynamics rollout, and the odometry trace are all logged
    # through this phase too, so every trace starts from the exact same
    # synchronized point once real waypoint-following begins. A plain sleep
    # outside the loop left the robot possibly still not fully responsive
    # once the first real commands went out, which is what caused the
    # dynamics-prediction trace to diverge early in the last run.
    BOOT_WAIT_SECONDS = 2.0
    boot_wait_steps = max(1, int(round(BOOT_WAIT_SECONDS / control_env.dt)))

    # --- Debug traces, all kept in the robot's raw LOCAL frame ---
    # - odom_trace:  the robot's own position estimate. In sim this is
    #   control_env.state_odom; on the real robot, raw_obs itself already
    #   IS this signal - the Sphero API reports its own dead-reckoned
    #   position, not ground truth, so no separate "odometry" attribute
    #   exists or is needed on the Robot class.
    # - gt_trace: ground truth, only available in sim (no equivalent on
    #   real hardware).
    # - dyn_trace: a pure OPEN-LOOP rollout of the dynamics() model, seeded
    #   from local_obs0 and advanced only using dynamics(dyn_state, action)
    #   each step - never corrected by any measurement. This isolates what
    #   the model itself predicts, independent of what the robot/EKF
    #   actually reports, so any mismatch against odom_trace is purely a
    #   dynamics-model error.
    # - ekf_trace: the EKF's filtered estimate, converted back to local
    #   frame for direct comparison with the traces above.
    gt_trace = []
    odom_trace = [local_obs0[:2].copy()]
    # dynamics() unpacks its state as exactly (x, y, heading, speed) - slice
    # explicitly rather than copying local_obs0 wholesale, since the real
    # robot's observation vector can carry extra trailing fields (e.g. raw
    # velocity components, battery, timestamp) beyond just these 4, which
    # would otherwise break the unpack inside dynamics().
    dyn_state = np.array(local_obs0[:4], dtype=np.float32)
    dyn_trace = [dyn_state[:2].copy()]
    ekf_trace = [to_local_frame(est_state)[:2].copy()]

    for _ in range(MAX_STEPS):
        if waypoint_idx >= len(waypoints):
            break

        if waypoint_idx == -1:
            # Boot-settle phase: hold zero speed for boot_wait_steps steps
            # before starting real waypoint-following. Deliberately never
            # touches waypoints[] or the distance/goal checks below - only
            # a plain step count, so it can't accidentally trip
            # waypoint_reached or the goal check against a stale target.
            if wait_counter < boot_wait_steps:
                action = (0.0, heading)
                wait_counter += 1
            else:
                print("Boot-settle complete - starting waypoint following.")
                waypoint_idx = 0
                wait_counter = 0
                waypoint_reached = False
                pid_distance.reset()
                continue
        else:
            target = waypoints[waypoint_idx]
            pos = est_state[:2]
            delta = target - pos
            distance = np.linalg.norm(delta)

            # Also check the actual goal directly, using the filtered estimate,
            # since the final grid waypoint can sit slightly off
            # control_env.goal_pos after quantization.
            goal_delta = est_state[:2] - np.asarray(control_env.goal_pos, dtype=np.float32)
            if np.dot(goal_delta, goal_delta) < control_env.goal_tolerance ** 2:
                break  # Goal reached

            if not waypoint_reached and distance <= GOAL_TOLERANCE:
                waypoint_reached = True  # latch — ignore distance from here until we move on

            if not waypoint_reached:
                action = controller.compute_action(est_state, target)
                heading = action[1]
            elif wait_counter < WAYPOINT_PAUSE_STEPS:
                action = (0.0, heading)  # zero speed, hold current heading and settle
                wait_counter += 1
            else:
                print(f"Waypoint {waypoint_idx} reached!")
                waypoint_idx += 1
                wait_counter = 0
                waypoint_reached = False
                pid_distance.reset()
                continue

        raw_obs, _, terminated, truncated, info = control_env.step(action)
        obs = to_map_frame(raw_obs)

        # --- EKF predict/update ---
        # Predict forward using the action just applied, then correct with
        # the (noisy) position measurement just received. The controller
        # and waypoint logic above always work off est_state, never the
        # raw obs, so sensor noise near waypoints gets smoothed out instead
        # of driving the heading command directly.
        ekf.predict(np.array(action, dtype=np.float32))
        ekf.update(obs)
        est_state = ekf.state_est.copy()

        # Visualise the belief (magenta mean + uncertainty ellipse), same as
        # lab2's teleop/waypoint loops. Convert back to the robot's raw local
        # frame first, since that's what the visualiser's other traces
        # (ground truth, odometry) are drawn in - the real robot env has no
        # visualiser, so this only fires for the sim env.
        if hasattr(control_env, "vis") and control_env.vis is not None:
            control_env.vis.set_belief(to_local_frame(ekf.state_est), ekf.P)

        # --- Advance the open-loop dynamics-only rollout with this same
        # action - completely independent of raw_obs/EKF, so it only ever
        # reflects the model's own assumptions.
        dyn_state = dynamics(dyn_state, np.array(action, dtype=np.float32))

        # --- Log traces for the post-run comparison plot ---
        if isinstance(control_env, SpheroEnv):
            gt_trace.append(control_env.state_true[:2].copy())
            odom_trace.append(control_env.state_odom[:2].copy())
        else:
            odom_trace.append(raw_obs[:2].copy())
        dyn_trace.append(dyn_state[:2].copy())
        ekf_trace.append(to_local_frame(est_state)[:2].copy())

        control_env.render()

    control_env.emergency_stop()

    # --- Ground truth / odometry / dynamics / EKF comparison plot ---
    # gt vs odom (sim only) shows measurement noise and process error.
    # dyn vs odom shows pure open-loop dynamics-model drift, with no
    # correction from any measurement at all. ekf vs the others shows how
    # much correction the filter is actually applying. All traces are in
    # the local frame.
    gt_arr = np.array(gt_trace)
    odom_arr = np.array(odom_trace)
    dyn_arr = np.array(dyn_trace)
    ekf_arr = np.array(ekf_trace)

    plt.figure()
    if len(gt_arr):
        plt.plot(gt_arr[:, 0], gt_arr[:, 1], label="ground truth", color="green")
    if len(odom_arr):
        plt.plot(odom_arr[:, 0], odom_arr[:, 1], label="odometry (measured)", color="orange")
    if len(dyn_arr):
        plt.plot(dyn_arr[:, 0], dyn_arr[:, 1], label="dynamics prediction (open-loop)", color="blue")
    if len(ekf_arr):
        plt.plot(ekf_arr[:, 0], ekf_arr[:, 1], label="EKF estimate", color="magenta")
    plt.legend()
    plt.gca().set_aspect("equal")
    plt.title("Trajectory comparison: dynamics prediction vs odometry vs EKF")
    plt.xlabel("x (local frame)")
    plt.ylabel("y (local frame)")
    compare_path = os.path.join(plot_dir, "trajectory_compare.png")
    plt.savefig(compare_path)
    plt.close()
    print(f"Saved trajectory comparison plot to {compare_path}")

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run simulation")
    args = parser.parse_args(argv)

    try:
        with managed_env(args.sim) as control_env:
            control_loop(control_env)
    except KeyboardInterrupt:
        # Catches Ctrl+C during setup/teardown too (e.g. scan_and_connect),
        # not just inside control_loop's step loop above.
        print("\nInterrupted - closing connection.")

if __name__ == "__main__":
    main()