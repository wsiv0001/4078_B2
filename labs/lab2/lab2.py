import time
import argparse
import numpy as np
import pygame
from contextlib import ExitStack, contextmanager
from labs.lab2.EKF import *

from sphero_unsw.sphero_edu import SpheroEduAPI
from sphero_env.robot.connect import scan_and_connect
from sphero_env.robot.robot import Robot
from sphero_env.envs import SpheroEnv

# group_b2 imports - same as lab1, needed for get_log_path, pid_distance, MAX_SPEED
from src.dynamics import *
from src.pid_control import *
from src.shared_functions import *

# ----------------------- Global constants for lab2.py ----------------------- #
LAB2_SEED = 0
DT = 0.1

SESSION_NAME = "session1_SB-DAE7"  # rename this per run - it becomes the folder under logs/lab2_logs/

# Waypoints for --mode waypoints. Reuses the goal used by teleop's env (0.5, 0.5),
# but you can add more points here if you want a multi-leg PID run.
WAYPOINTS = np.array([
    [0.0, 0.5],
    [0.5, 0.5],
    [0.5, 0.0],
    [0.0, 0.0]
], dtype=np.float32)

GOAL_TOLERANCE = 0.1
WAIT_STEPS = 20         # steps to sit at each waypoint (DT=0.1 -> 2s)
MAX_TOTAL_STEPS = 2000  # safety cap


### Control loop to handle the action and update the environments,
# edit this to include the EKF prediction and update steps, and to visualize the belief state in the simulator.
def teleop_control_loop(env, ekf, robot_env=None, action=None, moving=False):
    """
    Teleop control loop: apply the keyboard-driven action, step sim (+ robot if connected),
    update the EKF, and visualize the belief state.
    """
    # Step simulator directly and update its logging/visualization state.
    sim_obs, _, _, _, sim_info = env.step(action)

    # Control robot directly (if connected)
    if robot_env is not None:
        robot_obs, _, _, _, robot_info = robot_env.step(action)
        # print(f"Robot state: {info['state_odom']}, Collision: {info['collision']}, Acceleration: {info['acceleration']}, Orientation: {info['orientation']}, Gyro: {info['gyroscope']}, Velocity: {info['velocity']}")

    if robot_env is not None and not moving:
        robot_env.emergency_stop()

     # Call the EKF to predict and update the state estimate based on the action and observation

    ekf.predict(action)  # Predict the next state using the EKF
    if robot_env is not None:
        ekf.update(robot_obs)  # Update the EKF with the odometry measurement - you may want to also include other measurements if available from the info dictionary (e.g., IMU, gyro, etc.)
    else:
        ekf.update(sim_obs)  # Update the EKF with the simulator observation

    env.vis.set_belief(ekf.state_est, ekf.P)  # Update the simulator with the EKF state estimate to visualise the belief state

    env.render()


def waypoint_control_loop(env, ekf, robot_env=None, waypoints=WAYPOINTS,
                           wait_steps=WAIT_STEPS, goal_tolerance=GOAL_TOLERANCE,
                           max_total_steps=MAX_TOTAL_STEPS):
    """
    PID-controlled waypoint following, structured like lab1's control_loop, but still
    running the EKF predict/update + belief visualization each step like lab2's teleop
    loop does. Drives the real robot alongside the sim if robot_env is connected.

    Press SPACE (or close the pygame window) to abort early - useful since this can
    drive the real robot unattended.
    """
    initial_heading = np.arctan2(waypoints[0][0], waypoints[0][1], dtype=np.float32)

    obs, _ = env.reset(seed=LAB2_SEED, options={"initial_heading": initial_heading})
    if robot_env is not None:
        robot_env.reset(seed=LAB2_SEED, options={"initial_heading": initial_heading})

    waypoint_idx = 0
    wait_counter = 0
    waypoint_reached = False
    heading = initial_heading

    for _ in range(max_total_steps):
        # Allow aborting mid-run (esp. important when driving the real robot).
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in (pygame.K_SPACE, pygame.K_q)
            ):
                print("Waypoint run aborted.")
                if robot_env is not None:
                    robot_env.emergency_stop()
                return

        if waypoint_idx >= len(waypoints):
            break

        target = waypoints[waypoint_idx]
        # Drive off the EKF belief rather than raw obs, since that's the whole point of lab2.
        pos = np.asarray(ekf.state_est[:2])
        delta = target - pos
        distance = np.linalg.norm(delta)

        if not waypoint_reached and distance <= goal_tolerance:
            waypoint_reached = True  # latch - ignore distance from here until we move on

        if not waypoint_reached:
            heading = np.arctan2(delta[0], delta[1], dtype=np.float32)
            speed_cmd = np.clip(
                pid_distance.compute(0.0, distance),
                -MAX_SPEED, MAX_SPEED)
            action = np.array([speed_cmd, heading], dtype=np.float32)
        elif wait_counter < wait_steps:
            action = np.array([0.0, heading], dtype=np.float32)
            wait_counter += 1
        else:
            print(f"Waypoint {waypoint_idx} reached!")
            waypoint_idx += 1
            wait_counter = 0
            waypoint_reached = False
            pid_distance.reset()
            continue

        sim_obs, _, _, _, sim_info = env.step(action)
        if robot_env is not None:
            robot_obs, _, _, _, robot_info = robot_env.step(action)

        ekf.predict(action)
        ekf.update(robot_obs if robot_env is not None else sim_obs)

        env.vis.set_belief(ekf.state_est, ekf.P)
        env.render()

    if robot_env is not None:
        robot_env.emergency_stop()


# This function creates a new simulator environment
def make_sim_env():
    return SpheroEnv(
        dt=0.1,
        max_steps=5000,
        vel_limit=0.15,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        occupancy_grid=None,
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
        dt=0.1,
        max_steps=5000,
        vel_limit=0.15,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.5, 0.5),
        goal_tolerance=0.1,
        render_mode="human",
        window_size=(800, 800),
    )

@contextmanager
def managed_sim_env():
    env = make_sim_env()
    # Same logging helper as lab1, so sim logs land under logs/lab2_logs/<SESSION_NAME>/
    sim_log_path = get_log_path(SESSION_NAME, is_real=False)
    env.set_log_path(sim_log_path)
    env.reset()
    env.start_logging()
    try:
        yield env
    finally:
        env.stop_logging()
        env.close()

@contextmanager
def managed_robot_env():
    # Scan for and connect to a Sphero robot, then set up logging for the robot environment
    selected_toy, _ = scan_and_connect()
    print(f"Selected: {selected_toy.name}")

    with SpheroEduAPI(selected_toy) as api:
        robot_env = make_real_env(api)
        # Same logging helper as lab1, so robot logs land under logs/lab2_logs/<SESSION_NAME>/
        real_log_path = get_log_path(SESSION_NAME, is_real=True)
        robot_env.set_log_path(real_log_path)
        robot_env.reset()
        robot_env.start_logging()
        try:
            yield robot_env
        finally:
            robot_env.stop_logging()
            robot_env.emergency_stop()
            robot_env.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperate or waypoint-drive Sphero with optional simulator-only mode")
    parser.add_argument("--sim", action="store_true", help="Run simulator-only mode (no robot connection)")
    parser.add_argument("--mode", choices=["teleop", "waypoints"], default="teleop",
                         help="teleop: drive with keyboard (default). waypoints: run the lab1-style PID waypoint controller.")
    return parser.parse_args()

def main(sim_only=False, mode="teleop"):
    """Main function for teleoperation or waypoint-driving of Sphero robot with simulator."""
    with ExitStack() as stack:
        env = stack.enter_context(managed_sim_env())
        robot_env = stack.enter_context(managed_robot_env()) if not sim_only else None

        env.render()

        ekf = EKF(dt=0.1)  # Initialize the EKF for state estimation

        stack.callback(pygame.quit)
        stack.callback(lambda: print("Stopped."))

        mode_label = "Simulator only" if sim_only else "Robot + Simulator"

        if mode == "waypoints":
            print(f"\nWaypoint run ready ({mode_label}). Driving through {len(WAYPOINTS)} waypoint(s).")
            print("  SPACE / Q = abort\n")
            waypoint_control_loop(env, ekf=ekf, robot_env=robot_env)
            print("Waypoint run complete.")
            return

        print(f"\nTele-op ready ({mode_label}):")
        print("  W       = move forward")
        print("  A/D     = turn left/right")
        print("  S       = move backward")
        print("  SPACE   = emergency stop")
        print("  +/-     = speed")
        print("  Q       = quit\n")

        speed_norm = 0.25  # normalized speed [0, 1]
        current_heading = 0.0 if robot_env is None else float(robot_env.get_odom_state()[2])

        running = True
        moving = False
        last_speed_change_time = 0.0

        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False

                    elif event.key == pygame.K_SPACE:
                        if robot_env is not None:
                            robot_env.emergency_stop()
                        moving = False
                        print("Emergency stop")

                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                        now = time.time()
                        if now - last_speed_change_time > 0.08:
                            speed_norm = min(1.0, speed_norm + 0.1)
                            print(f"Speed: {speed_norm:.1f}")
                            last_speed_change_time = now

                    elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                        now = time.time()
                        if now - last_speed_change_time > 0.08:
                            speed_norm = max(0.0, speed_norm - 0.1)
                            print(f"Speed: {speed_norm:.1f}")
                            last_speed_change_time = now

            # Handle continuous key presses
            pressed = pygame.key.get_pressed()

            v_cmd = 0.0
            if pressed[pygame.K_w]:
                v_cmd = speed_norm * 1.0  # forward
                moving = True
            elif pressed[pygame.K_s]:
                v_cmd = -speed_norm * 1.0  # backward
                moving = True
            else:
                moving = False

            # Handle turning
            turn_rate = 0.2  # radians per frame
            if pressed[pygame.K_a]:
                current_heading -= turn_rate  # turn left

            if pressed[pygame.K_d]:
                current_heading += turn_rate  # turn right

            # Normalize heading to [-pi, pi]
            current_heading = (current_heading + np.pi) % (2 * np.pi) - np.pi

            # Create action [v, theta]
            action = np.array([v_cmd, current_heading], dtype=np.float32)

            teleop_control_loop(env, ekf=ekf, robot_env=robot_env, action=action, moving=moving)

            if robot_env is not None:
                time.sleep(0.01)  # Small delay to prevent busy-waiting
            else:
                time.sleep(0.1)  # Small delay to prevent busy-waiting in sim-only mode

if __name__ == "__main__":
    args = parse_args()
    main(sim_only=args.sim, mode=args.mode)