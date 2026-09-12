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

#============================== GLOBALS =================================

LAB1_SEED = 0
MAX_STEPS = 5000
GOAL_TOLERANCE = 0.05
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

MAX_TURN_RATE = 7.0
MAX_ACCEL = 0.15

#============================== DYNAMICS =================================

def dynamics(state, action):
    """
    action = [speed, heading_cmd] — both speed and heading are now TARGETS.
    The robot ramps toward the target speed, and turns toward the target
    heading at a limited rate, rather than snapping to either instantly.
    """
    x, y, heading, speed = state
    speed_cmd, heading_cmd = action

    # --- Heading: turn toward target heading at a limited rate ---
    heading_error = wrap_angle(heading_cmd - heading)
    max_turn_delta = MAX_TURN_RATE * DT
    heading_new = wrap_angle(heading + np.clip(heading_error, -max_turn_delta, max_turn_delta))

    # --- Speed: ramp toward target speed at a limited rate ---
    max_speed_delta = MAX_ACCEL * DT
    speed_error = speed_cmd - speed
    speed_new = speed + np.clip(speed_error, -max_speed_delta, max_speed_delta)
    speed_new = np.clip(speed_new, 0.0, 1.0)

    x_new = x + speed_new * SIM_SPEED_SCALE * np.sin(heading_new) * DT
    y_new = y + speed_new * SIM_SPEED_SCALE * np.cos(heading_new) * DT

    return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)

## If needed, add the EKF from lab 2 here too, and integrate below.

#============================== EKF =================================

class EKF:
    def __init__(self, dt=DT):
        self.dt = dt

        # [x, y, heading, speed]
        self.state_est = np.zeros(4)

        # Initial covariance
        self.P = np.diag([
            0.00243,   # x variance (m^2)
            0.00235,   # y variance (m^2)
            0.0025,    # heading variance (rad)^2
            0.0025     # speed variance (m/s)^2 (pretty sure on the unit)
        ])

        # Process noise covariance
        self.Q = 4*np.diag([
            1e-3,   # x model noise
            1e-3,   # y model noise
            1e-2,   # heading model noise
            2.5e-5  # speed model noise
        ])

        # Measurement noise covariance
        # self.R = np.diag([2.43124e-2,2.34517e-2])
        self.R = 4*np.array([
            [2.43642832e-02, -8.96859747e-04],
            [-8.96859747e-04, 3.46949137e-02]
        ])

        self.nis = None
        self._initialised = False

    def numerical_jacob(self, state, action, eps=1e-6):
        """
        Numerically calculate the Jacobian of dynamics() with respect
        to the state.
        F[i,j] = d f_i / d x_j
        State:
            [x, y, heading, speed]
        Action:
            [speed_cmd, heading_cmd]
        """

        state = np.asarray(state, dtype=float)
        action = np.asarray(action, dtype=float)

        n = len(state)
        F = np.zeros((n, n))

        for i in range(n):
            state_plus = state.copy()
            state_minus = state.copy()

            state_plus[i] += eps
            state_minus[i] -= eps

            f_plus = dynamics(state_plus.astype(np.float32),action.astype(np.float32))
            f_minus = dynamics(state_minus.astype(np.float32),action.astype(np.float32))

            F[:, i] = (f_plus - f_minus) / (2.0 * eps)

        return F

    def predict(self, action):
        """
        Predict the next state and covariance given a control action.
        Action:
            [speed_cmd, heading_cmd]
        State:
            [x, y, heading, speed]
        """

        action = np.asarray(action, dtype=np.float32)

        # Calculate Jacobian of the ACTUAL dynamics model.
        F = self.numerical_jacob(self.state_est,action)

        # Predict state using the actual dynamics model.
        self.state_est = np.asarray(dynamics(self.state_est.astype(np.float32),action),dtype=float)

        # Predict covariance.
        self.P = F @ self.P @ F.T + self.Q

        # Keep covariance numerically symmetric.
        self.P = 0.5 * (self.P + self.P.T)

        return self.state_est, self.P

    def update(self, measurement):
        """
        Correct the state estimate using position measurements.

        Measurement:
            [x, y, heading, speed]

        Only x and y are used by the EKF measurement model.
        """

        measurement = np.asarray(measurement, dtype=float)

        # Only position is measured.
        z = measurement[:2]

        # Initialise position from first measurement.
        if not self._initialised:
            self.state_est[:2] = z

            # Initial position uncertainty is the measurement uncertainty.
            self.P[:2, :2] = self.R.copy()
            self._initialised = True
            return self.state_est, self.P

        # Measurement model:
        # z = [x, y]
        H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])

        # Innovation.
        innov = z - self.state_est[:2]

        # Innovation covariance.
        S = H @ self.P @ H.T + self.R

        # Kalman gain.
        K = self.P @ H.T @ np.linalg.inv(S)

        # NIS for tuning.
        self.nis = float(innov @ np.linalg.inv(S) @ innov)

        # State correction.
        self.state_est = self.state_est + K @ innov

        # Wrap heading.
        self.state_est[2] = wrap_angle(self.state_est[2])

        # Josepi form covariance update.
        I_KH = np.eye(4) - K @ H
        self.P = (I_KH @ self.P @ I_KH.T + K @ self.R @ K.T)
        # Keep covariance numerically symmetric.
        self.P = 0.5 * (self.P + self.P.T)

        return self.state_est, self.P

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
        obs_noise_std_pos=0.0,
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

#============================== MAIN CONTROL LOOP =================================

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

    # --- Waypoint-following state (ported from lab1's control_loop) ---
    waypoint_idx = 0
    wait_counter = 0
    waypoint_reached = False  # latch: once True, ignore distance until we advance
    heading = obs[2]

    for _ in range(MAX_STEPS):
        if waypoint_idx >= len(waypoints):
            break

        target = waypoints[waypoint_idx]
        pos = obs[:2]
        delta = target - pos
        distance = np.linalg.norm(delta)

        # Also check the actual goal directly, since the final grid waypoint
        # can sit slightly off control_env.goal_pos after quantization.
        goal_delta = obs[:2] - np.asarray(control_env.goal_pos, dtype=np.float32)
        if np.dot(goal_delta, goal_delta) < control_env.goal_tolerance ** 2:
            break  # Goal reached

        if not waypoint_reached and distance <= GOAL_TOLERANCE:
            waypoint_reached = True  # latch — ignore distance from here until we move on

        if not waypoint_reached:
            action = controller.compute_action(obs, target)
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