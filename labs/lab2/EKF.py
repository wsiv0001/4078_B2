"""
M2-3 Check-in: Extended Kalman Filter
Student ID: 34951954, Group B2

State:
    [x, y, heading, speed]

Model:
    The robot is modelled using a unicycle modl. 
    The commanded speed is converted from command units to physical velocity 
    The commanded heading is converted into an angular velocity command 
    and follows rate-limited rotational dynamics.

    Position is then propagated using the resulting translational
    velocity and heading. Model parameters were determined from
    real robot observations where possible and are intended to
    represent the nominal behaviour of the Sphero BOLT rather than
    minimise trajectory error for a single task. These parameters were 
    then verified with the actions and GT from M1-2 Submission to verify
    acceptable RMSE for each set of actions.

EKF:
    The Extended Kalman Filter predicts the state using the robot
    dynamics and propagates uncertainty using the corresponding
    state-transition Jacobian (numerically calculated). 
    Position measurements from the robot are incorporated during 
    the measurement update.

    Measurement covariance R was estimated experimentally from
    stationary real-robot observations. Process covariance Q
    represents uncertainty in the robot dynamics and unmodelled
    motion.

The dynamics function is pure: the returned state depends only on
the supplied state and action, so repeated calls with identical
inputs produce identical outputs.
"""

import numpy as np

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

def dynamics(state, action):
    x, y, heading, speed = state
    speed_cmd, heading_cmd = action

    #Heading: unchanged
    
    heading_error = wrap_angle(heading_cmd - heading)
    max_turn_delta = MAX_TURN_RATE * DT
    heading_new = wrap_angle(heading + np.clip(heading_error, -max_turn_delta, max_turn_delta))
    # heading_new = wrap_angle(heading+heading_error)

    # Speed: ramp toward target speed, still in "commanded" units
    max_speed_delta = MAX_ACCEL * DT
    speed_error = speed_cmd - speed
    speed_new = speed + np.clip(speed_error, -max_speed_delta, max_speed_delta)
    # speed_new = speed + speed_error
    speed_new = np.clip(speed_new, 0.0, COMMANDED_MAX_SPEED)

    # Position: this is the only place the real-world scale enters
    x_new = x + speed_new * SIM_SPEED_SCALE * np.sin(heading_new) * DT
    y_new = y + speed_new * SIM_SPEED_SCALE * np.cos(heading_new) * DT

    return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)


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