import numpy as np
from src.shared_functions import wrap_angle


class PID:
    """Simple PID controller.

    output = kp * error + ki * integral(error) + kd * derivative(error)
    """

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, dt=0.1, output_limits=None, is_angle=False):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self.output_limits = output_limits  # (min, max) or None
        self.is_angle = is_angle            # True for heading control, wraps error to [-pi, pi]

        self.integral = 0.0
        self.prev_error = None

    def compute(self, current_value, target_value):
        error = target_value - current_value
        if self.is_angle:
            error = wrap_angle(error)

        self.integral += error * self.dt

        derivative = 0.0 if self.prev_error is None else (error - self.prev_error) / self.dt
        self.prev_error = error

        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        if self.output_limits is not None:
            output = np.clip(output, self.output_limits[0], self.output_limits[1])

        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = None


# --- Controllers used by the waypoint follower -----------------------------

MAX_SPEED = 0.15  # m/s, matches Robot's vel_limit

pid_distance = PID(
    kp=0.06,
    ki=0.002,
    kd=0.07,
    dt=0.1,
    output_limits=(0.0, MAX_SPEED),
    is_angle=False,
)

pid_heading = PID(
    kp=1.0,
    ki=0.0,
    kd=0.0,
    dt=0.1,
    output_limits=(-np.pi, np.pi),
    is_angle=True,
)