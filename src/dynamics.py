# import numpy as np
# from src.shared_functions import *

# dt = 0.1

# def dynamics(state, action):
#     """
#     action = [speed, heading_cmd] — both speed and heading are now TARGETS.
#     The robot ramps toward the target speed, and turns toward the target
#     heading at a limited rate, rather than snapping to either instantly.
#     """
#     x, y, heading, speed = state
#     speed_cmd, heading_cmd = action

#     dt = 0.1

#     # --- Heading: turn toward target heading at a limited rate ---
#     MAX_TURN_RATE = 7.0  # rad/s — tune this for turning feel
#     heading_error = wrap_angle(heading_cmd - heading)
#     max_turn_delta = MAX_TURN_RATE * dt
#     heading_new = wrap_angle(heading + np.clip(heading_error, -max_turn_delta, max_turn_delta))

#     # --- Speed: ramp toward target speed at a limited rate ---
#     MAX_ACCEL = 0.15  # max speed change per second
#     max_speed_delta = MAX_ACCEL * dt
#     speed_error = speed_cmd - speed
#     speed_new = speed + np.clip(speed_error, -max_speed_delta, max_speed_delta)
#     speed_new = np.clip(speed_new, 0.0, 1.0)

#     SPEED_TO_MPS = 35.0

#     x_new = x + speed_new * SPEED_TO_MPS * np.sin(heading_new) * dt
#     y_new = y + speed_new * SPEED_TO_MPS * np.cos(heading_new) * dt

#     return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)
