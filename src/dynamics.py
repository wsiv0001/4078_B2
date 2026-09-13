import numpy as np
from src.shared_functions import *

DT = 0.1
VELOCITY_LIMIT = 0.15

def dynamics(state, action):
    """
    action = [speed, heading_cmd] — both speed and heading are now TARGETS.
    The robot ramps toward the target speed, and turns toward the target
    heading at a limited rate, rather than snapping to either instantly.
    """
    x, y, heading, speed = state
    speed_cmd, heading_cmd = action

    # --- Heading: turn toward target heading at a limited rate ---
    MAX_TURN_RATE = 7.0  # rad/s — tune this for turning feel
    heading_error = wrap_angle(heading_cmd - heading)
    max_turn_delta = MAX_TURN_RATE * DT
    heading_new = wrap_angle(heading + np.clip(heading_error, -max_turn_delta, max_turn_delta))

    # --- Speed: ramp toward target speed at a limited rate ---
    MAX_ACCEL = 0.15  # max speed change per second
    max_speed_delta = MAX_ACCEL * DT
    speed_error = speed_cmd - speed
    speed_new = speed + np.clip(speed_error, -max_speed_delta, max_speed_delta)
    speed_new = np.clip(speed_new, 0.0, 1.0)

    SPEED_TO_MPS = 35.0

    x_new = x + speed_new * SPEED_TO_MPS * np.sin(heading_new) * DT
    y_new = y + speed_new * SPEED_TO_MPS * np.cos(heading_new) * DT

    return np.array([x_new, y_new, heading_new, speed_new], dtype=np.float32)


#=================================== OLD DYNAMICS ====================================

# # Measured/estimated physical parameters
# REAL_TOP_SPEED_MPS = 4
# COMMANDED_MAX_SPEED = VELOCITY_LIMIT
# SIM_SPEED_SCALE = REAL_TOP_SPEED_MPS / COMMANDED_MAX_SPEED

# MAX_TURN_RATE = 100
# MAX_ACCEL = 0.2

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