"""
Dynamics calibration script for the Sphero BOLT.

WHY: dynamics.py's MAX_ACCEL / MAX_TURN_RATE / SPEED_TO_MPS were hand-tuned
guesses. This script drives the real robot through a battery of speed-step
and heading-step maneuvers, logs commanded action vs. resulting odometry,
and fits all three constants directly from that data.

IMPORTANT: run this AFTER fixing the obs-noise bug in make_real_env() (i.e.
obs_noise_std_pos / obs_noise_std_vel should be ~1e-6 / 0, not the 0.05
defaults) - otherwise the fits below will absorb that injected noise instead
of real dynamics.

USAGE
    # 1. Collect data on the real robot (robot must be free to drive ~2-3
    #    metres in whatever direction it's pointed - clear the area).
    python calibrate_dynamics.py --collect --out dyn_calib.csv

    # 2. Fit constants from the collected log (no hardware needed for this
    #    step - safe to run on your laptop after copying the CSV off).
    python calibrate_dynamics.py --analyze --in dyn_calib.csv

You can also run --collect and --analyze in the same invocation.
"""

import argparse
import csv
import time

import numpy as np

DT = 0.1
VELOCITY_LIMIT = 0.15  # must match src.dynamics.VELOCITY_LIMIT / Robot's vel_limit

# ---------------------------------------------------------------------------
# Test battery parameters - tweak these if your robot needs more/less time
# to visibly saturate. Err on the side of MORE hold steps: the analysis
# throws away samples near the target and near the start of each hold, so
# extra steps just give it more clean data to work with, they don't hurt.
# ---------------------------------------------------------------------------

SPEED_LEVELS = [0.01, 0.05, 0.08, 0.11, 0.15]     # speed_cmd values to sweep (units: same as VELOCITY_LIMIT)
STEPS_PER_SPEED_HOLD = 12                   # ~10s at DT=0.1
STEPS_PER_SPEED_ZERO = 30                   # let it fully decelerate between trials
TRIALS_PER_SPEED = 1

HEADING_STEPS_DEG = [30, 90, 150, -30, -90, -150]
STEPS_PER_HEADING_HOLD = 30                  # ~3s - MAX_TURN_RATE=7 rad/s means even
                                              # a 150 deg turn should complete in <0.5s
TRIALS_PER_HEADING = 2


# ---------------------------------------------------------------------------
# Data collection (hardware imports are local to this function so --analyze
# works on a machine with no Sphero libraries installed)
# ---------------------------------------------------------------------------

def run_collection(out_path):
    from contextlib import ExitStack
    from sphero_env.robot.connect import scan_and_connect
    from sphero_unsw.sphero_edu import SpheroEduAPI
    from sphero_env.robot.robot import Robot

    stack = ExitStack()
    selected_toy, _ = scan_and_connect()
    print(f"Selected: {selected_toy.name}")
    api = stack.enter_context(SpheroEduAPI(selected_toy))

    robot = Robot(
        api=api,
        dt=DT,
        max_steps=100_000,
        vel_limit=VELOCITY_LIMIT,
        world_width=5.0,
        world_height=5.0,
        goal_pos=(0.0, 0.0),
        goal_tolerance=0.03,
        # Real sensor noise, not the class defaults - see the docstring above.
        obs_noise_std_pos=0,
        obs_noise_std_vel=0.0,
        render_mode=None,
    )

    rows = []
    step_idx = 0

    def hold(speed_cmd, heading_cmd, n_steps, phase):
        nonlocal step_idx
        for _ in range(n_steps):
            _, _, _, _, info = robot.step(np.array([speed_cmd, heading_cmd], dtype=np.float32))
            st = info["state_odom"]
            rows.append({
                "step": step_idx, "phase": phase, "t": step_idx * DT,
                "speed_cmd": speed_cmd, "heading_cmd": heading_cmd,
                "x": float(st[0]), "y": float(st[1]),
                "heading": float(st[2]), "speed": float(st[3]),
            })
            step_idx += 1
            time.sleep(DT)

    try:
        robot.reset(seed=0)

        print("=== Speed sweep (out-and-back within corridor) ===")
        for trial in range(TRIALS_PER_SPEED):
            for spd in SPEED_LEVELS:
                print(f"  trial {trial}: speed_cmd={spd}")
                tag = f"{spd}_t{trial}"

                # Out: drive away from the start point
                hold(spd, 0.0, STEPS_PER_SPEED_HOLD, f"speed_hold_out_{tag}")
                hold(0.0, 0.0, STEPS_PER_SPEED_ZERO, f"speed_zero_out_{tag}")

                # Turn 180° on the spot (speed_cmd stays 0 the whole time)
                hold(0.0, np.pi, STEPS_PER_HEADING_HOLD, f"heading_step_return_{tag}")

                # Back: drive the same commanded speed back toward you
                hold(spd, np.pi, STEPS_PER_SPEED_HOLD, f"speed_hold_return_{tag}")
                hold(0.0, np.pi, STEPS_PER_SPEED_ZERO, f"speed_zero_return_{tag}")

                # Turn back to face outward, ready for the next trial
                hold(0.0, 0.0, STEPS_PER_HEADING_HOLD, f"heading_step_reset_{tag}")

        print("=== Heading sweep (stationary, from heading=0) ===")
        for trial in range(TRIALS_PER_HEADING):
            for deg in HEADING_STEPS_DEG:
                rad = float(np.deg2rad(deg))
                print(f"  trial {trial}: heading_cmd={deg} deg")
                hold(0.0, 0.0, 10, "heading_reset")  # let it settle back near 0 first
                hold(0.0, rad, STEPS_PER_HEADING_HOLD, f"heading_step_{deg}")
    finally:
        robot.close()
        stack.close()

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_path}")


# ---------------------------------------------------------------------------
# Fitting logic (pure data-processing, no hardware dependency, unit-tested
# against a synthetic version of dynamics.py before shipping)
# ---------------------------------------------------------------------------

def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def fit_max_accel(df, near_target_thresh=0.02, min_delta=1e-4):
    """
    `speed` in the odometry is just an echo of the last commanded value, so
    there's no ramp to see in it directly. Reconstruct real speed from
    consecutive (x, y) position deltas instead, and fit the ramp from that.
    """
    estimates = []
    for phase, group in df.groupby("phase"):
        group = group.sort_values("step").reset_index(drop=True)
        if len(group) < 5:
            continue  # too short to safely trim edge samples

        target = group["speed_cmd"].iloc[0]
        x, y, t = group["x"].values, group["y"].values, group["t"].values

        dt = np.diff(t)
        dt[dt <= 0] = DT  # guard against duplicate/zero timestamps
        est_speed = np.hypot(np.diff(x), np.diff(y)) / dt

        for i in range(1, len(est_speed) - 2):
            delta = est_speed[i + 1] - est_speed[i]
            still_far = abs(target - est_speed[i]) > near_target_thresh
            if still_far and abs(delta) > min_delta:
                estimates.append(abs(delta) / DT)

    return float(np.median(estimates)) if estimates else None

def fit_max_turn_rate(df, near_target_thresh_deg=5.0, min_delta=1e-4):
    """Same idea as fit_max_accel but for heading, using wrapped differences."""
    estimates = []
    for phase, group in df.groupby("phase"):
        if not str(phase).startswith("heading_step"):
            continue
        group = group.sort_values("step").reset_index(drop=True)
        target = group["heading_cmd"].iloc[0]
        heading = group["heading"].values
        for i in range(1, len(heading) - 2):
            err_now = wrap_angle(target - heading[i])
            delta = wrap_angle(heading[i + 1] - heading[i])
            still_far = abs(err_now) > np.deg2rad(near_target_thresh_deg)
            if still_far and abs(delta) > min_delta:
                estimates.append(abs(delta) / DT)
    return float(np.median(estimates)) if estimates else None


def fit_speed_to_mps(df, tail_frac=2.0 / 3.0, band_frac=0.05):
    """
    Once speed has converged near speed_cmd (last third of each hold, within
    5% of target), real displacement/DT gives true m/s. Regressing that
    against the steady-state 'speed' state (forced through the origin, since
    speed=0 must give velocity=0) gives a single scale factor pooled across
    every speed level and trial.
    """
    vel_points = []
    for phase, group in df.groupby("phase"):
        if not str(phase).startswith("speed_hold"):
            continue
        group = group.sort_values("step").reset_index(drop=True)
        target = group["speed_cmd"].iloc[0]
        if target == 0:
            continue
        n = len(group)
        tail = group.iloc[int(n * tail_frac):]
        tail = tail[np.abs(tail["speed"] - target) < band_frac * max(target, 1e-6)]
        xs, ys, speeds = tail["x"].values, tail["y"].values, tail["speed"].values
        for i in range(len(tail) - 1):
            dist = np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])
            vel_points.append((speeds[i], dist / DT))
    if not vel_points:
        return None
    s = np.array([p[0] for p in vel_points])
    v = np.array([p[1] for p in vel_points])
    return float(np.sum(s * v) / np.sum(s ** 2))


def run_analysis(in_path):
    import pandas as pd
    df = pd.read_csv(in_path)

    current = {"MAX_ACCEL": 0.024, "MAX_TURN_RATE": 7.0, "SPEED_TO_MPS": 25}
    fitted = {
        "MAX_ACCEL": fit_max_accel(df),
        "MAX_TURN_RATE": fit_max_turn_rate(df),
        "SPEED_TO_MPS": fit_speed_to_mps(df),
    }

    print("=" * 50)
    print(f"{'parameter':<16}{'current':>12}{'fitted':>12}")
    print("=" * 50)
    for k in current:
        f = fitted[k]
        if f is None:
            print(f"{k:<16}{current[k]:>12.5f}{'  not enough data':>18}")
        else:
            print(f"{k:<16}{current[k]:>12.5f}{f:>12.5f}")

    print("\nSuggested dynamics.py constants:")
    for k, v in fitted.items():
        if v is not None:
            print(f"    {k} = {v:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true", help="run the test battery on real hardware")
    parser.add_argument("--analyze", action="store_true", help="fit constants from a collected CSV")
    parser.add_argument("--out", default="dyn_calib.csv", help="CSV path to write when collecting")
    parser.add_argument("--in", dest="in_path", default="dyn_calib.csv", help="CSV path to read when analyzing")
    args = parser.parse_args()

    if not args.collect and not args.analyze:
        parser.error("pass --collect and/or --analyze")

    if args.collect:
        run_collection(args.out)
    if args.analyze:
        run_analysis(args.in_path)