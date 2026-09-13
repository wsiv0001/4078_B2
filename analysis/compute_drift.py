"""
compute_drift.py

Estimates EKF process-noise (Q) diagonal terms for x, y, heading, and speed
by comparing a REAL robot run against a SIM run of the dynamics model,
logged separately but in the same CSV format:

  est_x,est_y,odom_x,odom_y,gt_x,gt_y,cov_x,cov_y,cov_xy,cov_yx,
  heading_cmd,speed_cmd,heading,speed,setpoint_x,setpoint_y

For each file:
  - est_x/est_y/cov_* are dropped (EKF not yet integrated -> all nan).
  - odom_x/odom_y/heading/speed are treated as that run's actual measured
    state (real robot's odometry for REAL_LOG_PATH, model's own state for
    SIM_LOG_PATH).
  - gt_x/gt_y are ignored entirely here, since they're not populated as an
    independent reference within a single file - the SIM_LOG_PATH file now
    plays that role.

Both logs are assumed to have been driven by the same commanded waypoint
sequence, one row per timestep at the same DT. A sanity check compares
heading_cmd/speed_cmd between the two files and warns if they don't line
up - if they don't, the row-by-row comparison below is meaningless.

Two drift estimates are reported for x, y, heading, speed:

1. CUMULATIVE residual = real - sim, at each aligned timestep. This is how
   far the real robot has diverged from the model by time t. If the two
   trajectories are integrated open-loop from the same start, this
   residual accumulates over time - its variance overestimates the true
   one-step process noise.

2. ONE-STEP DIFFERENCED residual = diff(cumulative residual). If the
   cumulative residual behaves like a random walk (R(t) = R(t-1) + w(t)),
   diff(R) approximates a single noise sample w(t) directly, and its
   variance is a much better estimate of the EKF's per-step Q. This is
   generally the one you want for Q.

Speed is also independently derived from position differences for both
logs: speed(t) ~= hypot(dx, dy) / DT. This gives a model-consistent speed
comparison as a cross-check against the logged `speed` columns. NOTE:
differentiating noisy positions amplifies noise (~sqrt(2)/DT) and is
least reliable during near-stationary holds where true speed ~= 0 but
position noise alone produces a nonzero derived speed.
"""

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
REAL_LOG_PATH = "logs/lab1_logs/session3_SB-DAE7 - SIM TUNING/real/lab1_real_2026-09-14_00-21-28.csv"   # <-- real robot run
SIM_LOG_PATH = "logs/lab1_logs/session3_SB-DAE7 - SIM TUNING/sim/lab1_sim.csv"     # <-- sim/model run, same commands
DT = 0.1                                       # matches lab1.py's DT
CMD_MISMATCH_TOL = 1e-6                        # tolerance for the alignment sanity check
# --------------------------------------------------------------------------- #


def wrap_to_pi(angle):
    """Wrap an angle (or Series/array of angles, in radians) into [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def load_log(path):
    df = pd.read_csv(path)
    ekf_cols = ["est_x", "est_y", "cov_x", "cov_y", "cov_xy", "cov_yx"]
    df = df.drop(columns=[c for c in ekf_cols if c in df.columns])
    needed = ["odom_x", "odom_y", "heading", "speed", "heading_cmd", "speed_cmd"]
    df = df.dropna(subset=[c for c in needed if c in df.columns]).reset_index(drop=True)
    return df


def check_alignment(real_df, sim_df, tol=CMD_MISMATCH_TOL):
    n = min(len(real_df), len(sim_df))
    if len(real_df) != len(sim_df):
        print(f"[warning] logs have different lengths (real={len(real_df)}, sim={len(sim_df)}); "
              f"truncating both to the first {n} rows.")

    real_cmd = real_df[["heading_cmd", "speed_cmd"]].to_numpy()[:n]
    sim_cmd = sim_df[["heading_cmd", "speed_cmd"]].to_numpy()[:n]
    mismatch = np.abs(real_cmd - sim_cmd) > tol
    n_mismatched = np.any(mismatch, axis=1).sum()

    if n_mismatched > 0:
        pct = 100 * n_mismatched / n
        print(f"[warning] heading_cmd/speed_cmd differ between the two logs on "
              f"{n_mismatched}/{n} rows ({pct:.1f}%). The two runs may not have been "
              f"driven by the same command sequence - treat the comparison below with caution.")
    else:
        print("[ok] heading_cmd/speed_cmd match between logs - runs appear aligned.\n")

    return n


def variance_report(residual, angular=False):
    residual = np.asarray(residual, dtype=float)
    diffed = np.diff(residual)
    if angular:
        diffed = wrap_to_pi(diffed)

    return {
        "cumulative_mean": np.mean(residual),
        "cumulative_var":  np.var(residual),
        "cumulative_std":  np.std(residual),
        "diff_mean":       np.mean(diffed),
        "diff_var":        np.var(diffed),
        "diff_std":        np.std(diffed),
    }


def derived_speed(x, y, dt=DT):
    """speed(t) ~= hypot(dx, dy) / dt, from position differences. Length n-1."""
    dx = np.diff(np.asarray(x, dtype=float))
    dy = np.diff(np.asarray(y, dtype=float))
    return np.hypot(dx, dy) / dt


def compute_drift(real_path=REAL_LOG_PATH, sim_path=SIM_LOG_PATH, dt=DT):
    real_df = load_log(real_path)
    sim_df = load_log(sim_path)

    n = check_alignment(real_df, sim_df)
    real_df = real_df.iloc[:n].reset_index(drop=True)
    sim_df = sim_df.iloc[:n].reset_index(drop=True)

    residual_x = real_df["odom_x"] - sim_df["odom_x"]
    residual_y = real_df["odom_y"] - sim_df["odom_y"]
    residual_heading = wrap_to_pi(real_df["heading"] - sim_df["heading"])
    residual_speed = real_df["speed"] - sim_df["speed"]

    results = {
        "x":       variance_report(residual_x),
        "y":       variance_report(residual_y),
        "heading": variance_report(residual_heading, angular=True),
        "speed":   variance_report(residual_speed),
    }

    speed_from_real_odom = derived_speed(real_df["odom_x"], real_df["odom_y"], dt)
    speed_from_sim_odom = derived_speed(sim_df["odom_x"], sim_df["odom_y"], dt)
    derived_speed_residual = speed_from_real_odom - speed_from_sim_odom

    derived = {
        "derived_speed_residual_var": np.var(derived_speed_residual),
        "derived_speed_residual_std": np.std(derived_speed_residual),
        "logged_speed_vs_real_derived_diff_std": np.std(
            real_df["speed"].to_numpy()[1:] - speed_from_real_odom
        ),
    }

    return results, derived, n


def print_report(results, derived, n_rows):
    print(f"Rows compared: {n_rows}\n")

    print("=== Cumulative residual (real vs sim) ===")
    print(f"{'state':<10}{'mean':>14}{'var':>16}{'std':>14}")
    for key, vals in results.items():
        print(f"{key:<10}{vals['cumulative_mean']:>14.6f}"
              f"{vals['cumulative_var']:>16.6e}{vals['cumulative_std']:>14.6f}")

    print("\n=== One-step differenced residual (recommended for Q) ===")
    print(f"{'state':<10}{'mean':>14}{'var':>16}{'std':>14}")
    for key, vals in results.items():
        print(f"{key:<10}{vals['diff_mean']:>14.6f}"
              f"{vals['diff_var']:>16.6e}{vals['diff_std']:>14.6f}")

    for key, vals in results.items():
        if abs(vals["cumulative_mean"]) > 0.5 * vals["cumulative_std"] and vals["cumulative_std"] > 0:
            print(f"\n[note] '{key}' cumulative residual mean ({vals['cumulative_mean']:.4g}) is large "
                  f"relative to its std ({vals['cumulative_std']:.4g}) - looks like a persistent bias, "
                  f"not just noise. Consider correcting the model rather than absorbing this into Q.")

    print("\n=== Derived speed cross-check (position-difference based) ===")
    print(f"  std(speed_from_real_odom - speed_from_sim_odom): {derived['derived_speed_residual_std']:.6f}")
    print(f"  var(speed_from_real_odom - speed_from_sim_odom): {derived['derived_speed_residual_var']:.6e}")
    print(f"  std(logged real 'speed' vs derived speed_from_real_odom): "
          f"{derived['logged_speed_vs_real_derived_diff_std']:.6f}")

    print("\nSuggested Q diagonal (one-step differenced variance):")
    print(f"  x:       {results['x']['diff_var']:.6e}")
    print(f"  y:       {results['y']['diff_var']:.6e}")
    print(f"  heading: {results['heading']['diff_var']:.6e}")
    print(f"  speed:   {results['speed']['diff_var']:.6e}")


if __name__ == "__main__":
    results, derived, n_rows = compute_drift()
    print_report(results, derived, n_rows)