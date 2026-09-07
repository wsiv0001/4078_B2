"""
plot_calibration.py

Telemetry plotting for Lab 1.

analyse_log()      - load a log CSV and compute error columns
plot_calibration()  - plot one run (sim OR real)
plot_comparison()   - overlay a sim run and a real run for the same commands

Set SIM_LOG_PATH and REAL_LOG_PATH below to the two files you want to compare,
then run this script directly.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.shared_functions import wrap_angle

# --- Set these to the two logs you want to compare --------------------------
SIM_LOG_PATH = "logs/lab1_logs/session1_SB-DAE7/sim/lab1_sim.csv"
REAL_LOG_PATH = "logs/lab1_logs/session1_SB-DAE7/real/lab1_real_2026-09-08_00-54-37.csv"

DT = 0.1  # must match the dt used in dynamics.py / lab1.py


def _detect_is_real(data):
    """Real robot logs never populate est_x (no state estimator hooked up
    yet); sim logs do. Used to auto-pick which speed/heading source to trust."""
    if "est_x" not in data.columns:
        return False
    return data.est_x.isna().all()


def add_motion_columns(data, dt=DT, window=3, min_speed=0.01):
    """Derive what the robot ACTUALLY did from its position trace.

    Needed for real logs, where the `speed`/`heading` columns just echo the
    commanded value rather than measuring anything. Sim logs don't need this
    since their speed/heading come from the physics model, not an echo.
    """
    dx = data.gt_x.diff(periods=window)
    dy = data.gt_y.diff(periods=window)

    data["speed_from_motion"] = np.hypot(dx, dy) / (window * dt)

    # Direction of travel is meaningless when basically stationary - blank
    # those samples out instead of plotting noise.
    heading_from_motion = np.arctan2(dx, dy)
    data["heading_from_motion"] = heading_from_motion.where(
        data.speed_from_motion >= min_speed
    )
    return data


def analyse_log(log_path, is_real=None):
    """Load a telemetry CSV and add error columns.

    Expects columns: gt_x, gt_y, setpoint_x, setpoint_y,
                      heading, heading_cmd, speed, speed_cmd
    (real logs should also have est_x, used to auto-detect real vs sim)
    """
    data = pd.read_csv(log_path)

    if is_real is None:
        is_real = _detect_is_real(data)
    data["is_real"] = is_real

    if is_real:
        data = add_motion_columns(data)
        # First `window` samples have no displacement to measure yet -
        # backfill from the raw (echoed) column so early metrics don't drop out.
        data["speed_measured"] = data.speed_from_motion.fillna(data.speed)
        data["heading_measured"] = data.heading_from_motion.fillna(data.heading)
    else:
        data["speed_measured"] = data.speed
        data["heading_measured"] = data.heading

    # Distance from goal marker
    data["position_error"] = np.sqrt(
        (data.gt_x - data.setpoint_x) ** 2 +
        (data.gt_y - data.setpoint_y) ** 2
    )

    # Heading error, wrapped to [-pi, pi]
    data["heading_error"] = wrap_angle(data.heading_cmd - data.heading_measured)

    # Speed error
    data["speed_error"] = data.speed_cmd - data.speed_measured

    return data


def plot_calibration(log_path, title="Calibration"):
    """Plot one run: speed response, heading response, distance-to-goal,
    heading error, speed error, and the path taken."""

    data = analyse_log(log_path)
    t = np.arange(len(data)) * DT

    fig, (ax1, ax2) = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")

    ax1[0].plot(t, data.speed_measured, lw=2, label="Measured")
    ax1[0].plot(t, data.speed_cmd, "--", lw=2, label="Command")
    ax1[0].set_title("Speed Response")
    ax1[0].set_xlabel("Time (s)")
    ax1[0].set_ylabel("Speed (m/s)")
    ax1[0].grid(True, alpha=0.3)
    ax1[0].legend()

    ax1[1].plot(t, data.heading_measured, lw=2, label="Measured")
    ax1[1].plot(t, data.heading_cmd, "--", lw=2, label="Command")
    ax1[1].set_title("Heading Response")
    ax1[1].set_xlabel("Time (s)")
    ax1[1].set_ylabel("Heading (rad)")
    ax1[1].grid(True, alpha=0.3)
    ax1[1].legend()

    ax1[2].plot(t, data.position_error, lw=2)
    ax1[2].set_title("Distance from Goal")
    ax1[2].set_xlabel("Time (s)")
    ax1[2].set_ylabel("Error (m)")
    ax1[2].grid(True, alpha=0.3)

    ax2[0].plot(t, data.heading_error, lw=2)
    ax2[0].set_title("Heading Error")
    ax2[0].set_xlabel("Time (s)")
    ax2[0].set_ylabel("Error (rad)")
    ax2[0].grid(True, alpha=0.3)

    ax2[1].plot(t, data.speed_error, lw=2)
    ax2[1].set_title("Speed Error")
    ax2[1].set_xlabel("Time (s)")
    ax2[1].set_ylabel("Error (m/s)")
    ax2[1].grid(True, alpha=0.3)

    ax2[2].plot(data.gt_x, data.gt_y, lw=2)
    ax2[2].set_title("Path")
    ax2[2].set_xlabel("x")
    ax2[2].set_ylabel("y")
    ax2[2].grid(True, alpha=0.3)
    ax2[2].set_aspect("equal", adjustable="box")

    source_tag = "real (from motion)" if data.is_real.iloc[0] else "sim"
    fig.suptitle(f"{title} - {source_tag}")

    print("\n--- Calibration Metrics ---")
    print(f"Position RMSE      : {np.sqrt(np.mean(data.position_error**2)):.3f} m")
    print(f"Max Position Error : {data.position_error.max():.3f} m")
    print(f"Heading RMSE       : {np.sqrt(np.mean(data.heading_error**2)):.3f} rad")
    print(f"Max Heading Error  : {np.abs(data.heading_error).max():.3f} rad")
    print(f"Speed RMSE         : {np.sqrt(np.mean(data.speed_error**2)):.3f} m/s")
    print(f"Max Speed Error    : {np.abs(data.speed_error).max():.3f} m/s")

    plt.show()


def plot_comparison(sim_path=SIM_LOG_PATH, real_path=REAL_LOG_PATH, title="Sim vs Real"):
    """Overlay a sim run and a real run for the SAME command sequence.

    Four panels: xy path, heading vs time, speed vs time, distance travelled.
    The runs do not need to be the same length.
    """
    sim = analyse_log(sim_path, is_real=False)
    real = analyse_log(real_path, is_real=True)

    t_sim = np.arange(len(sim)) * DT
    t_real = np.arange(len(real)) * DT

    fig, ax = plt.subplots(2, 2, figsize=(10, 7))

    # Path
    ax[0, 0].plot(sim.gt_x, sim.gt_y, lw=2, label="Sim")
    ax[0, 0].plot(real.gt_x, real.gt_y, lw=2, label="Real")
    ax[0, 0].scatter(sim.setpoint_x.iloc[0], sim.setpoint_y.iloc[0],
                      marker="*", s=180, color="gold", zorder=3, label="Goal")
    ax[0, 0].set_title("Path")
    ax[0, 0].set_xlabel("x (m)")
    ax[0, 0].set_ylabel("y (m)")
    ax[0, 0].set_aspect("equal", adjustable="datalim")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].legend()

    # Heading
    ax[0, 1].plot(t_sim, sim.heading_cmd, "--", lw=2, color="gray", label="Command")
    ax[0, 1].plot(t_sim, sim.heading, lw=2, label="Sim")
    ax[0, 1].plot(t_real, real.heading_measured, lw=2, label="Real (from motion)")
    ax[0, 1].set_title("Heading")
    ax[0, 1].set_xlabel("Time (s)")
    ax[0, 1].set_ylabel("Heading (rad)")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].legend(fontsize=8)

    # Speed
    ax[1, 0].plot(t_sim, sim.speed_cmd, "--", lw=2, color="gray", label="Command")
    ax[1, 0].plot(t_sim, sim.speed, lw=2, label="Sim")
    ax[1, 0].plot(t_real, real.speed_measured, lw=2, label="Real (from motion)")
    ax[1, 0].set_title("Speed")
    ax[1, 0].set_xlabel("Time (s)")
    ax[1, 0].set_ylabel("Speed (m/s)")
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].legend(fontsize=8)

    # Distance travelled (cumulative)
    sim_dist = np.hypot(sim.gt_x.diff(), sim.gt_y.diff()).fillna(0.0).cumsum()
    real_dist = np.hypot(real.gt_x.diff(), real.gt_y.diff()).fillna(0.0).cumsum()
    ax[1, 1].plot(t_sim, sim_dist, lw=2, label="Sim")
    ax[1, 1].plot(t_real, real_dist, lw=2, label="Real")
    ax[1, 1].set_title("Distance Travelled")
    ax[1, 1].set_xlabel("Time (s)")
    ax[1, 1].set_ylabel("Distance (m)")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].legend()

    fig.suptitle(title)
    fig.tight_layout()

    print("\n--- Sim vs Real ---")
    print(f"Steps                 : sim {len(sim)}, real {len(real)}")
    print(f"Final position   sim  : ({sim.gt_x.iloc[-1]:+.3f}, {sim.gt_y.iloc[-1]:+.3f}) m")
    print(f"Final position   real : ({real.gt_x.iloc[-1]:+.3f}, {real.gt_y.iloc[-1]:+.3f}) m")
    print(f"Distance travelled    : sim {sim_dist.iloc[-1]:.3f} m, real {real_dist.iloc[-1]:.3f} m")

    plt.show()


if __name__ == "__main__":
    plot_comparison()