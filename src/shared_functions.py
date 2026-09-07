import numpy as np
import os
import re
import sys
from datetime import datetime

def wrap_angle(angle):
    #This ensures that the output angle is within the valid -pi to pi range
    return (angle + np.pi) % (2.0 * np.pi) - np.pi  # Normalize to [-pi, pi)


def get_log_path(session_name, is_real, lab_number=None):
    """
    Build (and create) a timestamped log file path, e.g.:

        logs/lab2_logs/session1_B0B0/real/lab2_real_2026-08-31_13-56-29.csv
        logs/lab2_logs/session1_B0B0/sim/lab2_sim_2026-08-31_13-56-29.csv

    session_name : whatever you want to call this run/session (e.g. "session1_B0B0")
    is_real       : True for a real-robot run, False for a sim run
    lab_number    : which labN_logs folder to use. Auto-detected from the
                    filename of the script you ran (e.g. running lab2.py
                    picks up "2") - pass it explicitly if that detection
                    doesn't work for you (e.g. running from a notebook).
    """
    if lab_number is None:
        main_file = getattr(sys.modules.get("__main__"), "__file__", None)
        match = re.search(r"lab(\d+)", os.path.basename(main_file)) if main_file else None
        if not match:
            raise ValueError(
                "Couldn't auto-detect the lab number from the running "
                "file's name - pass lab_number explicitly, "
                "e.g. get_log_path(session_name, is_real, lab_number=2)"
            )
        lab_number = int(match.group(1))

    run_type = "real" if is_real else "sim"

    log_dir = os.path.join("logs", f"lab{lab_number}_logs", session_name, run_type)
    os.makedirs(log_dir, exist_ok=True)

    if is_real:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"lab{lab_number}_{run_type}_{timestamp}.csv"
    else:
        # Sim runs always write to the same file, so each run just overwrites the last
        filename = f"lab{lab_number}_{run_type}.csv"

    return os.path.join(log_dir, filename)