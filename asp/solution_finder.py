# asp/solution_finder.py
"""Wrapper for running the ASP solver runner script from project root."""

import os
import sys
import subprocess

from config.config import PROJECT_ROOT


def run_sparc_solution_finder(display_predicates=None):
    """Run ASP solver runner with display predicates configured via env."""
    if display_predicates is None:
        display_predicates = [
            "occurs",
            "show_operated_holds_name",
            "show_changed_holds",
            "show_changed_holds_name",
            "show_start_holds",
            "show_last_holds",
        ]

    os.environ["DISPLAY_PREDICATES"] = ",".join(display_predicates)

    runner_path = os.path.join(PROJECT_ROOT, "asp", "asp_solution_finder_runner.py")

    if not os.path.exists(runner_path):
        raise FileNotFoundError(
            f"asp_solution_finder_runner.py not found at: {runner_path}"
        )

    cmd = [sys.executable, runner_path]
    print(f"[ASP] Running: {cmd} (cwd={PROJECT_ROOT})")

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError as e:
        print(f"Error running ASP solution finder: {e}")
        raise
