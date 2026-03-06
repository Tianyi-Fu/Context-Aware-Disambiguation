# asp/goals.py

import os
from pathlib import Path
from asp.file_manager import (
    add_goal_to_asp_file,
    add_predicted_goal_to_asp_file,
    clean_existing_goals_and_success
)
from asp.solution_finder import run_sparc_solution_finder
from config.config import SHOW_CHANGED_HOLDS_OUTPUT, SHOW_CHANGED_HOLDS_NAME_OUTPUT

def execute_user_goal(user_goal):

    clean_existing_goals_and_success()

    add_goal_to_asp_file(user_goal)


    predicates_to_show = ["occurs","show_operated_holds_name","show_changed_holds",
        "show_changed_holds_name","show_start_holds","show_last_holds"]
    run_sparc_solution_finder(display_predicates=predicates_to_show)

    rename_changed_holds_files("_user")


def execute_predicted_goal(predicted_goal):

    add_predicted_goal_to_asp_file(predicted_goal)

    full_predicates = [
        "occurs","show_operated_holds_name","show_changed_holds",
        "show_changed_holds_name","show_start_holds","show_last_holds"
    ]
    run_sparc_solution_finder(display_predicates=full_predicates)


    rename_changed_holds_files("_predicted")


def rename_changed_holds_files(suffix="_user"):

    old_changed = Path(SHOW_CHANGED_HOLDS_OUTPUT)
    new_changed = old_changed.with_name(f"{old_changed.stem}{suffix}{old_changed.suffix}")

    old_changed_name = Path(SHOW_CHANGED_HOLDS_NAME_OUTPUT)
    new_changed_name = old_changed_name.with_name(f"{old_changed_name.stem}{suffix}{old_changed_name.suffix}")


    if new_changed.exists():
        new_changed.unlink()
    if new_changed_name.exists():
        new_changed_name.unlink()


    if old_changed.exists():
        old_changed.rename(new_changed)

    if old_changed_name.exists():
        old_changed_name.rename(new_changed_name)
