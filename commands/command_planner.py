# commands/command_planner.py
# ------------------------------------------------------------
"""
Run SPARC, then parse its textual outputs to obtain

    • steps   –  plan length  (count of ‘occurs( … )’)
    • touched –  set of entities whose state changed

Counting the literal substring  “occurs(”  is the most robust way;
it works no matter what time-step numbers SPARC puts inside.
"""

from __future__ import annotations
import re
import pathlib
from typing import Set, Dict, Optional, Callable

from asp.goals import execute_user_goal
from config import config as cfg

# ------------------------------------------------------------------
_OCCURS_RE   = re.compile(r"occurs\(")        # just the marker
_TOUCHED_RE  = re.compile(r"\(([\w_]+)\)")    # any (…) capture first token

HistoryRecorder = Optional[Callable[[Optional[str], str, str], None]]

_HISTORY_RECORDER: HistoryRecorder = None

def register_history_recorder(fn: HistoryRecorder) -> None:
    """Register optional history recorder callback."""
    global _HISTORY_RECORDER
    _HISTORY_RECORDER = fn


def _count_occurs(path: pathlib.Path) -> int:
    """Return how many ‘occurs(’ tokens are in *path*; 0 if file missing."""
    if not path.exists():
        return 0
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return len(_OCCURS_RE.findall(txt))


def _parse_touched(path: pathlib.Path) -> Set[str]:
    """Return set of entity names that appear inside ‘( … )’."""
    if not path.exists():
        return set()
    txt = path.read_text(encoding="utf-8", errors="ignore")
    return {m.group(1) for m in _TOUCHED_RE.finditer(txt)}


# ------------------------------------------------------------------
def plan(
    asp_goal: str,
    *,
    line_id: Optional[str] = None,
    nl: str = "",
    history_recorder: HistoryRecorder = None,
) -> Dict[str, object]:
    """
    Call SPARC via `execute_user_goal(asp_goal)` and read the resulting
    OCCURS / OPERATED files.

    Parameters
    ----------
    asp_goal : str
        Final ASP command to execute.
    line_id : Optional[str]
        Group line id, such as "[g-k]".
    nl : str
        Original natural language command.
    history_recorder : Optional[Callable]
        Callback signature: history_recorder(line_id, asp_goal, nl)

    Returns
    -------
    dict { "steps": int, "touched": set[str] }
    """
    execute_user_goal(asp_goal)

    rec = history_recorder if history_recorder is not None else _HISTORY_RECORDER
    if rec is not None:
        try:
            rec(line_id, asp_goal, nl or asp_goal)
        except Exception:
            pass

    occurs_file   = pathlib.Path(cfg.OCCURS_OUTPUT)
    operated_file = pathlib.Path(cfg.OPERATED_OUTPUT)

    steps = _count_occurs(occurs_file)
    if steps == 0:
        steps = 1

    touched = _parse_touched(operated_file)

    return {"steps": steps, "touched": touched}
