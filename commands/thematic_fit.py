from __future__ import annotations
from typing import Optional, Dict, Tuple, List
import math
from functools import lru_cache
from kg.history_manager import load_user_history, set_active_history_file as _set_active_history_file
from kg.history_analyzer import extract_asp_command as hx_extract, extract_keywords as hx_keywords, filter_valid_history_entries
_EPS = 1e-09
_CURRENT_HISTORY_FILE: Optional[str] = None

def set_thematic_history_file(path: str | None) -> None:
    global _CURRENT_HISTORY_FILE
    _CURRENT_HISTORY_FILE = str(path) if path else None
    try:
        if path:
            _set_active_history_file(str(path))
    except Exception:
        pass
    try:
        _build_lift_stats.cache_clear()
    except Exception:
        pass

def _target_for(pred_now: str, s: Optional[str], o: Optional[str]) -> Optional[str]:
    if pred_now in ('inside', 'on', 'has'):
        return o
    if pred_now == 'heated':
        return s
    return o or s

@lru_cache(maxsize=1)
def _build_lift_stats() -> Tuple[Dict[str, int], int, Dict[Tuple[str, Optional[str]], Dict[str, int]], Dict[Tuple[str, Optional[str]], int]]:
    base_counts: Dict[str, int] = {}
    base_total: int = 0
    cond_counts: Dict[Tuple[str, Optional[str]], Dict[str, int]] = {}
    cond_totals: Dict[Tuple[str, Optional[str]], int] = {}
    try:
        lines: List[str] = list(load_user_history())
    except Exception:
        lines = []
    rows = filter_valid_history_entries(lines)
    for line in rows:
        asp = hx_extract(line)
        if not asp:
            continue
        try:
            pred, subj, obj, _ = hx_keywords(asp)
        except Exception:
            continue
        if not pred:
            continue
        tgt_obj = _target_for(pred, subj, obj)
        if tgt_obj:
            base_total += 1
            base_counts[tgt_obj] = base_counts.get(tgt_obj, 0) + 1
        if not tgt_obj:
            continue
        key_p = (pred, None)
        cond_totals[key_p] = cond_totals.get(key_p, 0) + 1
        cond_counts.setdefault(key_p, {}).setdefault(tgt_obj, 0)
        cond_counts[key_p][tgt_obj] += 1
        if pred in ('inside', 'on', 'has') and subj:
            key_pf = (pred, subj)
            cond_totals[key_pf] = cond_totals.get(key_pf, 0) + 1
            cond_counts.setdefault(key_pf, {}).setdefault(tgt_obj, 0)
            cond_counts[key_pf][tgt_obj] += 1
    return (base_counts, base_total, cond_counts, cond_totals)

def _squash_lift(lift: float) -> float:
    diff = lift - 1.0
    return diff / (1.0 + abs(diff))

def thematic_fit(pred_now: str, obj_name: str, furn_now: Optional[str]) -> float:
    pred_now = (pred_now or '').strip()
    obj_name = (obj_name or '').strip()
    furn_now = (furn_now or '').strip() if furn_now else None
    if not pred_now or not obj_name:
        return 0.0
    base_counts, base_total, cond_counts, cond_totals = _build_lift_stats()
    if base_total == 0:
        return 0.0
    alpha = 1.0
    V = max(1, len(base_counts))
    c_base = base_counts.get(obj_name, 0)
    p_base = (c_base + alpha) / (base_total + alpha * V)
    cond_key = (pred_now, None)
    if pred_now in ('inside', 'on', 'has') and furn_now:
        cond_key = (pred_now, furn_now)
    c_cond = cond_counts.get(cond_key, {}).get(obj_name, 0)
    cond_total = cond_totals.get(cond_key, 0)
    if cond_total < 3 and cond_key != (pred_now, None):
        cond_key = (pred_now, None)
        c_cond = cond_counts.get(cond_key, {}).get(obj_name, 0)
        cond_total = cond_totals.get(cond_key, 0)
    if cond_total + alpha * V > 0:
        p_cond = (c_cond + alpha) / (cond_total + alpha * V)
    else:
        p_cond = 1.0 / V
    lift = p_cond / max(p_base, _EPS)
    score = _squash_lift(lift)
    k = 5.0
    evidence = 1.0 - math.exp(-(cond_total / k))
    return float(score * evidence)
