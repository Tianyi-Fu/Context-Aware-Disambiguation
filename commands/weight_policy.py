from __future__ import annotations

import math
import os
import re
from typing import Dict, List, Optional


_SETTINGS: Dict[str, float | bool] = {}


def configure_weight_policy(**kwargs) -> None:
    _SETTINGS.update(kwargs)


def _cfg(name: str, default):
    return _SETTINGS.get(name, default)


def _run_level(run_name: str | None) -> Optional[int]:
    if not run_name:
        return None
    m = re.search('_l([1-4])(?=$|_)', run_name.lower())
    if not m:
        return None
    return int(m.group(1))


def _is_pronoun_sentence(sentence: str | None) -> bool:
    if not sentence:
        return False
    s = str(sentence).strip().lower().replace('_', ' ')
    return re.search('\\b(it|this|that|them|these|those|one|ones)\\b', s) is not None


def _is_pronoun_run(run_name: str | None, sentence: str | None = None) -> bool:
    lname = (run_name or '').lower()
    if 'pronoun' in lname:
        return True
    lv = _run_level(run_name)
    return lv == 4 and _is_pronoun_sentence(sentence)


def _weights_for_run(run_name: str | None, sentence: str | None = None) -> tuple[float, float, float]:
    if run_name:
        lname = run_name.lower()
        level = _run_level(run_name)
        if level == 1:
            return (_cfg('W_SEM_L1', 0.50), _cfg('W_THEM_L1', 0.35), _cfg('W_SAL_L1', 0.15))
        if level == 2:
            return (_cfg('W_SEM_L2', 0.35), _cfg('W_THEM_L2', 0.50), _cfg('W_SAL_L2', 0.15))
        if level == 3:
            return (_cfg('W_SEM_L3', 0.40), _cfg('W_THEM_L3', 0.45), _cfg('W_SAL_L3', 0.15))
        if level == 4:
            if _is_pronoun_run(run_name, sentence):
                return (_cfg('W_SEM_PRON', 0.45), _cfg('W_THEM_PRON', 0.50), _cfg('W_SAL_PRON', 0.05))
            return (_cfg('W_SEM_L4', _cfg('W_SEM', 0.60)), _cfg('W_THEM_L4', _cfg('W_THEM', 0.30)), _cfg('W_SAL_L4', _cfg('W_SAL', 0.10)))
        if 'hypernym' in lname:
            return (_cfg('W_SEM_HYP', 0.70), _cfg('W_THEM_HYP', 0.25), _cfg('W_SAL_HYP', 0.05))
        if _is_pronoun_run(run_name, sentence):
            return (_cfg('W_SEM_PRON', 0.45), _cfg('W_THEM_PRON', 0.50), _cfg('W_SAL_PRON', 0.05))
    return (_cfg('W_SEM', 0.60), _cfg('W_THEM', 0.30), _cfg('W_SAL', 0.10))


def _lead_threshold(run_name: str | None) -> float:
    level = _run_level(run_name)
    if level == 1:
        return float(os.environ.get('CLEAR_LEAD_RATIO_L1', '0.30'))
    if level == 2:
        return float(os.environ.get('CLEAR_LEAD_RATIO_L2', '0.30'))
    if level == 3:
        return float(os.environ.get('CLEAR_LEAD_RATIO_L3', '0.32'))
    if level == 4:
        return float(os.environ.get('CLEAR_LEAD_RATIO_L4', str(_cfg('CLEAR_LEAD_RATIO_BASE', 0.2))))
    if run_name and 'hypernym' in run_name.lower():
        return float(_cfg('CLEAR_LEAD_RATIO_HYP', 0.15))
    return float(_cfg('CLEAR_LEAD_RATIO_BASE', 0.2))


def _adjust_weights_for_sem_collapse(*, w_sem: float, w_them: float, w_sal: float, w_ctx: float, sem_scores: List[float]) -> tuple[float, float, float, float, bool]:
    if not bool(_cfg('SEM_COLLAPSE_ENABLED', True)):
        return (w_sem, w_them, w_sal, w_ctx, False)
    if len(sem_scores) <= 1:
        return (w_sem, w_them, w_sal, w_ctx, False)
    s_min = min(sem_scores)
    s_max = max(sem_scores)
    s_range = s_max - s_min
    mu = sum(sem_scores) / len(sem_scores)
    var = sum(((x - mu) ** 2 for x in sem_scores)) / max(1, len(sem_scores))
    std = math.sqrt(var)
    if s_range > float(_cfg('SEM_COLLAPSE_RANGE', 0.06)) or std > float(_cfg('SEM_COLLAPSE_STD', 0.02)):
        return (w_sem, w_them, w_sal, w_ctx, False)
    sem_collapse_target = float(_cfg('SEM_COLLAPSE_TARGET', 0.30))
    if w_sem <= sem_collapse_target:
        return (w_sem, w_them, w_sal, w_ctx, True)
    delta = w_sem - sem_collapse_target
    w_sem = sem_collapse_target
    other = w_them + w_sal + w_ctx
    if other > 0:
        w_them += delta * (w_them / other)
        w_sal += delta * (w_sal / other)
        w_ctx += delta * (w_ctx / other)
    return (w_sem, w_them, w_sal, w_ctx, True)


def _adjust_weights_for_sem_reliability(*, w_sem: float, w_them: float, w_sal: float, w_ctx: float, sem_map: Dict[str, float], them_map: Dict[str, float], sal_map: Dict[str, float]) -> tuple[float, float, float, float, bool, str]:
    if not bool(_cfg('SEM_RELIABILITY_GATING', True)):
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    if len(sem_map) <= 1:
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    sem_reliability_target = float(_cfg('SEM_RELIABILITY_TARGET', 0.20))
    if w_sem <= sem_reliability_target:
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    sem_sorted = sorted(sem_map.items(), key=lambda x: x[1], reverse=True)
    them_sorted = sorted(them_map.items(), key=lambda x: x[1], reverse=True) if them_map else []
    sem_top_name, sem_top = sem_sorted[0]
    sem_second = sem_sorted[1][1] if len(sem_sorted) > 1 else 0.0
    sem_range = sem_sorted[0][1] - sem_sorted[-1][1]
    sem_gap = sem_top - sem_second
    sem_flat = sem_range < float(_cfg('SEM_RELIABILITY_MIN_RANGE', 0.10)) or sem_gap < float(_cfg('SEM_RELIABILITY_MIN_GAP', 0.04))
    them_conflict = False
    them_gap = 0.0
    if len(them_sorted) > 1:
        them_top_name, them_top = them_sorted[0]
        them_second = them_sorted[1][1]
        them_gap = them_top - them_second
        if sem_top_name != them_top_name and them_gap >= float(_cfg('SEM_RELIABILITY_CONFLICT_THEM_GAP', 0.10)) and (sem_gap <= float(_cfg('SEM_RELIABILITY_CONFLICT_SEM_GAP', 0.08))):
            them_conflict = True
    if not sem_flat and (not them_conflict):
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    reason = 'flat_semantic' if sem_flat and (not them_conflict) else 'semantic_thematic_conflict'
    delta = w_sem - sem_reliability_target
    if delta <= 0:
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    w_sem = sem_reliability_target
    other = w_them + w_sal + w_ctx
    if other > 0:
        w_them += delta * (w_them / other)
        w_sal += delta * (w_sal / other)
        w_ctx += delta * (w_ctx / other)
    if w_them <= 1e-09 and w_sal + w_ctx > 0:
        total = w_sal + w_ctx
        w_sal = w_sal / total * (1.0 - w_sem)
        w_ctx = w_ctx / total * (1.0 - w_sem)
    return (w_sem, w_them, w_sal, w_ctx, True, f'{reason}; sem_gap={sem_gap:.3f}; them_gap={them_gap:.3f}')


def _factor_confidence_from_map(score_map: Dict[str, float]) -> float:
    if len(score_map) <= 1:
        return 0.0
    vals = sorted((float(v) for v in score_map.values()), reverse=True)
    top1 = vals[0]
    top2 = vals[1] if len(vals) > 1 else 0.0
    bottom = vals[-1]
    gap = max(0.0, top1 - top2)
    rng = max(0.0, top1 - bottom)
    g = max(0.0, gap - float(_cfg('FACTOR_REL_MIN_GAP', 0.02))) / max(1e-09, float(_cfg('FACTOR_REL_GAP_SCALE', 0.12)))
    r = max(0.0, rng - float(_cfg('FACTOR_REL_MIN_RANGE', 0.08))) / max(1e-09, float(_cfg('FACTOR_REL_RANGE_SCALE', 0.25)))
    conf = float(_cfg('FACTOR_REL_GAP_WEIGHT', 0.65)) * g + float(_cfg('FACTOR_REL_RANGE_WEIGHT', 0.35)) * r
    return max(0.0, min(1.0, conf))


def _map_top_gap(score_map: Dict[str, float]) -> float:
    if len(score_map) <= 1:
        return 0.0
    vals = sorted((float(v) for v in score_map.values()), reverse=True)
    return max(0.0, vals[0] - (vals[1] if len(vals) > 1 else 0.0))


def _map_top_name(score_map: Dict[str, float]) -> str:
    if not score_map:
        return ''
    return max(score_map.items(), key=lambda kv: float(kv[1]))[0]


def _boost_one_factor(*, target: str, boost: float, w_sem: float, w_them: float, w_sal: float) -> tuple[float, float, float]:
    boost = max(0.0, float(boost))
    if boost <= 1e-09:
        return (w_sem, w_them, w_sal)
    w = {'sem': float(w_sem), 'them': float(w_them), 'sal': float(w_sal)}
    if target not in w:
        return (w_sem, w_them, w_sal)
    others = [k for k in ('sem', 'them', 'sal') if k != target]
    donor_total = max(1e-09, sum((max(0.0, w[k]) for k in others)))
    delta = min(boost, donor_total)
    w[target] += delta
    for k in others:
        share = max(0.0, w[k]) / donor_total
        w[k] = max(0.0, w[k] - delta * share)
    z = max(1e-09, w['sem'] + w['them'] + w['sal'])
    return (w['sem'] / z, w['them'] / z, w['sal'] / z)


def _enforce_min_share(weights: List[float], active: List[bool], total_budget: float, min_share: float) -> List[float]:
    if total_budget <= 0 or min_share <= 0:
        return weights
    active_ids = [i for i, a in enumerate(active) if a]
    if len(active_ids) <= 1:
        return weights
    share_cap = max(0.0, 1.0 / len(active_ids) - 1e-06)
    floor = min(min_share, share_cap) * total_budget
    adjusted = list(weights)
    need = [i for i in active_ids if adjusted[i] < floor]
    if not need:
        return adjusted
    deficit = sum((floor - adjusted[i] for i in need))
    donors = [i for i in active_ids if adjusted[i] > floor]
    donor_room = sum((adjusted[i] - floor for i in donors))
    if donor_room <= 1e-09:
        return adjusted
    for i in need:
        adjusted[i] = floor
    for i in donors:
        give = deficit * ((adjusted[i] - floor) / donor_room)
        adjusted[i] = max(floor, adjusted[i] - give)
    return adjusted


def _adjust_weights_for_factor_reliability(*, w_sem: float, w_them: float, w_sal: float, w_ctx: float, sem_map: Dict[str, float], them_map: Dict[str, float], sal_map: Dict[str, float]) -> tuple[float, float, float, float, bool, str]:
    if not bool(_cfg('FACTOR_RELIABILITY_ADAPTIVE', False)):
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    base_total = w_sem + w_them + w_sal
    if base_total <= 1e-09:
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    conf_sem = _factor_confidence_from_map(sem_map)
    conf_them = _factor_confidence_from_map(them_map)
    conf_sal = _factor_confidence_from_map(sal_map)
    conf = [conf_sem, conf_them, conf_sal]
    base_w = [w_sem, w_them, w_sal]
    active = [w > 1e-09 for w in base_w]
    if not any(active):
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    beta = max(0.0, float(_cfg('FACTOR_REL_BETA', 1.0)))
    raw = [0.0, 0.0, 0.0]
    for i in range(3):
        if not active[i]:
            raw[i] = 0.0
            continue
        raw[i] = base_w[i] * (1.0 + beta * conf[i])
    z = sum(raw)
    if z <= 1e-12:
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    scaled = [v * (base_total / z) for v in raw]
    scaled = _enforce_min_share(scaled, active, base_total, max(0.0, float(_cfg('FACTOR_REL_MIN_SHARE', 0.10))))
    new_sem, new_them, new_sal = scaled
    changed = abs(new_sem - w_sem) > 1e-06 or abs(new_them - w_them) > 1e-06 or abs(new_sal - w_sal) > 1e-06
    info = f'conf(sem/them/sal)={conf_sem:.3f}/{conf_them:.3f}/{conf_sal:.3f}; beta={beta:.2f}'
    return (new_sem, new_them, new_sal, w_ctx, changed, info)


def _adjust_weights_by_difficulty_policy(*, run_name: str, w_sem: float, w_them: float, w_sal: float, w_ctx: float, sem_map: Dict[str, float], them_map: Dict[str, float], sal_map: Dict[str, float]) -> tuple[float, float, float, float, bool, str]:
    if not bool(_cfg('DIFFICULTY_DYNAMIC_WEIGHTS', False)):
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    if bool(_cfg('UNWEIGHTED_THREE_FACTOR', True)):
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    lv = _run_level(run_name)
    if lv not in {1, 2, 3, 4}:
        return (w_sem, w_them, w_sal, w_ctx, False, '')
    sem_gap = _map_top_gap(sem_map)
    them_gap = _map_top_gap(them_map)
    sal_gap = _map_top_gap(sal_map)
    sem_top = _map_top_name(sem_map)
    them_top = _map_top_name(them_map)
    sal_top = _map_top_name(sal_map)
    margin = max(0.0, float(_cfg('DIFF_DYN_MARGIN', 0.02)))
    target = None
    boost = 0.0
    if lv == 1:
        if bool(_cfg('DIFF_DYN_L1_CONSENSUS_ENABLE', True)) and sem_top and (sem_top == them_top) and (sem_top != sal_top) and (sem_gap >= max(0.0, float(_cfg('DIFF_DYN_L1_CONSENSUS_MIN_SEM_GAP', 0.05)))) and (them_gap >= max(0.0, float(_cfg('DIFF_DYN_L1_CONSENSUS_MIN_THEM_GAP', 0.05)))):
            target = 'them' if them_gap >= sem_gap else 'sem'
            boost = max(0.0, float(_cfg('DIFF_DYN_L1_CONSENSUS_BOOST', 0.22)))
            n_sem, n_them, n_sal = _boost_one_factor(target=target, boost=boost, w_sem=w_sem, w_them=w_them, w_sal=w_sal)
            changed = abs(n_sem - w_sem) > 1e-06 or abs(n_them - w_them) > 1e-06 or abs(n_sal - w_sal) > 1e-06
            info = f'lv=1 consensus target={target} boost={boost:.2f} tops sem/them/sal={sem_top}/{them_top}/{sal_top} gaps sem/them/sal={sem_gap:.3f}/{them_gap:.3f}/{sal_gap:.3f}'
            return (n_sem, n_them, n_sal, w_ctx, changed, info)
        min_gap = max(0.0, float(_cfg('DIFF_DYN_MIN_GAP_L1', 0.08)))
        boost = max(0.0, float(_cfg('DIFF_DYN_BOOST_L1', 0.12)))
        if sem_gap >= min_gap and sem_gap >= them_gap + margin and (sem_gap >= sal_gap + margin):
            target = 'sem'
        elif them_gap >= min_gap and them_gap >= sem_gap + margin and (them_gap >= sal_gap + margin):
            target = 'them'
    elif lv == 2:
        min_gap = max(0.0, float(_cfg('DIFF_DYN_MIN_GAP_L2', 0.06)))
        boost = max(0.0, float(_cfg('DIFF_DYN_BOOST_L2', 0.12)))
        if them_gap >= min_gap and them_gap >= sem_gap + margin and (them_gap >= sal_gap + margin):
            target = 'them'
        elif sem_gap >= min_gap + margin and sem_gap >= them_gap + margin and (sem_gap >= sal_gap + margin):
            target = 'sem'
    else:
        min_gap = max(0.0, float(_cfg('DIFF_DYN_MIN_GAP_L34', 0.05)))
        boost = max(0.0, float(_cfg('DIFF_DYN_BOOST_L34', 0.15)))
        if them_gap >= min_gap and them_gap >= sal_gap + margin:
            target = 'them'
        elif sal_gap >= min_gap and sal_gap >= them_gap + margin:
            target = 'sal'
        elif sem_gap >= min_gap + margin:
            target = 'sem'
    if not target:
        return (w_sem, w_them, w_sal, w_ctx, False, f'lv={lv} no-boost gaps sem/them/sal={sem_gap:.3f}/{them_gap:.3f}/{sal_gap:.3f}')
    n_sem, n_them, n_sal = _boost_one_factor(target=target, boost=boost, w_sem=w_sem, w_them=w_them, w_sal=w_sal)
    changed = abs(n_sem - w_sem) > 1e-06 or abs(n_them - w_them) > 1e-06 or abs(n_sal - w_sal) > 1e-06
    info = f'lv={lv} target={target} boost={boost:.2f} gaps sem/them/sal={sem_gap:.3f}/{them_gap:.3f}/{sal_gap:.3f}'
    return (n_sem, n_them, n_sal, w_ctx, changed, info)
