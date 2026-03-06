from __future__ import annotations
import math
import json
import re
from itertools import combinations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from config.config import PROJECT_ROOT, CONCEPT_SOURCE_MODE, CONCEPT_MATCH_MODE, CONCEPT_STHEM_MODE, CONCEPT_ENV_ENABLED, CONCEPT_ENV_MODE, CONCEPT_ACTION_WEIGHT, CONCEPT_ENV_WEIGHT, CONCEPT_TOPK, CONCEPT_ENV_MAX_SRC, CONCEPT_ENV_WEAK_MARGIN, CONCEPT_ENV_WEAK_SCALE, CONCEPT_FURN_NMIN, CONCEPT_FURN_BACKOFF, CONCEPT_ENV_PRED_NMIN, CONCEPT_ENV_PRED_BACKOFF, CONCEPT_HABIT_MAX_SIZE, CONCEPT_HABIT_MIN_SUPPORT, CONCEPT_HABIT_MIN_LIFT, CONCEPT_HABIT_MIN_ACTIVE_OVERLAP, CONCEPT_HABIT_IDF_GAMMA, CONCEPT_HABIT_SCOPE, CONCEPT_HABIT_BASKET_MODE, CONCEPT_HABIT_BASKET_MODE_L1, CONCEPT_HABIT_BASKET_MODE_L2, CONCEPT_HABIT_BASKET_MODE_L3, CONCEPT_HABIT_BASKET_MODE_L4, CONCEPT_IDF_GAMMA, CONCEPT_LOCAL_DISC_GAMMA, CONCEPT_MIN_LOCAL_DISC
from kg.history_manager import get_active_history_file
from kg.history_analyzer import extract_keywords as hx_keywords
_GROUP_HDR_RE = re.compile('^\\s*\\[\\s*(?:%\\s*)?(\\d+)\\b')
CONCEPT_DIR = Path(PROJECT_ROOT) / 'data' / 'concepts'
OBJ2CONCEPTS_PATH = CONCEPT_DIR / 'object_to_concepts.json'
L0_CACHE_PATH = CONCEPT_DIR / 'l0_cache.json'
DOMAIN_OBJECTS_PATH = CONCEPT_DIR / 'domain_objects.json'
_FURNITURE_TYPES = {'OnFurniture', 'InsideFurniture', 'Light'}
DEFAULT_NMIN = 20
DEFAULT_ROOM_BACKOFF = 0.85
DEFAULT_PRED_BACKOFF = 0.7
_EPS = 1e-09

@dataclass(frozen=True)
class ConceptTransaction:
    group: str
    line: int
    pred: str
    furniture: str
    concepts: Tuple[str, ...]
    context_tokens: Tuple[str, ...]
    prev_changed_concepts: Tuple[str, ...]
    current_changed_concepts: Tuple[str, ...]
    current_changed_entities: Tuple[str, ...]
    target: str

def _iter_history_lines_with_group(path: Path) -> Iterable[Tuple[str, int, str]]:
    cur_group: Optional[str] = None
    line_idx = 0
    with path.open('r', encoding='utf-8') as f:
        for raw in f:
            s = raw.strip()
            if not s:
                continue
            m = _GROUP_HDR_RE.match(s)
            if m:
                cur_group = m.group(1)
                line_idx = 0
                continue
            if s == ']':
                cur_group = None
                line_idx = 0
                continue
            if '|' not in s:
                continue
            line_idx += 1
            asp = s.split('|', 1)[0].strip()
            if not asp:
                continue
            yield (cur_group or '0', line_idx, asp)

def _target_for(pred: str, subj: Optional[str], obj: Optional[str]) -> Optional[str]:
    if pred in ('inside', 'on', 'has'):
        return obj
    if pred == 'heated':
        return subj
    return obj or subj

def _furniture_key_for(pred: str, subj: Optional[str], furn_now: Optional[str]=None) -> str:
    if pred in ('inside', 'on', 'has'):
        base = furn_now if furn_now is not None else subj
        return (base or '__none__').strip()
    return '__none__'

@lru_cache(maxsize=4)
def _load_object_to_concepts(path: str) -> Dict[str, List[str]]:
    if str(CONCEPT_SOURCE_MODE).strip().lower() == 'l0':
        p0 = Path(L0_CACHE_PATH)
        if not p0.exists():
            return {}
        raw = json.loads(p0.read_text(encoding='utf-8'))
        out: Dict[str, List[str]] = {}
        if isinstance(raw, dict):
            for oid, row in raw.items():
                if not isinstance(row, dict):
                    continue
                l0 = row.get('l0', {})
                feats: Set[str] = set()
                if isinstance(l0, dict):
                    for dim in ('structure', 'shape', 'physical'):
                        vals = l0.get(dim, [])
                        if isinstance(vals, list):
                            for v in vals:
                                s = str(v).strip()
                                if s:
                                    feats.add(f'{dim}:{s}')
                if feats:
                    out[str(oid)] = sorted(feats)
        return out
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding='utf-8'))

@lru_cache(maxsize=4)
def _load_object_type_map(path: str) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding='utf-8'))
    out: Dict[str, str] = {}
    for row in data.get('objects', []):
        oid = str(row.get('object_id', '')).strip()
        typ = str(row.get('object_type', '')).strip()
        if oid:
            out[oid] = typ
    return out

@lru_cache(maxsize=4)
def _build_transactions(history_path: str) -> Tuple[ConceptTransaction, ...]:
    path = Path(history_path)
    if not path.exists():
        return tuple()
    obj2concepts = _load_object_to_concepts(str(OBJ2CONCEPTS_PATH))
    txs: List[ConceptTransaction] = []
    prev_concepts_by_group: Dict[str, Tuple[str, ...]] = {}
    for group, line_idx, asp in _iter_history_lines_with_group(path):
        try:
            pred, subj, obj, _ = hx_keywords(asp)
        except Exception:
            continue
        if not pred:
            continue
        target = _target_for(pred, subj, obj)
        if not target:
            continue
        concepts = obj2concepts.get(target, [])
        prev_concepts = prev_concepts_by_group.get(group, tuple())
        curr_concepts = tuple(sorted(set(concepts)))
        curr_entities = (str(target),) if target else tuple()
        txs.append(ConceptTransaction(group=str(group), line=int(line_idx), pred=str(pred), furniture=_furniture_key_for(pred, subj), concepts=curr_concepts, context_tokens=tuple(), prev_changed_concepts=tuple(sorted(set(prev_concepts))), current_changed_concepts=curr_concepts, current_changed_entities=curr_entities, target=str(target)))
        prev_concepts_by_group[group] = curr_concepts
    return tuple(txs)

class ConceptHistoryIndex:

    def __init__(self, *, history_path: str, nmin: int=DEFAULT_NMIN, room_backoff: float=DEFAULT_ROOM_BACKOFF, pred_backoff: float=DEFAULT_PRED_BACKOFF) -> None:
        self.history_path = history_path
        self.nmin = max(1, int(nmin))
        self.room_backoff = float(room_backoff)
        self.pred_backoff = float(pred_backoff)
        self._transactions = _build_transactions(history_path)
        self._by_pred: Dict[str, List[ConceptTransaction]] = {}
        for t in self._transactions:
            self._by_pred.setdefault(t.pred, []).append(t)
        self._obj2concepts = _load_object_to_concepts(str(OBJ2CONCEPTS_PATH))
        self._obj_type_map = _load_object_type_map(str(DOMAIN_OBJECTS_PATH))
        self._obj_count = max(1, len(self._obj2concepts))
        self._concept_df: Dict[str, int] = {}
        for _, cids in self._obj2concepts.items():
            for cid in set(cids):
                self._concept_df[cid] = self._concept_df.get(cid, 0) + 1
        self._env_pair_counts_by_group: Dict[str, Dict[str, Dict[str, int]]] = {}
        self._env_src_totals_by_group: Dict[str, Dict[str, int]] = {}
        self._env_pair_counts_by_group_pred: Dict[str, Dict[str, Dict[str, Dict[str, int]]]] = {}
        self._env_src_totals_by_group_pred: Dict[str, Dict[str, Dict[str, int]]] = {}
        by_group: Dict[str, List[ConceptTransaction]] = {}
        for t in self._transactions:
            by_group.setdefault(t.group, []).append(t)
        for g in by_group:
            by_group[g].sort(key=lambda x: x.line)
            seq = by_group[g]
            pair_counts: Dict[str, Dict[str, int]] = {}
            src_totals: Dict[str, int] = {}
            pair_counts_pred: Dict[str, Dict[str, Dict[str, int]]] = {}
            src_totals_pred: Dict[str, Dict[str, int]] = {}
            for i in range(len(seq) - 1):
                src_tx = seq[i]
                dst_tx = seq[i + 1]
                if not src_tx.current_changed_entities or not dst_tx.concepts:
                    continue
                dst_target = str(dst_tx.target)
                dst_pred = str(dst_tx.pred)
                for src_ent in src_tx.current_changed_entities:
                    if str(src_ent) == dst_target:
                        continue
                    src_typ = self._obj_type_map.get(str(src_ent), '')
                    if src_typ in _FURNITURE_TYPES:
                        continue
                    src_concepts = self._obj2concepts.get(str(src_ent), [])
                    if not src_concepts:
                        continue
                    for src in sorted(set(src_concepts)):
                        src_totals[src] = src_totals.get(src, 0) + 1
                        d = pair_counts.setdefault(src, {})
                        for dst in dst_tx.concepts:
                            d[dst] = d.get(dst, 0) + 1
                        sp = src_totals_pred.setdefault(dst_pred, {})
                        sp[src] = sp.get(src, 0) + 1
                        dp = pair_counts_pred.setdefault(dst_pred, {}).setdefault(src, {})
                        for dst in dst_tx.concepts:
                            dp[dst] = dp.get(dst, 0) + 1
            self._env_pair_counts_by_group[g] = pair_counts
            self._env_src_totals_by_group[g] = src_totals
            self._env_pair_counts_by_group_pred[g] = pair_counts_pred
            self._env_src_totals_by_group_pred[g] = src_totals_pred
        self._last_debug: Dict[str, object] = {}

    def _concept_idf(self, cid: str) -> float:
        df = float(self._concept_df.get(cid, 0))
        return math.log((self._obj_count + 1.0) / (df + 1.0)) + 1.0

    def _aggregate_concepts(self, concept_ids: List[str], concept_freq: Dict[str, float], concept_weight: Optional[Dict[str, float]]=None) -> float:
        if not concept_ids:
            return 0.0
        vals: List[Tuple[float, float]] = []
        for cid in concept_ids:
            base = float(concept_freq.get(cid, 0.0))
            w = float((concept_weight or {}).get(cid, 1.0))
            if w <= 0.0:
                continue
            vals.append((base * w, w))
        if not vals:
            return 0.0
        k = int(CONCEPT_TOPK)
        if k > 0 and len(vals) > k:
            vals.sort(key=lambda x: x[0], reverse=True)
            vals = vals[:k]
        num = sum((v for v, _ in vals))
        den = sum((w for _, w in vals))
        if den <= _EPS:
            return num / max(1, len(vals))
        return num / den

    def _filter_by_context(self, txs: Iterable[ConceptTransaction], ctx_tokens: Set[str]) -> List[ConceptTransaction]:
        if not ctx_tokens:
            return list(txs)
        return [t for t in txs if ctx_tokens.issubset(set(t.context_tokens))]

    def _concept_freqs(self, txs: List[ConceptTransaction]) -> Dict[str, float]:
        if not txs:
            return {}
        total = len(txs)
        counts: Dict[str, int] = {}
        for t in txs:
            for c in t.concepts:
                counts[c] = counts.get(c, 0) + 1
        return {c: cnt / total for c, cnt in counts.items()}

    def _env_concept_freqs(self, prev_changed_concepts: List[str], *, pred: Optional[str]=None, exclude_group: Optional[str]=None) -> Tuple[Dict[str, float], str]:
        if not prev_changed_concepts:
            return ({}, 'none')
        ex = str(exclude_group) if exclude_group is not None else None

        def _collect_global() -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
            pc: Dict[str, Dict[str, int]] = {}
            st: Dict[str, int] = {}
            for g, pair_map in self._env_pair_counts_by_group.items():
                if ex is not None and g == ex:
                    continue
                src_map = self._env_src_totals_by_group.get(g, {})
                for src, total in src_map.items():
                    st[src] = st.get(src, 0) + int(total)
                for src, dst_counts in pair_map.items():
                    d = pc.setdefault(src, {})
                    for dst, cnt in dst_counts.items():
                        d[dst] = d.get(dst, 0) + int(cnt)
            return (pc, st)

        def _collect_pred(p: str) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int]]:
            pc: Dict[str, Dict[str, int]] = {}
            st: Dict[str, int] = {}
            for g, pred_map in self._env_pair_counts_by_group_pred.items():
                if ex is not None and g == ex:
                    continue
                pair_map = pred_map.get(p, {})
                src_map = self._env_src_totals_by_group_pred.get(g, {}).get(p, {})
                for src, total in src_map.items():
                    st[src] = st.get(src, 0) + int(total)
                for src, dst_counts in pair_map.items():
                    d = pc.setdefault(src, {})
                    for dst, cnt in dst_counts.items():
                        d[dst] = d.get(dst, 0) + int(cnt)
            return (pc, st)
        mode = 'global'
        backoff = 1.0
        agg_pair_counts, agg_src_totals = _collect_global()
        if pred:
            pred_pc, pred_st = _collect_pred(str(pred))
            pred_total = sum(pred_st.values())
            if pred_total >= max(1, int(CONCEPT_ENV_PRED_NMIN)):
                agg_pair_counts, agg_src_totals = (pred_pc, pred_st)
                mode = 'pred'
                backoff = 1.0
            elif pred_total > 0:
                mode = 'pred_backoff_global'
                backoff = float(CONCEPT_ENV_PRED_BACKOFF)
        srcs = [s for s in sorted(set(prev_changed_concepts)) if s in agg_pair_counts]
        if not srcs:
            return ({}, mode)
        max_src = max(1, int(CONCEPT_ENV_MAX_SRC))
        if len(srcs) > max_src:
            srcs = sorted(srcs, key=lambda s: self._concept_idf(s), reverse=True)[:max_src]
        agg: Dict[str, float] = {}
        for src in srcs:
            total = max(1, agg_src_totals.get(src, 0))
            for dst, cnt in agg_pair_counts.get(src, {}).items():
                agg[dst] = agg.get(dst, 0.0) + cnt / float(total)
        scale = max(1, len(srcs))
        for k in list(agg.keys()):
            agg[k] = agg[k] / float(scale) * backoff
        return (agg, mode)

    def _tx_basket_for_habit(self, tx: ConceptTransaction, level: Optional[int]=None) -> Set[str]:
        mode = self._habit_basket_mode(level=level)
        if mode in {'trigger_only', 'changed_only', 'trigger'}:
            basket: Set[str] = set(tx.prev_changed_concepts)
        elif mode in {'target_only', 'target'}:
            basket = set(tx.concepts)
        else:
            basket = set(tx.prev_changed_concepts)
            basket.update(tx.concepts)
        return basket

    def _habit_basket_mode(self, level: Optional[int]=None) -> str:
        mode = str(CONCEPT_HABIT_BASKET_MODE).strip().lower()
        if level == 1 and CONCEPT_HABIT_BASKET_MODE_L1:
            mode = CONCEPT_HABIT_BASKET_MODE_L1
        elif level == 2 and CONCEPT_HABIT_BASKET_MODE_L2:
            mode = CONCEPT_HABIT_BASKET_MODE_L2
        elif level == 3 and CONCEPT_HABIT_BASKET_MODE_L3:
            mode = CONCEPT_HABIT_BASKET_MODE_L3
        elif level == 4 and CONCEPT_HABIT_BASKET_MODE_L4:
            mode = CONCEPT_HABIT_BASKET_MODE_L4
        return mode

    def _mine_habit_patterns(self, txs: List[ConceptTransaction], level: Optional[int]=None) -> Tuple[List[Tuple[Tuple[str, ...], float]], Dict[str, float]]:
        baskets: List[Set[str]] = []
        for t in txs:
            b = self._tx_basket_for_habit(t, level=level)
            if len(b) >= 2:
                baskets.append(b)
        n = len(baskets)
        if n <= 0:
            return ([], {'num_baskets': 0.0, 'num_patterns': 0.0})
        single_cnt: Dict[str, int] = {}
        for b in baskets:
            for c in b:
                single_cnt[c] = single_cnt.get(c, 0) + 1
        max_size = max(2, int(CONCEPT_HABIT_MAX_SIZE))
        min_support = max(2, int(CONCEPT_HABIT_MIN_SUPPORT))
        combo_cnt: Dict[Tuple[str, ...], int] = {}
        for b in baskets:
            sb = sorted(b)
            kmax = min(max_size, len(sb))
            for k in range(2, kmax + 1):
                for comb in combinations(sb, k):
                    combo_cnt[comb] = combo_cnt.get(comb, 0) + 1
        min_lift = max(1.0, float(CONCEPT_HABIT_MIN_LIFT))
        patterns: List[Tuple[Tuple[str, ...], float]] = []
        for comb, cnt in combo_cnt.items():
            if cnt < min_support:
                continue
            p_joint = cnt / float(n)
            if len(comb) <= 2:
                p_prod = 1.0
                valid = True
                for c in comb:
                    s = single_cnt.get(c, 0)
                    if s <= 0:
                        valid = False
                        break
                    p_prod *= s / float(n)
                if not valid or p_prod <= _EPS:
                    continue
                lift = p_joint / p_prod
            else:
                full = set(comb)
                denom_best = 0.0
                for sub in combinations(comb, len(comb) - 1):
                    sub_t = tuple(sorted(sub))
                    sub_cnt = combo_cnt.get(sub_t, 0)
                    if sub_cnt <= 0:
                        continue
                    p_sub = sub_cnt / float(n)
                    new_tokens = full - set(sub_t)
                    if len(new_tokens) != 1:
                        continue
                    new_tok = next(iter(new_tokens))
                    new_cnt = single_cnt.get(new_tok, 0)
                    if new_cnt <= 0:
                        continue
                    p_new = new_cnt / float(n)
                    denom = p_sub * p_new
                    if denom > denom_best:
                        denom_best = denom
                if denom_best <= _EPS:
                    continue
                lift = p_joint / denom_best
            if lift < min_lift:
                continue
            w = p_joint * math.log1p(max(0.0, lift - 1.0))
            if w > 0.0:
                patterns.append((comb, w))
        patterns.sort(key=lambda x: x[1], reverse=True)
        dbg = {'num_baskets': float(n), 'num_patterns': float(len(patterns)), 'min_support': float(min_support), 'min_lift': float(min_lift), 'lift_mode': 1.0}
        return (patterns, dbg)

    def _habit_pattern_scores(self, txs: List[ConceptTransaction], cands: List[str], active_concepts: List[str], level: Optional[int]=None) -> Tuple[Dict[str, float], Dict[str, float]]:
        patterns, dbg = self._mine_habit_patterns(txs, level=level)
        active = set(sorted(set(active_concepts or [])))
        if not patterns or not active:
            return ({c: 0.0 for c in cands}, dbg)
        min_overlap = max(1, int(CONCEPT_HABIT_MIN_ACTIVE_OVERLAP))
        idf_gamma = max(0.0, float(CONCEPT_HABIT_IDF_GAMMA))
        out: Dict[str, float] = {c: 0.0 for c in cands}
        for c in cands:
            cset = set(self._obj2concepts.get(c, []))
            if not cset:
                continue
            sc = 0.0
            for comb, w in patterns:
                pset = set(comb)
                ov = len(pset & active)
                if ov < min_overlap:
                    continue
                missing = pset - active
                if not missing:
                    continue
                covered = missing & cset
                if not covered:
                    continue
                act_ratio = ov / float(len(pset))
                comp_ratio = len(covered) / float(len(missing))
                if idf_gamma > 0.0:
                    avg_idf = sum((self._concept_idf(cid) for cid in covered)) / float(len(covered))
                    idf_gain = avg_idf ** idf_gamma
                else:
                    idf_gain = 1.0
                sc += w * act_ratio * comp_ratio * idf_gain
            out[c] = sc
        dbg = dict(dbg)
        dbg['active_size'] = float(len(active))
        dbg['min_active_overlap'] = float(min_overlap)
        return (out, dbg)

    def score_candidates(self, pred: Optional[str], furn: Optional[str], ctx_tokens: List[str], cands: List[str], *, exclude_group: Optional[str]=None, prev_changed_concepts: Optional[List[str]]=None, level: Optional[int]=None) -> Dict[str, float]:
        pred = (pred or '').strip()
        if not pred or not cands:
            return {c: 0.0 for c in cands}
        txs = self._by_pred.get(pred, [])
        if exclude_group is not None:
            txs = [t for t in txs if t.group != str(exclude_group)]
        if not txs:
            return {c: 1.0 / len(cands) for c in cands}
        mode = CONCEPT_MATCH_MODE
        use_pred_only = mode in {'pred_only', 'predicate_only', 'pred'}
        use_pred_furn = mode in {'pred_furn', 'predicate_furniture', 'pf'}
        use_pred_furn_backoff = mode in {'pred_furn_backoff', 'predicate_furniture_backoff', 'pf_backoff', 'pfb'}
        if use_pred_only:
            used = txs
            use_weight = 1.0
        elif use_pred_furn:
            furn_key = _furniture_key_for(pred, None, furn_now=furn)
            used = [t for t in txs if t.furniture == furn_key]
            use_weight = 1.0
            if not used:
                return {c: 1.0 / len(cands) for c in cands}
        elif use_pred_furn_backoff:
            furn_key = _furniture_key_for(pred, None, furn_now=furn)
            used_furn = [t for t in txs if t.furniture == furn_key]
            coarse_ctx = {t for t in ctx_tokens or [] if str(t).startswith('ctx:') or str(t).startswith('room:')}
            used_furn_ctx = self._filter_by_context(used_furn, coarse_ctx) if used_furn and coarse_ctx else []
            ctx_nmin = max(2, int(max(1, int(CONCEPT_FURN_NMIN)) * 0.5))
            if len(used_furn_ctx) >= ctx_nmin:
                used = used_furn_ctx
                use_weight = 1.0 if len(used_furn_ctx) >= max(1, int(CONCEPT_FURN_NMIN)) else float(CONCEPT_FURN_BACKOFF)
                mode = 'pred_furn_ctx_backoff'
            elif len(used_furn) >= max(1, int(CONCEPT_FURN_NMIN)):
                used = used_furn
                use_weight = 1.0
            elif used_furn:
                used = txs
                use_weight = float(CONCEPT_FURN_BACKOFF)
            else:
                used = txs
                use_weight = float(CONCEPT_FURN_BACKOFF)
        else:
            ctx_set = set(ctx_tokens)
            room_tokens = {t for t in ctx_set if t.startswith('room:')}
            tx_full = self._filter_by_context(txs, ctx_set)
            use_weight = 1.0
            used = tx_full
            if len(tx_full) < self.nmin:
                tx_room = self._filter_by_context(txs, room_tokens)
                if room_tokens and len(tx_room) >= self.nmin:
                    used = tx_room
                    use_weight = self.room_backoff
                else:
                    used = txs
                    use_weight = self.pred_backoff
        sthem_mode = str(CONCEPT_STHEM_MODE).strip().lower()
        use_action_channel = sthem_mode in {'both', 'instruction', 'instr', 'action'}
        use_env_channel = sthem_mode in {'both', 'environment', 'env'}
        if not use_action_channel and (not use_env_channel):
            use_action_channel = True
            use_env_channel = bool(CONCEPT_ENV_ENABLED)
            sthem_mode = 'both' if use_env_channel else 'instruction'
        if not CONCEPT_ENV_ENABLED:
            use_env_channel = False
            if not use_action_channel:
                use_action_channel = True
                sthem_mode = 'instruction'
        concept_freq_action = self._concept_freqs(used) if use_action_channel else {}
        concept_freq_env: Dict[str, float] = {}
        env_mode = 'disabled'
        habit_dbg: Dict[str, float] = {}
        env_impl = str(CONCEPT_ENV_MODE).strip().lower()
        use_habit_env = use_env_channel and env_impl in {'habit', 'pattern', 'habit_pattern'}
        if use_env_channel and (not use_habit_env):
            concept_freq_env, env_mode = self._env_concept_freqs(prev_changed_concepts or [], pred=pred, exclude_group=exclude_group)
        concept_occ_in_cands: Dict[str, int] = {}
        cand_concepts_map: Dict[str, List[str]] = {}
        for c in cands:
            ids = self._obj2concepts.get(c, [])
            cand_concepts_map[c] = ids
            for cid in set(ids):
                concept_occ_in_cands[cid] = concept_occ_in_cands.get(cid, 0) + 1
        n_cands = max(1, len(cands))
        concept_weight: Dict[str, float] = {}
        for cid, occ in concept_occ_in_cands.items():
            if n_cands <= 1:
                local_disc = 1.0
            else:
                local_disc = 1.0 - (float(occ) - 1.0) / max(1.0, float(n_cands - 1))
            local_disc = max(float(CONCEPT_MIN_LOCAL_DISC), local_disc)
            idf = self._concept_idf(cid)
            w = local_disc ** float(CONCEPT_LOCAL_DISC_GAMMA) * idf ** float(CONCEPT_IDF_GAMMA)
            concept_weight[cid] = max(_EPS, w)
        raw_action: Dict[str, float] = {}
        for c in cands:
            concept_ids = cand_concepts_map.get(c, [])
            if not concept_ids:
                raw_action[c] = 0.0
                continue
            raw_action[c] = self._aggregate_concepts(concept_ids, concept_freq_action, concept_weight) * use_weight
        if use_habit_env:
            basket_mode = self._habit_basket_mode(level=level)
            habit_scope = str(CONCEPT_HABIT_SCOPE).strip().lower()
            if habit_scope in {'global', 'all', 'user_global'}:
                habit_txs = list(self._transactions)
                if exclude_group is not None:
                    habit_txs = [t for t in habit_txs if t.group != str(exclude_group)]
            else:
                habit_txs = list(used)
                habit_scope = 'matched'
            if basket_mode in {'trigger_only', 'changed_only', 'trigger'}:
                concept_freq_env, trans_mode = self._env_concept_freqs(prev_changed_concepts or [], pred=pred, exclude_group=exclude_group)
                raw_env = {}
                for c in cands:
                    concept_ids = cand_concepts_map.get(c, [])
                    if not concept_ids:
                        raw_env[c] = 0.0
                        continue
                    raw_env[c] = self._aggregate_concepts(concept_ids, concept_freq_env, concept_weight)
                env_mode = f'trigger_transition:{trans_mode}'
                habit_dbg = {'basket_mode_trigger_only': 1.0, 'scope_global': 1.0 if habit_scope == 'global' else 0.0, 'tx_count': float(len(habit_txs))}
            else:
                raw_env, habit_dbg = self._habit_pattern_scores(habit_txs, cands, prev_changed_concepts or [], level=level)
                env_mode = 'habit'
                habit_dbg = dict(habit_dbg)
                habit_dbg['scope_global'] = 1.0 if habit_scope == 'global' else 0.0
                habit_dbg['tx_count'] = float(len(habit_txs))
        else:
            raw_env: Dict[str, float] = {}
            for c in cands:
                concept_ids = cand_concepts_map.get(c, [])
                if not concept_ids:
                    raw_env[c] = 0.0
                    continue
                raw_env[c] = self._aggregate_concepts(concept_ids, concept_freq_env, concept_weight)
        if use_action_channel and (not use_env_channel):
            wa, we = (1.0, 0.0)
            raw_scores = dict(raw_action)
        elif use_env_channel and (not use_action_channel):
            wa, we = (0.0, 1.0)
            raw_scores = dict(raw_env)
        else:
            wa = max(0.0, float(CONCEPT_ACTION_WEIGHT))
            we = max(0.0, float(CONCEPT_ENV_WEIGHT)) if use_env_channel else 0.0
            if we > 0.0 and sum(raw_env.values()) <= _EPS:
                we = 0.0
            elif we > 0.0:
                env_vals = sorted((float(v) for v in raw_env.values()), reverse=True)
                top1 = env_vals[0] if env_vals else 0.0
                top2 = env_vals[1] if len(env_vals) > 1 else 0.0
                if top1 - top2 < float(CONCEPT_ENV_WEAK_MARGIN):
                    we *= max(0.0, float(CONCEPT_ENV_WEAK_SCALE))
            norm = wa + we
            if norm <= _EPS:
                wa, we = (1.0, 0.0)
            else:
                wa /= norm
                we /= norm
            raw_scores = {c: wa * raw_action.get(c, 0.0) + we * raw_env.get(c, 0.0) for c in cands}
        Z = sum(raw_scores.values())
        if Z <= _EPS:
            self._last_debug = {'sthem_mode': sthem_mode, 'weights': {'action': wa, 'env': we}, 'env_impl': env_impl, 'env_mode': env_mode, 'habit_dbg': habit_dbg, 'prev_changed_concepts': sorted(set(prev_changed_concepts or [])), 'action_prob': {c: 0.0 for c in cands}, 'env_prob': {c: 0.0 for c in cands}, 'combined_prob': {c: 1.0 / len(cands) for c in cands}}
            return {c: 1.0 / len(cands) for c in cands}
        out = {c: raw_scores[c] / Z for c in cands}
        self._last_debug = {'sthem_mode': sthem_mode, 'weights': {'action': wa, 'env': we}, 'match_mode': mode, 'env_mode': env_mode, 'env_impl': env_impl, 'concept_topk': int(CONCEPT_TOPK), 'habit_dbg': habit_dbg, 'prev_changed_concepts': sorted(set(prev_changed_concepts or [])), 'action_prob': {c: round(raw_action.get(c, 0.0), 6) for c in cands}, 'env_prob': {c: round(raw_env.get(c, 0.0), 6) for c in cands}, 'combined_prob': {c: round(out.get(c, 0.0), 6) for c in cands}}
        return out

    def last_debug(self) -> Dict[str, object]:
        return dict(self._last_debug)

@lru_cache(maxsize=4)
def get_concept_history_index(*, history_path: Optional[str]=None, nmin: int=DEFAULT_NMIN, room_backoff: float=DEFAULT_ROOM_BACKOFF, pred_backoff: float=DEFAULT_PRED_BACKOFF) -> ConceptHistoryIndex:
    path = history_path or get_active_history_file()
    return ConceptHistoryIndex(history_path=str(path), nmin=nmin, room_backoff=room_backoff, pred_backoff=pred_backoff)

def reset_concept_history_cache() -> None:
    get_concept_history_index.cache_clear()
    _build_transactions.cache_clear()
    _load_object_to_concepts.cache_clear()
    _load_object_type_map.cache_clear()
__all__ = ['ConceptHistoryIndex', 'get_concept_history_index', 'reset_concept_history_cache']
