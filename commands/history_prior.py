from __future__ import annotations
from typing import Dict, List, Optional
import math
from pathlib import Path
from kg.history_manager import load_user_history
from kg.history_analyzer import extract_asp_command as hx_extract, extract_keywords as hx_keywords, filter_valid_history_entries
from kg.loader import get_graph, EX

class HistoryPrior:

    def __init__(self, history_file: str, context_weight: float=0.25, min_samples: int=1, debug: bool=True):
        self.history_file = history_file or ''
        self.context_weight = max(0.0, min(context_weight, 1.0))
        self.min_samples = max(1, int(min_samples))
        self.debug = bool(debug)

    def _ctx_weight_for(self, name: str, ctx_label: Optional[str]) -> float:
        if not ctx_label:
            return 0.0
        g = get_graph()
        q = f'\n        PREFIX ex:<http://example.org/>\n        SELECT ?w WHERE {{\n          ex:{name} ex:hasContextWeight ?bn .\n          ?bn ex:context ex:{ctx_label} ;\n              ex:importanceWeight ?w .\n        }}\n        '
        try:
            vals: List[float] = []
            for row in g.query(q):
                w = float(row['w'])
                if w > 0:
                    vals.append(w)
            return max(vals) if vals else 0.0
        except Exception:
            return 0.0

    def _normalize(self, scores: Dict[str, float]) -> Dict[str, float]:
        mx = max(scores.values()) if scores else 0.0
        if mx <= 0:
            return {k: 0.0 for k in scores}
        return {k: v / mx for k, v in scores.items()}

    def _extract_next_object_for_pred(self, pred_now: str, s2: Optional[str], o2: Optional[str]) -> Optional[str]:
        if pred_now in ('inside', 'on', 'has'):
            return o2
        if pred_now in ('heated',):
            return s2
        return o2

    def _load_history_lines(self) -> List[str]:
        if self.history_file:
            p = Path(self.history_file)
            if p.exists():
                try:
                    return p.read_text(encoding='utf-8').splitlines()
                except Exception:
                    if self.debug:
                        print(f'[HIST-PRIOR-DBG] failed to read history_file={p}, fallback to load_user_history()')
        try:
            return load_user_history()
        except Exception:
            return []

    def score_candidates(self, *, candidates: List[str], predicate: str, furniture: Optional[str]=None, prev_pred_current: Optional[str]=None, context_label: Optional[str]=None) -> Dict[str, float]:
        if not candidates:
            return {}
        raw = self._load_history_lines()
        try:
            hist = filter_valid_history_entries(raw)
        except Exception:
            hist = []
        counts = {c: 0 for c in candidates}
        total = 0
        for i in range(len(hist) - 1):
            curr_asp = hx_extract(hist[i])
            next_asp = hx_extract(hist[i + 1])
            if not curr_asp or not next_asp:
                continue
            c_pred, c_s, c_o, _ = hx_keywords(curr_asp)
            n_pred, n_s, n_o, _ = hx_keywords(next_asp)
            if not n_pred:
                continue
            if prev_pred_current and c_pred != prev_pred_current:
                continue
            if n_pred != predicate:
                continue
            if predicate in ('inside', 'on') and furniture:
                if n_s != furniture:
                    continue
            total += 1
            obj_name = self._extract_next_object_for_pred(predicate, n_s, n_o)
            if obj_name in counts:
                counts[obj_name] += 1
        if total > 0:
            base_prob: Dict[str, float] = {c: counts[c] / total for c in candidates}
        else:
            base_prob = {c: 0.0 for c in candidates}
        ctx_raw = {c: self._ctx_weight_for(c, context_label) for c in candidates}
        ctx_norm = self._normalize(ctx_raw)
        w = self.context_weight
        mixed = {c: (1 - w) * base_prob.get(c, 0.0) + w * ctx_norm.get(c, 0.0) for c in candidates}
        for k, v in mixed.items():
            if v < 0:
                mixed[k] = 0.0
            elif v > 1:
                mixed[k] = 1.0
        if self.debug:

            def _r(d: Dict[str, float]) -> Dict[str, float]:
                return {k: round(float(v), 3) for k, v in d.items()}
            print(f'[HIST-PRIOR-DBG] pred={predicate}, furn={furniture}, prev_pred={prev_pred_current}, ctx={context_label}, total={total}')
            print(f'[HIST-PRIOR-DBG]   counts   : {counts}')
            print(f'[HIST-PRIOR-DBG]   base_prob: {_r(base_prob)}')
            print(f'[HIST-PRIOR-DBG]   ctx_norm : {_r(ctx_norm)}')
            print(f'[HIST-PRIOR-DBG]   mixed    : {_r(mixed)}')
        return mixed
