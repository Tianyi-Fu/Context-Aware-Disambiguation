from __future__ import annotations
from typing import Dict, List
from nltk.corpus import wordnet as wn
from kg.loader import get_graph, EX
from kg.history_manager import load_user_history, extract_asp_command
from config import config as cfg
import json

def _wn_sim(ctx1: str, ctx2: str) -> float:
    if not ctx1 or not ctx2:
        return 0.0
    c1, c2 = (ctx1.replace('Context', '').lower(), ctx2.replace('Context', '').lower())
    if c1 == c2:
        return 1.0
    s1, s2 = (wn.synsets(c1, wn.NOUN), wn.synsets(c2, wn.NOUN))
    if not s1 or not s2:
        return 0.0
    sim = s1[0].path_similarity(s2[0])
    return sim or 0.0

def dominant_context(item: str) -> str:
    g = get_graph()
    uri = EX[item]
    best_tag, best_w = ('', -1.0)
    for b in g.objects(uri, EX.hasContextWeight):
        tag_node = g.value(b, EX.context)
        w_node = g.value(b, EX.importanceWeight)
        if not tag_node or not w_node:
            continue
        tag = str(tag_node).split('/')[-1]
        try:
            w = float(str(w_node))
        except ValueError:
            continue
        if w > best_w:
            best_w = w
            best_tag = tag
    return best_tag or 'UnknownContext'
_REPEAT_POOL: List[str] = []

def set_repeat_pool(pool: List[str]):
    global _REPEAT_POOL
    _REPEAT_POOL = list(pool)
_REPEAT_HISTORY_PATH: str | None = None

def set_repeat_history_file(path: str | None):
    global _REPEAT_HISTORY_PATH
    _REPEAT_HISTORY_PATH = path

class ConfidenceScorer:

    def __init__(self):
        self.w1, self.w2, self.w3 = cfg.CONF_WEIGHTS

    def ctx_similarity(self, ctx1: str, ctx2: str | None) -> float:
        if not ctx2:
            return 0.0
        return _wn_sim(ctx1, ctx2)

    def c_cost(self, steps: int) -> float:
        return min(steps / cfg.MAX_STEPS_1COST, 1.0)

    def _repeat_hits(self, asp_cmd: str) -> int:
        hits = 0
        target = asp_cmd.strip()
        if _REPEAT_HISTORY_PATH:
            hist = load_user_history(_REPEAT_HISTORY_PATH)
        else:
            hist = load_user_history()
        hist = hist[-cfg.HISTORY_WINDOW:]
        for l in hist:
            if '|' not in l:
                continue
            cmd = extract_asp_command(l) or ''
            if cmd.strip() == target:
                hits += 1
        for cmd in _REPEAT_POOL:
            if cmd.strip() == target:
                hits += 1
        return hits

    def c_semantic(self, curr_ctx: str, prev_ctx: str | None) -> float:
        if not prev_ctx:
            return 0.0
        return _wn_sim(curr_ctx, prev_ctx)

    def score(self, *, steps: int, asp_cmd: str, curr_ctx: str, prev_ctx: str | None='') -> Dict:
        cost_norm = self.c_cost(steps)
        cost_score = 1.0 - cost_norm
        hits = self._repeat_hits(asp_cmd)
        sem_sim = self.c_semantic(curr_ctx, prev_ctx)
        conf = self.w1 * cost_score + self.w2 * hits + self.w3 * sem_sim
        conf = min(conf, 1.0)
        confidence_expr = f'conf = {self.w1}*cost({cost_norm:.3f}->score={cost_score:.3f}) + {self.w2}*hits({hits}) + {self.w3}*semSim({sem_sim:.3f}) = {conf:.3f}'
        detail = {'steps': steps, 'cost_norm': round(cost_norm, 3), 'cost_score': round(cost_score, 3), 'repeat_hits': hits, 'semantic_sim': round(sem_sim, 3), 'semantic_diff': round(1 - sem_sim, 3), 'curr_ctx': curr_ctx, 'prev_ctx': prev_ctx or '', 'weights': cfg.CONF_WEIGHTS, 'confidence': round(conf, 3), 'confidence_expr': confidence_expr}
        return {'confidence': conf, 'cost': cost_norm, 'cost_score': cost_score, 'repeat_hits': hits, 'semantic': sem_sim, 'semantic_diff': 1 - sem_sim, 'pass_threshold': conf >= cfg.CONF_THRESHOLD, 'detail': json.dumps(detail, ensure_ascii=False, indent=2), 'confidence_expr': confidence_expr}
