from __future__ import annotations
from typing import Dict, Any, List, Tuple
from commands.confidence_scorer import ConfidenceScorer, dominant_context
from config import config as cfg
import inflect
from commands.nlp_parse import extract_parts
from commands.kg_name_resolver import resolve_name
from commands.kg_utils import is_inside_furn as _is_inside_furn, is_on_furn as _is_on_furn
from commands.wordnet_utils import top_k_scores as _top3_scores
from nltk.corpus import wordnet as wn
from experiments.generate_fuzzy_sets import INSIDE_FURN, ON_FURN, SWITCH_FURN, CONTAINER_FURN, ALL_FURN, OBJ_HYPERNYM_MAP, PARENT_MAP_EXTRA
P = inflect.engine()
HYPER_FURN_MAP = {'inside_furniture': INSIDE_FURN, 'on_furniture': ON_FURN, 'switch_furniture': SWITCH_FURN, 'container_furniture': CONTAINER_FURN, 'furniture': ALL_FURN}

def analyse_command_en(sentence: str) -> Dict[str, Any]:
    has_obj_placeholder = '__obj__' in sentence.lower()
    has_furn_placeholder = '__furn__' in sentence.lower()
    vb, part, obj_ph, furn_ph, mods, has_on, has_in, has_off = extract_parts(sentence)
    PRON = {'it', 'that', 'this'}
    vb_lower = (vb or '').lower()
    if obj_ph and obj_ph.lower() in PRON:
        if not furn_ph and vb_lower in {'open', 'close', 'switch', 'switchon', 'switchoff', 'switch_on', 'switch_off'}:
            furn_ph = '__FURN__'
            obj_ph = None
        else:
            obj_ph = '__OBJ__'
    if furn_ph and furn_ph.lower() in PRON:
        furn_ph = '__FURN__'
    if has_obj_placeholder:
        obj_ph = '__OBJ__'
    if has_furn_placeholder:
        furn_ph = '__FURN__'
    if obj_ph and obj_ph in HYPER_FURN_MAP and (not furn_ph):
        furn_ph, obj_ph = (obj_ph, None)
    if has_obj_placeholder:
        obj = {'name': '__OBJ__', 'status': 'placeholder', 'members': []}
    else:
        obj_lower = obj_ph.lower() if obj_ph else ''
        class_tokens = set(OBJ_HYPERNYM_MAP.values()) | set(PARENT_MAP_EXTRA.keys()) | set(PARENT_MAP_EXTRA.values())
        if obj_lower in class_tokens:
            members = []
            for k in OBJ_HYPERNYM_MAP.keys():
                cur = OBJ_HYPERNYM_MAP.get(k)
                if cur == obj_lower:
                    members.append(k)
                    continue
                for _ in range(3):
                    if not cur:
                        break
                    cur = PARENT_MAP_EXTRA.get(cur)
                    if cur == obj_lower:
                        members.append(k)
                        break
            obj = {'name': obj_lower, 'status': 'class', 'members': members}
        else:
            obj = resolve_name(obj_ph)
    if obj['status'] in {'class', 'subclass'} and obj['name']:
        n_lower = str(obj['name']).lower()
        if n_lower in OBJ_HYPERNYM_MAP.values():
            mapped = [k for k, v in OBJ_HYPERNYM_MAP.items() if v == n_lower]
            if not obj['members'] or len(set(obj['members'])) < len(mapped):
                obj['members'] = mapped
    if obj['status'] == 'missing' and mods:
        obj = resolve_name(obj_ph.split()[-1])
    if obj['status'] == 'missing':
        return {'error': f"Object '{obj_ph}' not found."}
    if obj['status'] in {'class', 'subclass'} and mods and obj['members']:
        adj = mods[0].lower()

        def _by_adj(cands: List[str]) -> List[str]:
            out = []
            for name in cands:
                defs = ' '.join((s.definition().lower() for s in wn.synsets(name)))
                if adj in defs:
                    out.append(name)
            return out
        filt = _by_adj(obj['members'])
        if filt:
            obj['members'] = filt
    if furn_ph:
        if furn_ph in HYPER_FURN_MAP:
            furn = {'name': furn_ph, 'status': 'class', 'members': list(HYPER_FURN_MAP[furn_ph])}
        elif furn_ph == '__FURN__':
            furn = {'name': '__FURN__', 'status': 'placeholder', 'members': []}
        else:
            furn = resolve_name(furn_ph, restrict_furn=True)
            if furn['status'] == 'missing':
                return {'error': f"Furniture '{furn_ph}' not found."}
    else:
        furn = {'name': None, 'status': None, 'members': []}
    ambiguous_slot = None
    if obj['status'] in {'missing', 'placeholder'} and furn['status'] not in {'missing', 'placeholder'}:
        ambiguous_slot = 'obj'
    elif furn['status'] in {'missing', 'placeholder'} and obj['status'] not in {'missing', 'placeholder'}:
        ambiguous_slot = 'furn'
    elif obj['status'] in {'missing', 'placeholder'} and furn['status'] in {'missing', 'placeholder'}:
        ambiguous_slot = 'unknown'
    obj_scores = _top3_scores(vb, obj['members']) if obj['members'] else []
    furn_scores = _top3_scores(vb, furn['members']) if furn['members'] else []
    obj_class = None
    if obj['status'] in {'class', 'subclass'} and obj.get('name'):
        obj_class = str(obj['name']).lower()
    return {'verb': vb, 'particle': part, 'has_on': has_on, 'has_in': has_in, 'has_off': has_off, 'object': obj['name'], 'obj_status': obj['status'], 'members': obj['members'], 'obj_ctx_top3': obj_scores, 'obj_class': obj_class, 'ambiguous_slot': ambiguous_slot, 'furniture': furn['name'], 'furn_status': furn['status'], 'furn_members': furn['members'], 'furn_ctx_top3': furn_scores}
