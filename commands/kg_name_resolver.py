from __future__ import annotations
from typing import Dict, Any, List, Set, Iterable
import difflib, os
import inflect
from rdflib import Graph, Namespace, RDF, RDFS, URIRef
from config.config import LIVINGROOM_BACKUP_TTL
from commands.kg_utils import is_inside_furn as _is_inside_furn, is_on_furn as _is_on_furn
from kg.loader import load_kg
P = inflect.engine()
EX = Namespace('http://example.org/')
G: Graph = load_kg(LIVINGROOM_BACKUP_TTL)
SIM_TH = float(os.getenv('SIM_TH', 0.45))
BONUS_HEAD = 0.2
BONUS_TOKEN = 0.1

def _local_name(uri: URIRef) -> str:
    return str(uri).split('/')[-1]

def _instances_of(class_uri: URIRef) -> List[str]:
    return [_local_name(s) for s in G.subjects(RDF.type, class_uri)]

def _subclasses_of(class_uri: URIRef) -> List[URIRef]:
    return [c for c in G.subjects(RDFS.subClassOf, class_uri)]

def _is_class(name: str) -> bool:
    uri = EX[name]
    return (uri, RDF.type, RDFS.Class) in G or any(G.triples((uri, RDFS.subClassOf, None)))

def _is_subclass_of(cls: URIRef, target: URIRef) -> bool:
    if cls == target:
        return True
    visited = set()
    stack = [cls]
    while stack:
        cur = stack.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for sup in G.objects(cur, RDFS.subClassOf):
            if sup == target:
                return True
            stack.append(sup)
    return False

def _types_of_instance(name: str) -> List[URIRef]:
    return list(G.objects(EX[name], RDF.type))

def _is_instance_of_any(name: str, supers: Iterable[URIRef]) -> bool:
    types = _types_of_instance(name)
    supers = list(supers)
    for t in types:
        for sup in supers:
            if t == sup or _is_subclass_of(t, sup):
                return True
    return False

def _is_furniture(name: str) -> bool:
    furn = EX.Furniture
    uri = EX[name]
    if any(G.triples((uri, RDF.type, None))):
        return _is_instance_of_any(name, [furn])
    if _is_class(name):
        return _is_subclass_of(uri, furn)
    return False

def _collect_names(parent_cls: str, seen: Set[str] | None=None) -> List[str]:
    seen = seen or set()
    parent = EX[parent_cls]
    out = _instances_of(parent)
    for sub in _subclasses_of(parent):
        sub_name = _local_name(sub)
        if sub_name not in seen:
            seen.add(sub_name)
            out += _collect_names(sub_name, seen)
    return out

def _variants(phrase: str) -> List[str]:
    raw = phrase.strip().lower()
    base = raw.replace(' ', '_')
    tight = raw.replace(' ', '')
    vars_ = {raw, base, tight, P.singular_noun(base) or base, P.plural(base)}
    return [v for v in vars_ if v]

def _head(name: str) -> str:
    return name.split('_')[-1]

def _nearest_candidates(query: str, cands: List[str]) -> List[str]:
    scored: List[tuple[float, str]] = []
    for cand in cands:
        ratio = difflib.SequenceMatcher(None, query, cand).ratio()
        bonus = BONUS_HEAD if _head(cand) == query else BONUS_TOKEN if query in cand.split('_') else 0.0
        score = ratio + bonus
        if score >= SIM_TH:
            scored.append((score, cand))
    scored.sort(reverse=True)
    return [c for _, c in scored]

def resolve_name(phrase: str, *, restrict_furn: bool=False, visited: Set[str] | None=None) -> Dict[str, Any]:
    if not phrase:
        return {'name': None, 'status': None, 'members': []}
    visited = visited or set()
    Item = EX.Item
    Furniture = EX.Furniture

    def _allowed_instance(name: str) -> bool:
        if restrict_furn:
            return _is_instance_of_any(name, [Furniture])
        return _is_instance_of_any(name, [Item, Furniture])
    for v in _variants(phrase):
        if v in visited:
            continue
        visited.add(v)
        if any(G.triples((EX[v], RDF.type, None))) and _allowed_instance(v):
            return {'name': v, 'status': 'unique', 'members': []}

    def _cls_exact(v: str):
        uri = EX[v]
        if _is_class(v):
            tag = 'subclass' if any(G.triples((uri, RDFS.subClassOf, None))) else 'class'
            mem = _collect_names(v)
            if restrict_furn:
                mem = [m for m in mem if _is_furniture(m)]
            return {'name': v, 'status': tag, 'members': mem}
        return None
    for v in _variants(phrase):
        hit = _cls_exact(v)
        if hit:
            return hit
    all_cls = [_local_name(c) for c in G.subjects(RDF.type, RDFS.Class)]
    if restrict_furn:
        all_cls = [c for c in all_cls if _is_furniture(c)]
    for v in _variants(phrase):
        near = difflib.get_close_matches(v, all_cls, n=1, cutoff=SIM_TH)
        if near:
            cls = near[0]
            mem = _collect_names(cls)
            if restrict_furn:
                mem = [m for m in mem if _is_furniture(m)]
            return {'name': cls, 'status': 'subclass', 'members': mem}
    all_items: List[str] = []
    for s, p, o in G.triples((None, RDF.type, None)):
        name = _local_name(s)
        if restrict_furn:
            if _is_instance_of_any(name, [Furniture]):
                all_items.append(name)
        elif _is_instance_of_any(name, [Item, Furniture]):
            all_items.append(name)
    seen = set()
    uniq_items = []
    for n in all_items:
        if n not in seen:
            seen.add(n)
            uniq_items.append(n)
    for v in _variants(phrase):
        cands = list(dict.fromkeys(_nearest_candidates(v, uniq_items)))
        if not cands:
            continue
        if len(cands) == 1:
            return {'name': cands[0], 'status': 'unique', 'members': []}
        return {'name': v.replace(' ', '_'), 'status': 'ambiguous', 'members': cands}

    def _all_candidate_instances() -> List[str]:
        candidates: list[str] = []
        Item = EX.Item
        Furniture = EX.Furniture
        for s, _, _ in G.triples((None, RDF.type, None)):
            name = _local_name(s)
            if restrict_furn:
                if _is_instance_of_any(name, [Furniture]):
                    candidates.append(name)
            elif _is_instance_of_any(name, [Item, Furniture]):
                candidates.append(name)
        seen: set[str] = set()
        uniq: list[str] = []
        for n in candidates:
            if n not in seen:
                seen.add(n)
                uniq.append(n)
        return uniq
    return {'name': phrase.replace(' ', '_'), 'status': 'ambiguous', 'members': _all_candidate_instances()}
__all__ = ['resolve_name', '_is_inside_furn', '_is_on_furn']
