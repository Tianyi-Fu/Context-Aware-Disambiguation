from __future__ import annotations
from typing import List, Set, Optional
from rdflib import Graph, RDF, Literal, Namespace
from kg.loader import get_graph
from commands.context_io import last_context
from config.config import PROJECT_ROOT
EX = Namespace('http://example.org/')
_GENERIC_TYPES = {'Thing', 'Item', 'Furniture', 'ContainerFurniture', 'Context', 'Agent', 'Room'}

def _tail(uri) -> str:
    s = str(uri)
    return s.split('/')[-1] if '/' in s else s

def _ctx_label_token(label: str | None) -> Optional[str]:
    if not label:
        return None
    s = label.strip()
    if not s or s.lower() == 'unknowncontext':
        return None
    if s.endswith('Context'):
        s = s[:-len('Context')]
    s = s.strip().lower()
    if not s:
        return None
    return f'ctx:{s}'

def _room_from_agent(g: Graph) -> Optional[str]:
    for room in g.objects(EX.agent1, EX['in']):
        return _tail(room)
    return None

def _furniture_in_room(g: Graph, room: Optional[str]) -> List[str]:
    if not room:
        return []
    out: List[str] = []
    for subj in g.subjects(RDF.type, EX.Furniture):
        loc = g.value(subj, EX.furniture_location)
        if loc is None:
            continue
        if _tail(loc) != room:
            continue
        out.append(_tail(subj))
    return out

def _furniture_types(g: Graph, furn_name: str) -> Set[str]:
    types: Set[str] = set()
    subj = EX[furn_name]
    for cls in g.objects(subj, RDF.type):
        t = _tail(cls)
        if t and t not in _GENERIC_TYPES:
            types.add(t)
    return types

def _bool_true(g: Graph, subj, pred) -> bool:
    for val in g.objects(subj, pred):
        if isinstance(val, Literal):
            try:
                if bool(val.toPython()):
                    return True
            except Exception:
                continue
    return False

def extract_context_tokens_from_graph(g: Graph, *, context_label: Optional[str]=None, exclude_items_on_furn: bool=True) -> List[str]:
    tokens: Set[str] = set()
    room = _room_from_agent(g)
    if room:
        tokens.add(f'room:{room}')
    ctx_tok = _ctx_label_token(context_label)
    if ctx_tok:
        tokens.add(ctx_tok)
    furns = _furniture_in_room(g, room)
    for furn in furns:
        for t in _furniture_types(g, furn):
            tokens.add(f'near:{t}')
    for furn in furns:
        subj = EX[furn]
        if _bool_true(g, subj, EX.open):
            tokens.add('state:open')
        if _bool_true(g, subj, EX.switched_on):
            tokens.add('state:switched_on')
    _ = exclude_items_on_furn
    return sorted(tokens)

def extract_context_tokens_current(*, exclude_items_on_furn: bool=True) -> List[str]:
    g = get_graph()
    label = last_context()
    return extract_context_tokens_from_graph(g, context_label=label, exclude_items_on_furn=exclude_items_on_furn)
__all__ = ['extract_context_tokens_from_graph', 'extract_context_tokens_current']
