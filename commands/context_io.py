from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
import re
from rdflib.term import Node
from kg.loader import get_graph, EX, XSD
from commands.confidence_scorer import dominant_context
from config.config import CONTEXT_FILE
_CTX_FILE = Path(CONTEXT_FILE)
ROOM_TO_CTX = {'kitchen': 'CookingContext', 'dining_room': 'DiningContext', 'living_room': 'RelaxationContext', 'bedroom': 'RestingContext'}
FURN_TO_CTX = {'microwave': 'CookingContext', 'fridge': 'CookingContext', 'kitchen_table': 'DiningContext', 'kitchen_counter': 'CookingContext', 'bookshelf': 'ReadingContext', 'desk': 'StudyingContext', 'sofa': 'RelaxationContext', 'tv_stand': 'EntertainmentContext', 'dish_bowl': 'DiningContext', 'table_lamp': 'LightingContext'}

def _tail(n: Node | str) -> str:
    s = str(n)
    return s.split('/')[-1] if '/' in s else s

def _ensure_ctx_file_dir():
    _CTX_FILE.parent.mkdir(parents=True, exist_ok=True)

def clear_context_file():
    _ensure_ctx_file_dir()
    _CTX_FILE.write_text('', encoding='utf-8')
    print(f'[CTX] cleared file at {_CTX_FILE}')

def last_context() -> str:
    try:
        if not _CTX_FILE.exists():
            return ''
        lines = _CTX_FILE.read_text(encoding='utf-8').splitlines()
        return lines[-1].strip() if lines else ''
    except Exception:
        return ''
_CHANGED_NAME_RE = re.compile('show_changed_holds_name\\(([^)]+)\\)')

def parse_changed_names_file(path: str) -> List[str]:
    try:
        text = Path(path).read_text(encoding='utf-8')
    except FileNotFoundError:
        print(f'[CTX] changed_names file not found: {path}')
        return []
    names = _CHANGED_NAME_RE.findall(text)
    seen, out = (set(), [])
    for n in names:
        n = n.strip()
        if n and n not in seen:
            seen.add(n)
            out.append(n)
    return out

def _sum_context_weights_for(names: List[str]) -> Dict[str, float]:
    g = get_graph()
    total: Dict[str, float] = {}
    for name in names:
        subj = EX[name]
        for _, _, b in g.triples((subj, EX.hasContextWeight, None)):
            ctx = g.value(b, EX.context)
            w = g.value(b, EX.importanceWeight)
            if ctx is None or w is None:
                continue
            ctx_key = _tail(ctx)
            try:
                w_val = float(str(w)) if getattr(w, 'datatype', None) == XSD.float else float(str(w))
            except Exception:
                continue
            total[ctx_key] = total.get(ctx_key, 0.0) + w_val
    return total

def _fallback_context(names: List[str]) -> Optional[str]:
    votes: Dict[str, int] = {}
    order: List[str] = []
    for n in names:
        ctx = dominant_context(n) or 'UnknownContext'
        if ctx != 'UnknownContext':
            if ctx not in votes:
                votes[ctx] = 0
                order.append(ctx)
            votes[ctx] += 1
    if votes:
        return max(order, key=lambda k: (votes[k], -order.index(k)))
    for n in names:
        low = n.lower()
        if low in FURN_TO_CTX:
            return FURN_TO_CTX[low]
        if low in ROOM_TO_CTX:
            return ROOM_TO_CTX[low]
    return None

def append_context_from_changed_names_file(path: str) -> str:
    names = parse_changed_names_file(path)
    if not names:
        prev = last_context()
        ctx = prev if prev else 'UnknownContext'
        _ensure_ctx_file_dir()
        with _CTX_FILE.open('a', encoding='utf-8') as f:
            f.write(ctx + '\n')
        print(f'[CTX] no changed names; append -> {ctx} (fallback prev)  file={_CTX_FILE}')
        return ctx
    totals = _sum_context_weights_for(names)
    if totals:
        ctx = max(totals.items(), key=lambda kv: kv[1])[0]
    else:
        ctx = _fallback_context(names) or last_context() or 'UnknownContext'
    _ensure_ctx_file_dir()
    with _CTX_FILE.open('a', encoding='utf-8') as f:
        f.write(ctx + '\n')
    print(f"[CTX] append (by changed names) -> {ctx}  file={_CTX_FILE}  names={names}  totals={totals or 'N/A'}")
    return ctx

def clear_context() -> None:
    _ensure_ctx_file_dir()
    _CTX_FILE.write_text('', encoding='utf-8')
