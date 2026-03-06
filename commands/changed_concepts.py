from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

from config.config import PROJECT_ROOT, SHOW_CHANGED_HOLDS_NAME_OUTPUT, SHOW_CHANGED_HOLDS_NAME_OUTPUT_USER
from commands.context_io import parse_changed_names_file

ROOT = Path(PROJECT_ROOT)
CONCEPT_DIR = ROOT / 'data' / 'concepts'
OBJ2CONCEPTS_PATH = CONCEPT_DIR / 'object_to_concepts.json'
DOMAIN_OBJECTS_PATH = CONCEPT_DIR / 'domain_objects.json'
_CHANGED_NAMES_CANDIDATES = [Path(SHOW_CHANGED_HOLDS_NAME_OUTPUT_USER), Path(SHOW_CHANGED_HOLDS_NAME_OUTPUT)]
_FURNITURE_TYPES = {'OnFurniture', 'InsideFurniture', 'Light'}


@lru_cache(maxsize=4)
def _load_obj2concepts(path: str) -> Dict[str, List[str]]:
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


def names_to_concepts(names: List[str], *, include_furniture: bool=False) -> List[str]:
    obj2concepts = _load_obj2concepts(str(OBJ2CONCEPTS_PATH))
    type_map = _load_object_type_map(str(DOMAIN_OBJECTS_PATH))
    out: Set[str] = set()
    for n in names:
        if not include_furniture and type_map.get(n, '') in _FURNITURE_TYPES:
            continue
        for c in obj2concepts.get(n, []):
            out.add(c)
    return sorted(out)


def prev_changed_concepts_current(changed_names_file: Optional[str]=None) -> List[str]:
    names: List[str] = []
    if changed_names_file:
        p = Path(changed_names_file)
        if p.exists() and p.stat().st_size > 0:
            names = parse_changed_names_file(str(p))
    else:
        for p in _CHANGED_NAMES_CANDIDATES:
            if p.exists() and p.stat().st_size > 0:
                names = parse_changed_names_file(str(p))
                if names:
                    break
    if not names:
        return []
    return names_to_concepts(names, include_furniture=False)


def reset_changed_concepts_cache() -> None:
    _load_obj2concepts.cache_clear()
    _load_object_type_map.cache_clear()


__all__ = ['prev_changed_concepts_current', 'names_to_concepts', 'reset_changed_concepts_cache']
