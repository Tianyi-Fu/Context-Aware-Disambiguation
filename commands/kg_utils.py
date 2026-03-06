from __future__ import annotations
from typing import Set, List
from rdflib import Namespace, RDF, RDFS
from kg.loader import get_graph
EX = Namespace('http://example.org/')
G = get_graph()

def is_inside_furn(name: str | None) -> bool:
    return bool(name) and bool(G.query(f'ASK WHERE {{ ex:{name} rdf:type/rdfs:subClassOf* ex:InsideFurniture . }}', initNs={'ex': EX}))

def is_on_furn(name: str | None) -> bool:
    return bool(name) and bool(G.query(f'ASK WHERE {{ ex:{name} rdf:type/rdfs:subClassOf* ex:OnFurniture . }}', initNs={'ex': EX}))

def _inst_of(cls_uri, seen: Set=None) -> List[str]:
    seen = seen or set()
    out = [str(s).split('/')[-1] for s in G.subjects(RDF.type, cls_uri)]
    for sub in G.subjects(RDFS.subClassOf, cls_uri):
        if sub not in seen:
            seen.add(sub)
            out += _inst_of(sub, seen)
    return out

def list_inst(cname: str, limit: int=20) -> List[str]:
    uri = next((c for c in G.subjects(RDF.type, RDFS.Class) if str(c).split('/')[-1].lower() == cname.lower()), None)
    return _inst_of(uri)[:limit] if uri else []
__all__ = ['is_inside_furn', 'is_on_furn', 'list_inst']
