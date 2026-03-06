import os
from rdflib import Graph, Namespace
try:
    from config.config import PROJECT_ROOT, LIVINGROOM_BACKUP_TTL
except Exception:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    LIVINGROOM_BACKUP_TTL = os.path.join(PROJECT_ROOT, 'kg', 'living_room_backup.ttl')
EX = Namespace('http://example.org/')
OWL = Namespace('http://www.w3.org/2002/07/owl#')
XSD = Namespace('http://www.w3.org/2001/XMLSchema#')
g = Graph()
g.bind('ex', EX)
g.bind('owl', OWL)
g.bind('xsd', XSD)

def _abs_path(p: str | None) -> str:
    if not p:
        return LIVINGROOM_BACKUP_TTL
    if os.path.isabs(p):
        return p
    return os.path.join(PROJECT_ROOT, p)

def load_kg(file_path: str | None=None) -> Graph:
    path = _abs_path(file_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f'KG file not found: {path}')
    g.parse(path, format='turtle')
    return g

def save_kg(file_path: str | None=None) -> None:
    path = _abs_path(file_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    g.serialize(destination=path, format='turtle')

def get_graph() -> Graph:
    return g
