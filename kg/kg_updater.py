import os
import shutil
import re
from rdflib import URIRef, Literal
from rdflib.namespace import RDF, RDFS
from config.config import PROJECT_ROOT, SHOW_CHANGED_HOLDS_NAME_OUTPUT
from kg.loader import load_kg, save_kg, get_graph, EX, OWL, XSD
from commands.context_io import append_context_from_changed_names_file
BACKUP_FILE = os.path.join(PROJECT_ROOT, 'kg', 'living_room_backup.ttl')
LIVINGROOM_TTL = os.path.join(PROJECT_ROOT, 'kg', 'living_room.ttl')
_DEF_CHANGED_NAMES = os.path.basename(SHOW_CHANGED_HOLDS_NAME_OUTPUT)

def _default_changed_names_path() -> str:
    return SHOW_CHANGED_HOLDS_NAME_OUTPUT

def _first_existing_nonempty(paths: list[str]) -> str | None:
    for p in paths:
        if not p:
            continue
        try:
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return p
        except Exception:
            pass
    return None

def _robust_find_changed_names_file(start_holds_file: str | None, last_holds_file: str | None, changed_holds_file: str | None, explicit: str | None=None) -> tuple[str | None, list[str]]:
    tried: list[str] = []
    cands: list[str] = []
    if explicit:
        cands.append(explicit)
    cands.append(_default_changed_names_path())
    for ref in (start_holds_file, last_holds_file, changed_holds_file):
        if not ref:
            continue
        d = os.path.dirname(os.path.abspath(ref))
        cands.append(os.path.join(d, _DEF_CHANGED_NAMES))
    seen_dirs = set()
    for ref in (start_holds_file, last_holds_file, changed_holds_file):
        if not ref:
            continue
        d = os.path.dirname(os.path.abspath(ref))
        if d in seen_dirs:
            continue
        seen_dirs.add(d)
        try:
            for name in os.listdir(d):
                low = name.lower()
                if low.startswith('risk__'):
                    continue
                if 'changed_holds_name' in low and name.endswith('.txt'):
                    cands.append(os.path.join(d, name))
        except Exception:
            pass
    tried = cands[:]
    hit = _first_existing_nonempty(cands)
    return (hit, tried)

def map_hold_to_triples(hold, action):
    predicate_pattern = '(\\w+)\\(([^)]+)\\)'
    match = re.match(predicate_pattern, hold)
    if not match:
        print(f"[WARN] map_hold_to_triples: could not parse '{hold}'")
        return []
    predicate = match.group(1).strip()
    args = [arg.strip() for arg in match.group(2).split(',')]
    if predicate == 'hasContextWeight':
        print(f"[INFO] Skipping update for predicate '{predicate}'.")
        return []
    triples = []
    if predicate == 'on':
        if len(args) != 2:
            print(f"[WARN] on predicate expects 2 arguments, got {len(args)} in '{hold}'")
            return []
        subject = URIRef(EX + args[1])
        obj = URIRef(EX + args[0])
        triples.append((subject, EX.on, obj))
    elif predicate == 'inside':
        if len(args) != 2:
            print(f"[WARN] inside predicate expects 2 arguments, got {len(args)} in '{hold}'")
            return []
        subject = URIRef(EX + args[1])
        obj = URIRef(EX + args[0])
        triples.append((subject, EX.inside, obj))
    if predicate == 'location':
        if len(args) == 3:
            item_name = args[0]
            furniture_name = args[2]
            item_uri = URIRef(EX + item_name)
            furniture_uri = URIRef(EX + furniture_name)
            triple_list = []
            if furniture_name.lower() in ['microwave', 'fridge', 'bookshelf']:
                triple_list.append((item_uri, EX.inside, furniture_uri))
            else:
                triple_list.append((item_uri, EX.on, furniture_uri))
            if action == 'ADD':
                return [('ADD', t) for t in triple_list]
            elif action == 'REMOVE':
                return [('REMOVE', t) for t in triple_list]
            else:
                print(f"[WARN] Unknown action '{action}' for hold '{hold}'")
                return []
        elif len(args) == 2:
            pass
    elif predicate == 'has':
        if len(args) != 2:
            print(f"[WARN] has predicate expects 2 arguments, got {len(args)} in '{hold}'")
            return []
        subject = URIRef(EX + args[0])
        obj = URIRef(EX + args[1])
        triples.append((subject, EX.has, obj))
    if predicate == 'open':
        if len(args) != 1:
            return []
        subj = URIRef(EX + args[0])
        triple = (subj, EX.open, Literal(True, datatype=XSD.boolean))
        if action == 'ADD':
            return [('ADD', triple)]
        elif action == 'REMOVE':
            return [('REMOVE', triple)]
        else:
            return []
    elif predicate == 'heated':
        if len(args) != 1:
            print(f"[WARN] heated predicate expects 1 argument, got {len(args)} in '{hold}'")
            return []
        subject = URIRef(EX + args[0])
        triples.append((subject, EX.heated, Literal(True, datatype=XSD.boolean)))
    elif predicate == 'in':
        if len(args) != 2:
            print(f"[WARN] in predicate expects 2 arguments, got {len(args)} in '{hold}'")
            return []
        subject = URIRef(EX + args[0])
        obj = URIRef(EX + args[1])
        triples.append((subject, EX['in'], obj))
    elif predicate == 'switched_on':
        if len(args) != 1:
            print(f"[WARN] switched_on predicate expects 1 argument, got {len(args)} in '{hold}'")
            return []
        subject = URIRef(EX + args[0])
        triples.append((subject, EX.switched_on, Literal(True, datatype=XSD.boolean)))
    elif predicate == 'switched_off':
        if len(args) != 1:
            print(f"[WARN] switched_off predicate expects 1 argument, got {len(args)} in '{hold}'")
            return []
        subject = URIRef(EX + args[0])
        triples.append((subject, EX.switched_on, Literal(False, datatype=XSD.boolean)))
    elif predicate == 'user_location':
        if len(args) != 1:
            print(f"[WARN] user_location predicate expects 1 argument, got {len(args)} in '{hold}'")
            return []
        subject = URIRef(EX + args[0])
        triples.append((subject, EX.user_location, Literal(True)))
    elif predicate == 'at_user':
        if len(args) != 1:
            print(f"[WARN] at_user predicate expects 1 argument, got {len(args)} in '{hold}'")
            return []
        subject = URIRef(EX + args[0])
        triples.append((subject, EX.at_user, Literal(True)))
    elif predicate == 'closed':
        if len(args) != 1:
            return []
        subj = URIRef(EX + args[0])
        triple = (subj, EX.open, Literal(False, datatype=XSD.boolean))
        if action == 'ADD':
            return [('ADD', triple)]
        elif action == 'REMOVE':
            return [('REMOVE', triple)]
        else:
            return []
    if action == 'ADD':
        return [('ADD', triple) for triple in triples]
    elif action == 'REMOVE':
        return [('REMOVE', triple) for triple in triples]
    else:
        print(f"[WARN] Unknown action '{action}' for hold '{hold}'")
        return []

def revert_kg_to_backup():
    if not os.path.exists(BACKUP_FILE):
        print(f'[WARN] Backup file not found at: {BACKUP_FILE}. Skipping revert.')
        return
    try:
        shutil.copyfile(BACKUP_FILE, LIVINGROOM_TTL)
        print('[INFO] KG has been reverted to backup state.')
        load_kg(LIVINGROOM_TTL)
    except Exception as e:
        print(f'[ERROR] Error reverting KG: {e}')

def parse_multi_holds(file_path, outer_pattern='show_last_holds\\(([^)]*)\\)'):
    holds = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace('{', '').replace('}', '')
        blocks = re.findall(outer_pattern, content)
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            parts = re.split('\\)\\s*,', block)
            for p in parts:
                p = p.strip()
                if not p.endswith(')'):
                    p += ')'
                if p:
                    holds.add(p.strip())
    except FileNotFoundError:
        print(f'[WARN] {file_path} not found. (parse_multi_holds)')
    return holds

def remove_conflicting_triples(g, triple):
    s, p, o = triple
    if p == EX.hasContextWeight:
        return
    if p == EX.switched_on and o == Literal(True, datatype=XSD.boolean):
        false_triple = (s, EX.switched_on, Literal(False, datatype=XSD.boolean))
        if false_triple in g:
            g.remove(false_triple)
            print('[CONFLICT-DEL]', false_triple)
    elif p == EX.switched_on and o == Literal(False, datatype=XSD.boolean):
        true_triple = (s, EX.switched_on, Literal(True, datatype=XSD.boolean))
        if true_triple in g:
            g.remove(true_triple)
            print('[CONFLICT-DEL]', true_triple)
    if p == EX.open and o == Literal(True, datatype=XSD.boolean):
        conflict = (s, EX.open, Literal(False, datatype=XSD.boolean))
        if conflict in g:
            g.remove(conflict)
            print('[CONFLICT-DEL]', conflict)
    elif p == EX.open and o == Literal(False, datatype=XSD.boolean):
        conflict = (s, EX.open, Literal(True, datatype=XSD.boolean))
        if conflict in g:
            g.remove(conflict)
            print('[CONFLICT-DEL]', conflict)
    if p == EX['in']:
        for old_s, old_p, old_o in g.triples((s, EX['in'], None)):
            if old_o != o:
                g.remove((old_s, old_p, old_o))
                print('[CONFLICT-DEL]', (old_s, old_p, old_o))
    if p == EX.inside:
        for old_s, old_p, old_o in g.triples((None, EX.location, s)):
            g.remove((old_s, old_p, old_o))
            print('[CONFLICT-DEL] Removed conflicting location:', (old_s, old_p, old_o))
    if p == EX.location:
        for old_s, old_p, old_o in g.triples((s, EX.inside, None)):
            g.remove((old_s, old_p, old_o))
            print('[CONFLICT-DEL] Removed conflicting inside:', (old_s, old_p, old_o))

def update_kg_from_asp_outputs(start_holds_file, last_holds_file, touched_items: list[str] | None=None, changed_holds_file=None, changed_names_file=None, auto_find_changed_names: bool=True):
    load_kg(LIVINGROOM_TTL)
    g = get_graph()
    if not os.path.exists(BACKUP_FILE):
        shutil.copyfile(LIVINGROOM_TTL, BACKUP_FILE)
        print('[INFO] Backup created.')
    start_pattern = 'show_start_holds\\(([^)]*)\\)'
    last_pattern = 'show_last_holds\\(([^)]*)\\)'
    start_holds = parse_multi_holds(start_holds_file, outer_pattern=start_pattern)
    last_holds = parse_multi_holds(last_holds_file, outer_pattern=last_pattern)
    start_holds = {h for h in start_holds if not h.startswith('hasContextWeight(')}
    last_holds = {h for h in last_holds if not h.startswith('hasContextWeight(')}
    added_holds = last_holds - start_holds
    removed_holds = start_holds - last_holds
    print(f'[INFO] added_holds = {added_holds}')
    print(f'[INFO] removed_holds = {removed_holds}')
    for hold in added_holds:
        print(f'[INFO] Processing added hold: {hold}')
        action_triples = map_hold_to_triples(hold, 'ADD')
        for action, triple in action_triples:
            if action == 'ADD':
                print(f'[INFO] Mapped triples: {triple}')
                remove_conflicting_triples(g, triple)
                if triple not in g:
                    g.add(triple)
                    print('[ADD]', triple)
                else:
                    print('[INFO] Triple already exists, skipping:', triple)
    for hold in removed_holds:
        print(f'[INFO] Processing removed hold: {hold}')
        action_triples = map_hold_to_triples(hold, 'REMOVE')
        for action, triple in action_triples:
            if action == 'REMOVE':
                if triple in g:
                    g.remove(triple)
                    print('[DEL] Removed triple:', triple)
                else:
                    print('[INFO] Triple does not exist, skipping removal:', triple)
    remove_duplicate_context_weights(g)
    save_kg(LIVINGROOM_TTL)
    print(f'[INFO] KG updated => {LIVINGROOM_TTL}')
    if added_holds or removed_holds:
        hit_path = None
        tried = []
        if auto_find_changed_names:
            hit_path, tried = _robust_find_changed_names_file(start_holds_file=start_holds_file, last_holds_file=last_holds_file, changed_holds_file=changed_holds_file, explicit=changed_names_file)
        else:
            hit_path = changed_names_file
            tried = [changed_names_file] if changed_names_file else []
        if hit_path:
            print(f'[INFO] Using changed_names_file: {hit_path}')
            ctx = append_context_from_changed_names_file(hit_path)
            print(f'[CTX] context <- {ctx}')
        else:
            if tried:
                print('[INFO] tried changed_names_file at:')
                for p in tried:
                    print('   -', p)
            print('[INFO] No changed_names_file provided/found  context not updated (as required).')
    else:
        print('[INFO]  KG context.txt')

def remove_duplicate_context_weights(graph):
    seen = {}
    for subject, predicate, blank_node in list(graph.triples((None, EX.hasContextWeight, None))):
        context = graph.value(blank_node, EX.context)
        weight = graph.value(blank_node, EX.importanceWeight)
        key = (subject, str(context), float(weight) if weight and weight.datatype == XSD.float else None)
        if key in seen:
            graph.remove((subject, EX.hasContextWeight, blank_node))
            for triple in list(graph.triples((blank_node, None, None))):
                graph.remove(triple)
        else:
            seen[key] = blank_node
