import logging
import re
from difflib import SequenceMatcher
from rdflib import URIRef
from nltk.corpus import wordnet as wn
from kg.history_manager import load_user_history
from kg.loader import get_graph, EX
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def extract_asp_command(history_entry):
    if '|' not in history_entry:
        return ''
    return history_entry.split('|', 1)[0].strip()

def extract_keywords(asp_command):
    pattern_2 = '(\\w+)\\(([^,]+),\\s*([^,]+)(?:,\\s*([^)]+))?\\)'
    m2 = re.match(pattern_2, asp_command)
    if m2:
        predicate = m2.group(1).strip()
        arg1 = m2.group(2).strip()
        arg2 = m2.group(3).strip()
        arg3 = m2.group(4).strip() if m2.group(4) else None
        return (predicate, arg1, arg2, arg3)
    pattern_1 = '(\\w+)\\(([^)]+)\\)'
    m1 = re.match(pattern_1, asp_command)
    if m1:
        predicate = m1.group(1).strip()
        arg1 = m1.group(2).strip()
        return (predicate, arg1, None, None)
    return (None, None, None, None)

def find_similar_history(current_asp_command, history, threshold=0.8):
    sim_list = []
    for entry in history:
        past_asp = extract_asp_command(entry)
        if not past_asp:
            continue
        ratio = SequenceMatcher(None, current_asp_command.lower(), past_asp.lower()).ratio()
        if ratio >= threshold:
            sim_list.append(entry)
    return sim_list

def predict_next_from_history(similar_history, full_history):
    potential_next = []
    for entry in similar_history:
        if entry in full_history:
            idx = full_history.index(entry)
            if idx + 1 < len(full_history):
                nxt = full_history[idx + 1]
                potential_next.append(nxt)
    if potential_next:
        freq_map = {}
        for p in potential_next:
            freq_map[p] = freq_map.get(p, 0) + 1
        logging.debug(f'[DEBUG] potential_next freq => {freq_map}')
        best = max(freq_map, key=freq_map.get)
        logging.info(f'Predicted next entry: {best}')
        return best
    return None

def determine_type_in_kg(name: str):
    g = get_graph()
    query = f'\n    PREFIX ex: <{EX}>\n    SELECT ?t WHERE {{\n      ex:{name} a ?t .\n    }}\n    '
    try:
        rows = g.query(query)
        types = [str(r.t).split('/')[-1] for r in rows]
        if 'Item' in types:
            return 'Item'
        elif 'Furniture' in types:
            return 'Furniture'
        else:
            return None
    except Exception as e:
        logging.error(f"Error in determine_type_in_kg('{name}'): {e}")
        return None

def measure_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def get_all_entities_by_type():
    g = get_graph()
    items = []
    q_items = f'\n    PREFIX ex: <{EX}>\n    SELECT ?x WHERE {{ ?x a ex:Item . }}\n    '
    for row in g.query(q_items):
        item_name = str(row.x).split('/')[-1]
        items.append(item_name)
    furns = []
    q_furn = f'\n    PREFIX ex: <{EX}>\n    SELECT ?x WHERE {{ ?x a ex:Furniture . }}\n    '
    for row in g.query(q_furn):
        furn_name = str(row.x).split('/')[-1]
        furns.append(furn_name)
    return (items, furns)

def find_top_similar_in_kg(target_word, expected_type, top_n=10):
    all_items, all_furns = get_all_entities_by_type()
    if expected_type == 'Item':
        candidates = all_items
    elif expected_type == 'Furniture':
        candidates = all_furns
    else:
        return []
    scored = []
    for c in candidates:
        sim = measure_similarity(c, target_word)
        scored.append((c, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]

def get_filtered_properties(name):
    relevant_props = {'on', 'inside', 'location', 'heated', 'open', 'switched_on', 'changed'}
    info = {}
    g = get_graph()
    query = f'\n    PREFIX ex: <{EX}>\n    SELECT ?p ?o WHERE {{\n      ex:{name} ?p ?o .\n      FILTER(?p IN (\n        ex:on, ex:inside, ex:location, ex:heated,\n        ex:open, ex:switched_on, ex:changed\n      ))\n    }}\n    '
    try:
        rows = g.query(query)
        for r in rows:
            prop = r.p.split('/')[-1]
            val = str(r.o)
            if isinstance(r.o, URIRef):
                val = val.split('/')[-1]
            if prop in relevant_props:
                info[prop] = val
    except Exception as e:
        logging.error(f"Error get_filtered_properties('{name}'): {e}")
    return info

def format_properties(name, props_dict):
    if not props_dict:
        return f'{name}: (no relevant props)'
    parts = []
    for p in ['on', 'inside', 'location', 'heated', 'open', 'switched_on', 'changed']:
        if p in props_dict:
            parts.append(f'{p}={props_dict[p]}')
    return f'{name}: ' + ', '.join(parts) if parts else f'{name}: (no relevant props)'

def filter_valid_history_entries(entries):
    valid = []
    for entry in entries:
        if entry.startswith('==='):
            continue
        if entry.startswith('[') or entry.startswith(']'):
            continue
        if '|' not in entry:
            continue
        valid.append(entry)
    return valid

def analyze_and_predict_next(current_asp_command):
    raw_history = load_user_history()
    history = filter_valid_history_entries(raw_history)
    logging.info(f'Loaded {len(history)} valid history entries')
    sim_list = find_similar_history(current_asp_command, history)
    logging.info(f'Found {len(sim_list)} similar history entries')
    predicted_entry = predict_next_from_history(sim_list, history)
    if not predicted_entry:
        logging.info('No predicted next entry => let LLM decide.')
        return (None, {})
    predicted_asp = extract_asp_command(predicted_entry)
    if not predicted_asp:
        logging.warning('predicted_entry found but no valid asp_command => skip expansions.')
        return (predicted_entry, {})
    predicate, s, o, arg3 = extract_keywords(predicted_asp)
    if not predicate:
        logging.warning(f"Cannot parse '{predicted_asp}', skip expansions.")
        return (predicted_entry, {})
    expansions = {}
    if s:
        stype = determine_type_in_kg(s)
        if stype in ('Item', 'Furniture'):
            expansions[s] = stype
    if o:
        otype = determine_type_in_kg(o)
        if otype in ('Item', 'Furniture'):
            expansions[o] = otype
    if not expansions:
        logging.info('No item/furniture found in predicted_asp => skip expansions.')
        return (predicted_entry, {})
    expanded_dict = {}
    for name, typ in expansions.items():
        top_similars = find_top_similar_in_kg(name, typ, top_n=10)
        detail_map = {}
        for ent, score in top_similars:
            props = get_filtered_properties(ent)
            props_str = format_properties(ent, props)
            detail_map[ent] = {'score': score, 'props_str': props_str}
        expanded_dict[name] = detail_map
    context_info = {'expanded': expanded_dict}
    return (predicted_entry, context_info)
