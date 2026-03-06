import argparse
import re
import json
from pathlib import Path
from typing import List, Tuple, Dict
from rdflib import Namespace, RDF
from kg.loader import get_graph
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / 'fuzzy_sets'
OUT_DIR.mkdir(exist_ok=True)
MULTI_USER_DIR = HERE / 'multi_user_dataset'
EX = Namespace('http://example.org/')
G = get_graph()
ATTR_PROPS = [EX.color, EX.temperature, EX.taste, EX.texture, EX.genre]

def _label(node) -> str:
    s = str(node)
    return s.split('/')[-1] if '/' in s else s

def _get_entity_attr_values(name: str) -> List[str]:
    subj = EX[name]
    vals: List[str] = []
    try:
        for p in ATTR_PROPS:
            for o in G.objects(subj, p):
                vals.append(str(o).lower())
    except Exception:
        pass
    return vals

def _entity_hypernym_from_kg(name: str) -> str:
    subj = EX[name]
    try:
        types = list(G.objects(subj, RDF.type))
    except Exception:
        types = []
    for t in types:
        ln = _label(t).lower()
        if ln not in {'item', 'object', 'thing'}:
            return ln
    return name
OBJ_HYPERNYM_MAP: Dict[str, str] = {'cupcake': 'snack', 'crackers': 'snack', 'cereal': 'snack', 'pound_cake': 'snack', 'chicken': 'meal', 'cutlets': 'meal', 'apple': 'fruit', 'bananas': 'fruit', 'peach': 'fruit', 'espresso': 'hot_drink', 'coffee': 'hot_drink', 'milk': 'hot_drink', 'juice': 'cold_drink', 'alcohol': 'cold_drink', 'classic_novel': 'novel', 'sci_fi_novel': 'novel', 'fantasy_novel': 'novel', 'superhero_comic': 'comic', 'graphic_memoir': 'comic', 'computer_science_textbook': 'study_item', 'physics_textbook': 'study_item', 'folder': 'study_item', 'notes': 'study_item', 'magazine': 'study_item', 'cellphone': 'study_item', 'mug': 'drinkware', 'cup': 'drinkware', 'water_glass': 'drinkware', 'plate': 'tableware'}
PARENT_MAP_EXTRA: Dict[str, str] = {'fruit': 'food', 'snack': 'food', 'meal': 'food', 'hot_food': 'food', 'hot_drink': 'drink', 'cold_drink': 'drink', 'drink': 'item', 'study_item': 'item', 'novel': 'book', 'comic': 'book', 'book': 'item', 'drinkware': 'item', 'tableware': 'item', 'food': 'item'}
OBJ_PARENTS_FULL: Dict[str, str] = {}
OBJ_PARENTS_FULL.update(OBJ_HYPERNYM_MAP)
OBJ_PARENTS_FULL.update(PARENT_MAP_EXTRA)
OBJ_TOKENS = set(OBJ_HYPERNYM_MAP.keys()) | set(OBJ_HYPERNYM_MAP.values())
OBJ_LEAVES = set(OBJ_HYPERNYM_MAP.keys())
HOT_DRINK_ITEMS = {k for k, v in OBJ_HYPERNYM_MAP.items() if v == 'hot_drink'}
HOT_FOOD_ITEMS = {k for k, v in OBJ_HYPERNYM_MAP.items() if v == 'hot_food'}
INSIDE_FURN = {'bookshelf', 'fridge', 'microwave'}
ON_FURN = {'sofa', 'kitchen_table', 'desk', 'desk_1', 'kitchen_counter', 'coffee_table', 'dish_bowl', 'tv_stand', 'audio_amplifier'}
LIGHT = {'table_lamp'}
SWITCH_FURN = {'microwave'} | LIGHT
CONTAINER_FURN = INSIDE_FURN | ON_FURN
ALL_FURN = CONTAINER_FURN | SWITCH_FURN
PHRASE_ALIAS: Dict[str, str] = {'water glass': 'water_glass', 'table lamp': 'table_lamp', 'kitchen table': 'kitchen_table', 'coffee table': 'coffee_table', 'tv stand': 'tv_stand', 'audio amplifier': 'audio_amplifier', 'dish bowl': 'dish_bowl', 'computer-science textbook': 'computer_science_textbook', 'computer science textbook': 'computer_science_textbook', 'physics textbook': 'physics_textbook', 'classic novel': 'classic_novel', 'sci-fi novel': 'sci_fi_novel', 'sci fi novel': 'sci_fi_novel', 'fantasy novel': 'fantasy_novel', 'superhero comic': 'superhero_comic', 'graphic memoir': 'graphic_memoir', 'pound cake': 'pound_cake'}

def normalize_phrases(nl: str) -> str:
    s = nl
    for k, v in PHRASE_ALIAS.items():
        s = re.sub(f'\\b{re.escape(k)}\\b', v, s, flags=re.IGNORECASE)
    return s

def load_groups(path: Path) -> List[List[Tuple[str, str]]]:
    text = path.read_text(encoding='utf-8')
    groups: List[List[Tuple[str, str]]] = []
    cur: List[Tuple[str, str]] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw or raw.startswith('%'):
            continue
        if raw.startswith('['):
            if cur:
                groups.append(cur)
                cur = []
            continue
        if raw.startswith(']'):
            continue
        if '|' in raw:
            pred, nl = [x.strip() for x in raw.split('|', 1)]
            cur.append((pred, nl))
    if cur:
        groups.append(cur)
    return groups

def is_multi_arg(nl_lower: str) -> bool:
    if re.search('\\bput\\b.*\\b(into|inside|on)\\b', nl_lower):
        return True
    if re.search('\\b(into|inside)\\b', nl_lower):
        return True
    if re.search('\\bswitch\\s+(on|off)\\b', nl_lower):
        return False
    return False

def furniture_hypernym(token: str, nl_lower: str) -> str:
    if re.search('\\bswitch\\s+(on|off)\\b', nl_lower):
        return 'switch_furniture'
    if re.search('\\b(open|close)\\b', nl_lower):
        return 'inside_furniture'
    if re.search('\\bon\\b', nl_lower):
        return 'on_furniture'
    if re.search('\\b(into|inside)\\b', nl_lower):
        return 'inside_furniture'
    return 'furniture'

def furniture_hypernym_k(token: str, nl_lower: str, k: int) -> str:
    _ = k
    return furniture_hypernym(token, nl_lower)

def obj_hypernym(token: str) -> str:
    tok = token.lower()
    seen = set()
    while tok in OBJ_PARENTS_FULL and tok not in seen:
        seen.add(tok)
        tok = OBJ_PARENTS_FULL[tok]
    return tok

def obj_hypernym_k(token: str, k: int) -> str:
    tok = token.lower()
    seen = set([tok])
    for _ in range(k):
        if tok not in OBJ_PARENTS_FULL or OBJ_PARENTS_FULL[tok] in seen:
            break
        tok = OBJ_PARENTS_FULL[tok]
        seen.add(tok)
    return tok

def fix_articles(s: str) -> str:
    s = re.sub('\\ban\\s+([^aeiou\\W])', 'a \\1', s)
    s = re.sub('\\ba\\s+([aeiou])', 'an \\1', s)
    return s

def fuzzy_hypernym(nl: str) -> str:
    nl_orig = nl
    nl_norm = normalize_phrases(nl)
    nl_lower = nl_norm.lower()
    multi = is_multi_arg(nl_lower)
    toks = nl_norm.split()
    changed = False
    for i, t in enumerate(toks):
        bare = re.sub('[^\\w_]', '', t).lower()
        if bare in OBJ_TOKENS:
            toks[i] = obj_hypernym(bare)
            changed = True
            break
        if not multi and bare in ALL_FURN:
            toks[i] = furniture_hypernym(bare, nl_lower)
            changed = True
            break
    if not changed:
        return nl_orig
    out = ' '.join(toks)
    return fix_articles(out)

def fuzzy_hypernym_k(nl: str, k: int) -> str:
    nl_orig = nl
    nl_norm = normalize_phrases(nl)
    nl_lower = nl_norm.lower()
    multi = is_multi_arg(nl_lower)
    toks = nl_norm.split()
    changed = False
    for i, t in enumerate(toks):
        bare = re.sub('[^\\w_]', '', t).lower()
        if bare not in OBJ_TOKENS:
            if not multi and bare in ALL_FURN:
                toks[i] = furniture_hypernym_k(bare, nl_lower, k)
                changed = True
                break
            continue
        toks[i] = obj_hypernym_k(bare, k)
        changed = True
        break
    if not changed:
        return nl_orig
    return fix_articles(' '.join(toks))

def fuzzy_pronoun(nl: str) -> str:
    nl_norm = normalize_phrases(nl)
    toks = nl_norm.split()
    for i, t in enumerate(toks):
        bare = re.sub('[^\\w_]', '', t).lower()
        if bare in OBJ_TOKENS or bare in ALL_FURN:
            toks[i] = 'it'
            out = ' '.join(toks)
            out = re.sub('\\b(?:a|an|the)\\s+(it|this|that)\\b', '\\1', out, flags=re.IGNORECASE)
            return fix_articles(out)
    return nl

def fuzzy_attribute(nl: str) -> str:
    nl_orig = nl
    nl_norm = normalize_phrases(nl)
    toks = nl_norm.split()
    changed = False
    for i, t in enumerate(toks):
        bare = re.sub('[^\\w_]', '', t).lower()
        if bare not in OBJ_HYPERNYM_MAP.keys():
            continue
        attrs = _get_entity_attr_values(bare)
        if not attrs:
            continue
        adj = sorted(attrs)[0]
        cat = _entity_hypernym_from_kg(bare)
        if cat in {'item', 'object', 'thing'}:
            cat = obj_hypernym(bare)
        cat_phrase = cat.replace('_', ' ')
        phrase = f'{adj} {cat_phrase}'
        toks[i] = phrase
        changed = True
        break
    if not changed:
        return nl_orig
    out = ' '.join(toks)
    return fix_articles(out)
LEVELS = [('l1', 'hypernym_l1'), ('l2', 'hypernym_l2'), ('l3', 'hypernym_top'), ('l4', 'pronoun')]
L2_AVOID_ITEM_USERS = {'user4_scholar'}
FUZZ_ALL_BUT_FIRST = True
FORCE_AMBIGUOUS_FALLBACK = False

def apply_fuzzy_method(nl: str, method: str, user_id: str | None=None) -> tuple[str, bool, str | None]:
    if method == 'hypernym':
        new_nl = fuzzy_hypernym(nl)
    elif method == 'hypernym_l1':
        new_nl = fuzzy_hypernym_k(nl, 1)
    elif method == 'hypernym_l2':
        new_nl = fuzzy_hypernym_k(nl, 2)
        if user_id in L2_AVOID_ITEM_USERS and re.search('\\bitem\\b', new_nl, flags=re.IGNORECASE):
            l1_nl = fuzzy_hypernym_k(nl, 1)
            if l1_nl != nl:
                new_nl = l1_nl
    elif method == 'hypernym_top':
        new_nl = fuzzy_hypernym(nl)
    elif method == 'pronoun':
        new_nl = fuzzy_pronoun(nl)
    elif method == 'attribute':
        new_nl = nl
    else:
        new_nl = nl
    if new_nl != nl:
        return (new_nl, True, None)
    if FORCE_AMBIGUOUS_FALLBACK and method != 'pronoun':
        fallback_nl = fuzzy_pronoun(nl)
        if fallback_nl != nl:
            return (fallback_nl, True, 'pronoun')
    return (new_nl, False, None)

def generate_for_user(history_path: Path, user_id: str, log: list):
    groups = load_groups(history_path)
    if not groups:
        print(f'[WARN] No groups found in {history_path}')
        return
    print(f"[INFO] Generating fuzzy sets for user '{user_id}' from {history_path} ({len(groups)} groups)")
    for level, method in LEVELS:
        fname = OUT_DIR / f'{user_id}_{level}.txt'
        with fname.open('w', encoding='utf-8') as fout:
            for g_idx, grp in enumerate(groups, start=1):
                fout.write(f'[  % {g_idx}\n')
                n = len(grp)
                idxs = range(2, n + 1) if FUZZ_ALL_BUT_FIRST else range(1, n + 1)
                for i, (pred, nl) in enumerate(grp, start=1):
                    if i in idxs:
                        new_nl, changed, fallback = apply_fuzzy_method(nl, method, user_id=user_id)
                        log.append({'user': user_id, 'round': 0, 'group': g_idx, 'line': i, 'level': level, 'method': method, 'type': level, 'original': nl, 'fuzzy': new_nl, 'predicate': pred, 'fallback': fallback, 'changed': bool(changed)})
                        fout.write(f'{pred} | {new_nl}\n')
                    else:
                        fout.write(f'{pred} | {nl}\n')
                fout.write(']\n\n')
        print(f'[INFO] Generated {fname}')

def main():
    parser = argparse.ArgumentParser(description='Generate fuzzy sets from *_history.txt files.')
    parser.add_argument('--users', nargs='*', help='Optional list of user ids (e.g., user2_caffeine user3_comfort).')
    args = parser.parse_args()
    if not MULTI_USER_DIR.exists():
        print(f'[ERROR] multi_user_dataset directory not found: {MULTI_USER_DIR}')
        return
    history_files = sorted(MULTI_USER_DIR.glob('*_history.txt'))
    if args.users:
        wanted = {u.strip() for u in args.users if u.strip()}
        history_files = [p for p in history_files if p.stem.replace('_history', '') in wanted]
    if not history_files:
        print(f'[ERROR] No *_history.txt files found under {MULTI_USER_DIR}')
        return
    log = []
    for hpath in history_files:
        stem = hpath.stem
        if stem.endswith('_history'):
            user_id = stem[:-len('_history')]
        else:
            user_id = stem
        generate_for_user(hpath, user_id, log)
    log_path = OUT_DIR / 'fuzz_log.json'
    with log_path.open('w', encoding='utf-8') as lf:
        json.dump(log, lf, ensure_ascii=False, indent=2)
    print(f'[INFO] Wrote fuzz_log.json with {len(log)} records at {log_path}')
if __name__ == '__main__':
    main()
