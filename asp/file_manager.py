# asp/file_manager.py
import os
import sys
import re
import logging
from pathlib import Path
from rdflib import URIRef, RDF, RDFS, Literal
from config.config import ASP_FILE, PROJECT_ROOT, INITIAL_CONDITIONS_FILE
from kg.loader import load_kg, save_kg, get_graph, EX, OWL, XSD

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s', stream=sys.stderr)

# Boolean properties and opposites used only in ASP layer.
BOOLEAN_PROPERTIES = {
    "open": {"InsideFurniture", "MicrowaveFurniture"},
    "switched_on": {"SwitchFurniture"},
    "heated": {"HotDrink", "HotFood"},
    "changed": {"InsideFurniture", "MicrowaveFurniture"}
}
OPPOSITE_BOOLEAN_PROPERTIES = {
    "switched_on": "switched_off",
    "open": "closed",
    "closed": "open",
}

def read_asp_file():
    with open(ASP_FILE, "r", encoding="utf-8") as f:
        return f.readlines()

def write_asp_file(lines):
    with open(ASP_FILE, "w", encoding="utf-8") as f:
        f.writelines(lines)

def append_to_asp_file(text):
    with open(ASP_FILE, "a", encoding="utf-8") as f:
        f.write(text)

def clean_existing_goals_and_success():
    """Remove existing goal/success rules while preserving declaration lines."""
    lines = read_asp_file()
    lines_to_keep = [
        "success :- goal_1(I).\n",
        "success().\n",
        "goal_1(#step).\n",
        "goal_2(#step).\n",
        "goal_rollback(#step).\n",
        "goal_furniture_restored(#step).\n"
    ]

    cleaned_lines = [
        line for line in lines if (
            line.strip().startswith('%') or
            line in lines_to_keep or
            not (line.strip().startswith("goal_1") or
                 line.strip().startswith("success") or
                 line.strip().startswith("goal_2") or
                 line.strip().startswith("goal_rollback"))
        )
    ]

    write_asp_file(cleaned_lines)
    print("Cleaned existing goals and success rules from ASP file, while preserving specified lines.")

def add_goal_to_asp_file(goal_condition):
    """Append goal_1 and success rules when they are missing."""
    lines = read_asp_file()
    formatted_goal = f"goal_1(I) :- holds({goal_condition}, I).\n"
    success_rule = "success :- goal_1(I).\n"

    goal_present = any(line.strip() == formatted_goal.strip() for line in lines if not line.strip().startswith('%'))
    success_present = any(line.strip() == success_rule.strip() for line in lines if not line.strip().startswith('%'))

    if not goal_present:
        append_to_asp_file(formatted_goal)
        print(f"Added goal to ASP file: {formatted_goal.strip()}")
    else:
        print("Goal already exists in ASP file; skipping addition.")

    if not success_present:
        append_to_asp_file(success_rule)
        print(f"Added success rule to ASP file: {success_rule.strip()}")
    else:
        print("Success rule already exists in ASP file; skipping addition.")

def add_predicted_goal_to_asp_file(goal_condition_2):
    """Append goal_2 and update success rule to require goal_1 and goal_2."""
    lines = read_asp_file()

    formatted_goal_2 = f"{goal_condition_2}\n"
    updated_success_rule = "success :- goal_1(I), goal_2(I).\n"

    new_asp_lines = []
    for line in lines:
        line_stripped = line.strip()
        if (line_stripped.startswith("%") or
            line_stripped.endswith("(#step).") or
            (not line_stripped.startswith("goal_2") and not line_stripped.startswith("success :-"))):
            new_asp_lines.append(line)

    new_asp_lines.append(formatted_goal_2)
    print(f"Added goal_2 to ASP file: {formatted_goal_2.strip()}")
    new_asp_lines.append(updated_success_rule)
    print(f"Updated success rule in ASP file: {updated_success_rule.strip()}")

    with open(ASP_FILE, "w", encoding="utf-8") as f:
        f.writelines(new_asp_lines)

def get_local_name(uri):
    if isinstance(uri, URIRef):
        name = uri.split("/")[-1]
        return name
    return str(uri)

def get_all_superclasses(g, cls):
    superclasses = set()
    for superclass in g.objects(cls, RDFS.subClassOf):
        local_super = get_local_name(superclass)
        superclasses.add(local_super)
        superclasses.update(get_all_superclasses(g, superclass))
    return superclasses

def _type_chain(g, entity):
    """Return type names of an entity, including ancestor classes."""
    out = set()
    for c in g.objects(entity, RDF.type):
        out.add(get_local_name(c))
        out.update(get_all_superclasses(g, c))
    return out

def is_heatable(g, item):
    types = set(get_local_name(t) for t in g.objects(item, RDF.type))
    return bool(types.intersection({"HotDrink", "HotFood"}))

def _lint_and_fix_ic(ic_lines):
    """Fix contradictory t=0 facts by keeping positive and dropping negative."""
    pos = {}
    neg = {}
    passthrough = []

    atom_re = re.compile(r"^\s*(-)?holds\((.+?),\s*0\)\.\s*$")

    for ln in ic_lines:
        m = atom_re.match(ln)
        if not m:
            passthrough.append(ln)
            continue
        sign, inner = m.groups()
        if sign == "-":
            neg.setdefault(inner, []).append(ln)
        else:
            pos.setdefault(inner, []).append(ln)

    conflicts = set(pos.keys()) & set(neg.keys())
    if conflicts:
        logging.warning(f"[IC-LINT] Found {len(conflicts)} t=0 contradictions, auto-fixing by keeping positive and dropping negative.")

    fixed = []
    fixed.extend(passthrough)
    for _, lines_pos in pos.items():
        fixed.extend(lines_pos)
    for inner, lines_neg in neg.items():
        if inner in conflicts:
            continue
        fixed.extend(lines_neg)

    fixed = sorted(set(fixed))
    return fixed

def extract_initial_conditions():
    """Generate ASP initial conditions from current KG into initial_conditions.txt."""
    g = get_graph()
    g.remove((None, None, None))
    load_kg(os.path.join(PROJECT_ROOT, "kg", "living_room.ttl"))

    ic = set()

    for furn in g.subjects(RDF.type, EX.Furniture):
        fname = get_local_name(furn)
        f_classes = _type_chain(g, furn)
        is_inside = bool(f_classes.intersection({"InsideFurniture", "MicrowaveFurniture"}))
        is_switch = bool(f_classes.intersection({"SwitchFurniture"}))

        room = g.value(furn, EX.furniture_location) or g.value(furn, EX.location)
        if room:
            ic.add(f"holds(furniture_location({fname}, {get_local_name(room)}), 0).")
        else:
            logging.warning(f"Furniture '{fname}' has no defined location.")

        if is_inside:
            open_lit = g.value(furn, EX.open)
            if open_lit is not None:
                is_open = str(open_lit).lower() in {"true", "1"}
            else:
                is_open = False

            if is_open:
                ic.add(f"holds(open({fname}), 0).")
                ic.add(f"-holds(closed({fname}), 0).")
            else:
                ic.add(f"-holds(open({fname}), 0).")
                ic.add(f"holds(closed({fname}), 0).")

            ic.add(f"-holds(changed({fname}), 0).")

        if is_switch:
            switched_on_lit = g.value(furn, EX.switched_on)
            if switched_on_lit is not None:
                is_on = str(switched_on_lit).lower() in {"true", "1"}
                ic.add(f"{'' if is_on else '-'}holds(switched_on({fname}), 0).")
            else:
                ic.add(f"-holds(switched_on({fname}), 0).")

    def _has_location(item_uri):
        """Whether item has on/inside/location triple in KG."""
        return (g.value(item_uri, EX.on) is not None or
                g.value(item_uri, EX.inside) is not None or
                g.value(item_uri, EX.location) is not None)

    for item in g.subjects(RDF.type, EX.Item):
        iname = get_local_name(item)

        if is_heatable(g, item):
            ic.add(f"-holds(heated({iname}), 0).")

        if _has_location(item):
            furn_on = g.value(item, EX.on)
            if furn_on:
                fname = get_local_name(furn_on)
                rname = get_local_name(g.value(furn_on, EX.furniture_location) or g.value(furn_on, EX.location))
                ic.add(f"holds(location({iname}, {rname}, {fname}), 0).")
                ic.add(f"holds(on({fname}, {iname}), 0).")

            # inside(...)
            furn_in = g.value(item, EX.inside)
            if furn_in:
                fname = get_local_name(furn_in)
                rname = get_local_name(g.value(furn_in, EX.furniture_location) or g.value(furn_in, EX.location))
                ic.add(f"holds(location({iname}, {rname}, {fname}), 0).")
                ic.add(f"holds(inside({fname}, {iname}), 0).")
        else:
            logging.warning(f"Item '{iname}' is neither on nor inside any furniture.")

    # Agent/User location and possession facts
    for ag in g.subjects(RDF.type, EX.Agent):
        ag_name = get_local_name(ag)
        if (room := g.value(ag, EX["in"])):
            ic.add(f"holds(in({ag_name}, {get_local_name(room)}), 0).")
        else:
            logging.warning(f"Agent '{ag_name}' lacks an 'in' property.")

    for usr in g.subjects(RDF.type, EX.User):
        if (room := g.value(usr, EX.user_location)):
            ic.add(f"holds(user_location({get_local_name(room)}), 0).")
        else:
            logging.warning("A user instance lacks a user_location.")

    # Write only existing ex:has triples
    for s, _, o in g.triples((None, EX.has, None)):
        ic.add(f"holds(has({get_local_name(s)}, {get_local_name(o)}), 0).")

    # Final consistency pass before writing
    lines = sorted(ic)
    lines = _lint_and_fix_ic(lines)

    with open(INITIAL_CONDITIONS_FILE, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    logging.info("Initial conditions have been successfully extracted and saved to 'initial_conditions.txt'")

# ─────────────────────────────────────────────────────────────
# Insert / remove IC block in ASP file (robust, keep END marker)
# ─────────────────────────────────────────────────────────────
def insert_initial_conditions_to_asp():
    """Replace INITIAL CONDITIONS block by markers; append block if absent."""
    ic = Path(INITIAL_CONDITIONS_FILE).read_text(encoding="utf-8")
    with open(ASP_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    start_marker = "% ===== INITIAL CONDITIONS START ====="
    end_marker   = "% ===== INITIAL CONDITIONS END ====="
    block = f"{start_marker}\n{ic}{end_marker}"

    if start_marker in text and end_marker in text:
        new = re.sub(
            rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}",
            block,
            text,
            flags=re.S
        )
    else:
        new = text.rstrip() + "\n\n" + block + "\n"

    with open(ASP_FILE, "w", encoding="utf-8") as f:
        f.write(new)

    print("Inserted initial conditions into ASP file.")

def remove_initial_conditions_from_asp():
    """Remove IC content while keeping START/END markers."""
    with open(ASP_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    start_marker = "% ===== INITIAL CONDITIONS START ====="
    end_marker   = "% ===== INITIAL CONDITIONS END ====="

    new = re.sub(
        rf"{re.escape(start_marker)}.*?{re.escape(end_marker)}",
        f"{start_marker}\n{end_marker}",
        text,
        flags=re.S
    )

    with open(ASP_FILE, "w", encoding="utf-8") as f:
        f.write(new)

    print("Removed initial conditions from ASP file.")

# ─────────────────────────────────────────────────────────────
# Utilities for holds <-> triples mapping 
# ─────────────────────────────────────────────────────────────
def update_user_holds(user, item, action):
    if action == 'add':
        hold_add = f"holds(has({user}, {item}), 1)."
        hold_remove = f"-holds(has({user}, {item}), 0)."
        return [hold_add, hold_remove]
    elif action == 'remove':
        hold_remove = f"-holds(has({user}, {item}), 0)."
        hold_add = f"holds(has({user}, {item}), 1)."
        return [hold_remove, hold_add]
    else:
        print(f"[WARN] Unknown action '{action}' for updating user holds.")
        return []

def map_hold_to_triples(predicate, subject, obj=None):
    # Skip temporary/internal predicates.
    if predicate in {"hasContextWeight", "changed"}:
        logging.info(f"Skipping predicate '{predicate}' in map_hold_to_triples.")
        return []

    predicate_mapping = {
        "switched_on": EX.switched_on,
        "switched_off": EX.switched_off,
        "has": EX.has,
        "changed": EX.changed,
        "at_furniture": EX.at_furniture,
        "open": EX.open,
        "inside": EX.inside,
        "in": EX.in_,
        "location": EX.location,
        "on": EX.on,
        "holds": EX.holds,
    }

    triples = []
    if predicate in predicate_mapping:
        pred_uri = predicate_mapping[predicate]
        subj_uri = URIRef(f"http://example.org/{subject}")
        if obj:
            obj_uri = URIRef(f"http://example.org/{obj}")
            triples.append((subj_uri, pred_uri, obj_uri))
        else:
            if predicate in {"switched_on", "switched_off", "open"}:
                triples.append((subj_uri, pred_uri, Literal(True, datatype=XSD.boolean)))
            else:
                triples.append((subj_uri, pred_uri, Literal(True)))
    else:
        logging.warning(f"map_hold_to_triples: predicate '{predicate}' not recognized.")
        return None

    return triples
