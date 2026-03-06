from __future__ import annotations
import os
from typing import Tuple, List
import spacy, inflect
from nltk.corpus import wordnet as wn
NLP = spacy.load('en_core_web_sm')
P = inflect.engine()
PREP_ON = {'on', 'onto', 'upon'}
PREP_IN = {'in', 'into', 'inside', 'within'}

def extract_parts(sent: str) -> Tuple[str, str | None, str, str | None, List[str], bool, bool, bool]:
    doc = NLP(sent.lower())
    root = next((t for t in doc if t.head == t))
    particle = next((c.text for c in root.children if c.dep_ == 'prt' and c.text in {'on', 'off'}), None)
    verb_base = root.lemma_
    dobj = next((c for c in root.children if c.dep_ in {'dobj', 'pobj', 'attr'}), None)
    if not dobj:
        nouns = [t for t in doc if t.pos_ in {'NOUN', 'PROPN'}]
        dobj = nouns[-1] if nouns else None
    mods, obj_phrase = ([], '')
    if dobj:
        for w in dobj.lefts:
            if w.dep_ in {'amod', 'compound'}:
                mods.append(w.text)
        obj_phrase = ' '.join([*mods, dobj.text])
    furn_phrase = None
    for prep in doc:
        if prep.dep_ == 'prep' and prep.text in PREP_ON | PREP_IN:
            pobj = next((c for c in prep.children if c.dep_ == 'pobj'), None)
            if pobj:
                furn_phrase = pobj.text
                break
    words = set(sent.lower().split())
    return (verb_base, particle, obj_phrase, furn_phrase, mods, bool(words & PREP_ON), bool(words & PREP_IN), 'off' in words)
