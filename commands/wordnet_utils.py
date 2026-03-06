from __future__ import annotations
import difflib
import os
from functools import lru_cache
from typing import List, Tuple, Set
import requests
from nltk.corpus import wordnet as wn

def wn_similarity(a: str, b: str) -> float:
    a = (a or '').strip().lower()
    b = (b or '').strip().lower()
    if not a or not b:
        return 0.0
    syn_a = wn.synsets(a)
    syn_b = wn.synsets(b)
    if not syn_a or not syn_b:
        return 0.0
    best = 0.0
    for s1 in syn_a:
        for s2 in syn_b:
            sim = s1.wup_similarity(s2)
            if sim and sim > best:
                best = sim
    return float(best)
CN_API_ROOT = os.getenv('CONCEPTNET_BASE_URL') or os.getenv('CONCEPTNET_API_ROOT') or 'http://127.0.0.1:8084'
NUMBERBATCH_PATH = os.getenv('NUMBERBATCH_PATH')
USE_NUMBERBATCH_ONLY = os.getenv('USE_NUMBERBATCH_ONLY', '').lower() in {'1', 'true', 'yes'}
_last_cn_source = 'none'

def _autodetect_offline_path() -> str | None:
    candidates = []
    here = os.path.abspath(os.path.dirname(__file__))
    candidates.append(os.path.join(here, '..', 'conceptnet_en.pkl'))
    candidates.append(os.path.abspath(os.path.join(os.getcwd(), 'conceptnet_en.pkl')))
    candidates.append(os.path.expanduser('~/Downloads/ConceptNet-offline-API-master/assertions_english.pkl'))
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None
CN_OFFLINE_PATH = os.getenv('CONCEPTNET_OFFLINE_PATH') or _autodetect_offline_path()

def _autodetect_numberbatch_path() -> str | None:
    candidates = []
    here = os.path.abspath(os.path.dirname(__file__))
    candidates.append(os.path.join(here, '..', 'numberbatch-en-19.08.txt'))
    candidates.append(os.path.abspath(os.path.join(os.getcwd(), 'numberbatch-en-19.08.txt')))
    for p in candidates:
        if os.path.isfile(p):
            return os.path.abspath(p)
    return None
if not NUMBERBATCH_PATH:
    NUMBERBATCH_PATH = _autodetect_numberbatch_path()

@lru_cache(maxsize=1)
def _get_conceptnet_offline():
    if not CN_OFFLINE_PATH:
        return None
    try:
        from conceptNet import ConceptNet
    except Exception as e:
        print(f'[CN-OFFLINE] import error: {e!r}')
        return None
    try:
        print(f'[CN-OFFLINE] loading database from {CN_OFFLINE_PATH!r} (this may take a while)...')
        return ConceptNet(CN_OFFLINE_PATH, language='english', save_language=False)
    except Exception as e:
        print(f'[CN-OFFLINE] load error: {e!r}')
        return None

def _cn_similarity_offline(a: str, b: str) -> float:
    db = _get_conceptnet_offline()
    if db is None:
        return -1.0
    try:
        res = []
        for s, e in ((a, b), (b, a)):
            q = db.get_query(start=[s], end=[e], relation=None)
            df = getattr(q, 'df', None)
            if df is None and hasattr(q, 'get_raw_dataframe'):
                df = q.get_raw_dataframe()
            if df is not None:
                res.append(df)
        if res:
            import pandas as _pd
            edges_df = _pd.concat(res, ignore_index=True)
        else:
            edges_df = None
    except Exception as e:
        print(f'[CN-OFFLINE] query error a={a!r} b={b!r} err={e!r}')
        return -1.0
    if edges_df is not None and (not edges_df.empty):
        total = 0.0
        if 'weight' in edges_df.columns:
            try:
                total = float(edges_df['weight'].clip(lower=0).sum())
            except Exception:
                total = 0.0
        if total <= 0.0:
            total = float(len(edges_df))
        if total > 0.0:
            score = total / (total + 10.0)
            if score < 0.0:
                score = 0.0
            if score > 1.0:
                score = 1.0
            return float(score)
    try:
        df_a_start = db.get_query(start=[a], end=None, relation=None)
        df_a_end = db.get_query(start=None, end=[a], relation=None)
        df_b_start = db.get_query(start=[b], end=None, relation=None)
        df_b_end = db.get_query(start=None, end=[b], relation=None)
    except Exception as e:
        print(f'[CN-OFFLINE] neighbor query error a={a!r} b={b!r} err={e!r}')
        return -1.0

    def _df(obj):
        if hasattr(obj, 'df') and obj.df is not None:
            return obj.df
        if hasattr(obj, 'get_raw_dataframe'):
            try:
                return obj.get_raw_dataframe()
            except Exception:
                return None
        return None

    def _parse_term(node: str) -> str:
        try:
            parts = node.strip('/').split('/')
            if len(parts) >= 3:
                return parts[2]
        except Exception:
            pass
        return node
    neighbors_a = set()
    neighbors_b = set()
    df_as = _df(df_a_start)
    if df_as is not None and (not df_as.empty):
        neighbors_a.update((_parse_term(x) for x in df_as.get('end', []) if isinstance(x, str)))
    df_ae = _df(df_a_end)
    if df_ae is not None and (not df_ae.empty):
        neighbors_a.update((_parse_term(x) for x in df_ae.get('start', []) if isinstance(x, str)))
    df_bs = _df(df_b_start)
    if df_bs is not None and (not df_bs.empty):
        neighbors_b.update((_parse_term(x) for x in df_bs.get('end', []) if isinstance(x, str)))
    df_be = _df(df_b_end)
    if df_be is not None and (not df_be.empty):
        neighbors_b.update((_parse_term(x) for x in df_be.get('start', []) if isinstance(x, str)))
    if not neighbors_a or not neighbors_b:
        return 0.0
    common = neighbors_a & neighbors_b
    if not common:
        return 0.0
    union = neighbors_a | neighbors_b
    score = len(common) / (len(union) + 5.0)
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0
    return float(score)
_nb_cache = None

def _load_numberbatch():
    global _nb_cache
    if _nb_cache is not None:
        return _nb_cache
    path = NUMBERBATCH_PATH
    if not path:
        return None
    if not os.path.isfile(path):
        print(f'[NB] file not found: {path}')
        _nb_cache = None
        return None

    def _nb_norm(w: str) -> str:
        w = (w or '').strip().lower()
        if w.startswith('/c/'):
            parts = w.split('/')
            if len(parts) >= 3:
                w = parts[3] if parts[0] == '' else parts[2]
        w = w.replace(' ', '_')
        return w
    try:
        import numpy as np
    except Exception as e:
        print(f'[NB] numpy import failed: {e!r}')
        _nb_cache = None
        return None
    try:
        if path.endswith('.npz'):
            data = np.load(path, allow_pickle=True)
            if 'words' in data and 'vectors' in data:
                words = [w if isinstance(w, str) else str(w) for w in data['words']]
                vecs = data['vectors']
                idx = {}
                for i, w in enumerate(words):
                    k = _nb_norm(w)
                    if k and k not in idx:
                        idx[k] = i
                _nb_cache = ('array', idx, vecs)
                print(f'[NB] loaded npz words/vectors from {path} (n={len(idx)})')
                return _nb_cache
            if 'arr_0' in data and hasattr(data['arr_0'], 'item'):
                obj = data['arr_0'].item()
                if isinstance(obj, dict):
                    _nb_cache = ('dict', {_nb_norm(k): np.array(v, dtype=float) for k, v in obj.items() if _nb_norm(k)})
                    print(f'[NB] loaded npz dict from {path} (n={len(_nb_cache[1])})')
                    return _nb_cache
        vecs = {}
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                word_raw, vals = (parts[0], parts[1:])
                word = _nb_norm(word_raw)
                if not word:
                    continue
                try:
                    vec = np.asarray([float(x) for x in vals], dtype=float)
                except Exception:
                    continue
                vecs[word] = vec
        if vecs:
            _nb_cache = ('dict', vecs)
            print(f'[NB] loaded text vectors from {path} (n={len(vecs)})')
        else:
            print(f'[NB] no vectors parsed from {path}')
            _nb_cache = None
    except Exception as e:
        print(f'[NB] load error from {path}: {e!r}')
        _nb_cache = None
    return _nb_cache

def _nb_similarity(a: str, b: str) -> float:
    cache = _load_numberbatch()
    if cache is None:
        return -1.0
    try:
        import numpy as np
    except Exception:
        return -1.0
    if cache[0] == 'array':
        _, idx, mat = cache
        ia = idx.get(a)
        ib = idx.get(b)
        if ia is None or ib is None:
            return -1.0
        va = mat[ia]
        vb = mat[ib]
    else:
        _, vecs = cache
        va = vecs.get(a)
        vb = vecs.get(b)
        if va is None or vb is None:
            return -1.0
        va = np.asarray(va, dtype=float)
        vb = np.asarray(vb, dtype=float)
    try:
        dot = float(np.dot(va, vb))
        na = float(np.linalg.norm(va)) or 1e-12
        nb = float(np.linalg.norm(vb)) or 1e-12
        sim = dot / (na * nb)
        return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))
    except Exception:
        return -1.0

@lru_cache(maxsize=4096)
def cn_similarity(a: str, b: str, timeout: float=0.5) -> float:
    a = (a or '').strip().lower().replace(' ', '_')
    b = (b or '').strip().lower().replace(' ', '_')
    if not a or not b:
        global _last_cn_source
        _last_cn_source = 'none'
        return 0.0
    val = wn_similarity(a, b)
    _last_cn_source = 'wordnet'
    return float(val or 0.0)

def get_last_cn_source() -> str:
    return _last_cn_source

def word_forms(verb: str, particle: str | None=None) -> Set[str]:
    verb = (verb or '').strip().lower()
    out: Set[str] = set()
    if not verb:
        return out
    base = verb
    out.add(base)
    out.add(base + 's')
    out.add(base + 'ed')
    out.add(base + 'ing')
    if particle:
        p = particle.strip().lower()
        if p:
            out.add(f'{base}_{p}')
            out.add(f'{base}s_{p}')
            out.add(f'{base}ed_{p}')
            out.add(f'{base}ing_{p}')
    return out

def top_k_scores(word: str, cands: List[str], k: int=3) -> List[Tuple[str, float]]:
    word = (word or '').strip().lower()
    scored: List[Tuple[str, float]] = []
    for c in cands:
        s = wn_similarity(word, c)
        scored.append((c, s))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]
