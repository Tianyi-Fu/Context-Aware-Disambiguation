import json
from typing import List, Tuple, Dict, Optional
import re
import openai
from config.config import OPENAI_API_KEY, LLM_MODEL
from commands.confidence_scorer import ConfidenceScorer, dominant_context
from commands.wordnet_utils import wn_similarity
scorer = ConfidenceScorer()
openai.api_key = OPENAI_API_KEY
NEAR_TAGS_TOPK: int = 3
USE_WN_SIM: bool = True
ASK_FACTOR_TEMP: float = 0.2
ASK_NEAR_TAGS_TEMP: float = 0.2
LLM_ASK_MIN_CAND: int = 8
LLM_CONF_DEFAULT: str = 'high'
LLM_REQUIRE_EXPLICIT_ASK: bool = True
LLM_REQUIRE_UNCERTAIN_REASON: bool = True

def _ask_llm_json(prompt: str, retries: int=2, temperature: float=0.0) -> Optional[Dict]:
    for _ in range(retries):
        resp = openai.chat.completions.create(model=LLM_MODEL, response_format={'type': 'json_object'}, messages=[{'role': 'system', 'content': 'You are a home robot.'}, {'role': 'user', 'content': prompt}], temperature=temperature).choices[0].message.content.strip()
        try:
            return json.loads(resp)
        except json.JSONDecodeError:
            prompt += '\n\nREMINDER: reply ONLY valid JSON!'
    return None
_ws = re.compile('\\s+')

def _tokens(s: str) -> List[str]:
    return [t for t in _ws.split((s or '').strip().lower()) if t]

def _head(name: str) -> str:
    s = (name or '').strip().lower().replace('-', '_')
    return s.split('_')[-1] if '_' in s else s

def _is_head_preserving(tag: str, head: str) -> bool:
    ts = _tokens(tag)
    return head in ts

def _short_two_word_phrase(tag: str) -> bool:
    return 1 <= len(_tokens(tag)) <= 2

def _normalize(s: str) -> str:
    return ' '.join(_tokens(s))
_NEAR_TAGS_BAN = {'hot', 'warm', 'cold', 'cool', 'steamed', 'frothed', 'foamed', 'whipped', 'cup', 'mug', 'glass', 'vessel', 'container', 'drink', 'beverage', 'liquid', 'bodily', 'fluid', 'nutrition', 'nourishment', 'microwave_safe', 'heat_safe', 'reheat_safe'}

def _ask_near_tags(head: str, kb_class: str, topk: int=NEAR_TAGS_TOPK) -> List[str]:
    prompt = f"""Generate short, near-surface variants for a concrete kitchen item.\nRules:\n- The head token is '{head}'. EVERY variant MUST include this head token exactly.\n- Each variant must be 12 words; concise and realistic: subtype, roast/strength/size modifier, recipe ratio (if common), or serving-style that preserves identity.\n- DO NOT use temperature words (hot, warm, cold), preparation foam/steam words (frothed, foamed, steamed, whipped), container words (cup, mug, glass, container), or generic categories (beverage, drink, liquid, fluid).\n- Avoid brand names, medical/biological terms, and long phrases.\n- Return JSON only.\n\nItem head: {head}\nClass: {kb_class}\n\nReply JSON: {{"variants": ["v1","v2","v3","v4","v5"]}}"""
    data = _ask_llm_json(prompt, temperature=ASK_NEAR_TAGS_TEMP) or {}
    variants = data.get('variants', [])
    if not isinstance(variants, list):
        variants = []
    out: List[str] = []
    seen = set()
    for v in variants:
        if not isinstance(v, str):
            continue
        s_norm = _normalize(v)
        if not s_norm:
            continue
        if not _is_head_preserving(s_norm, head):
            continue
        if not _short_two_word_phrase(s_norm):
            continue
        toks = set(_tokens(s_norm))
        if toks & _NEAR_TAGS_BAN:
            continue
        if s_norm not in seen:
            seen.add(s_norm)
            out.append(s_norm)
        if len(out) >= topk:
            break
    return out

def _fallback_near_tags(head: str) -> List[str]:
    if head == 'milk':
        return ['skim milk', 'whole milk', 'lowfat milk']
    if head == 'espresso':
        return ['double espresso', 'short espresso', 'ristretto']
    return [f'{head} classic', f'{head} small', f'{head} strong']

def _get_near_tags_for_candidate(name: str, kb_class: str, topk: int=NEAR_TAGS_TOPK) -> List[str]:
    head = _head(name)
    tags = _ask_near_tags(head, kb_class, topk=topk)
    if not tags:
        tags = _fallback_near_tags(head)[:topk]
    if head not in ' '.join(tags):
        tags = [*tags, head]
    seen, out = (set(), [])
    for t in tags:
        t2 = _normalize(t)
        if t2 and t2 not in seen:
            seen.add(t2)
            out.append(t2)
    return out[:topk]

def _string_match_sim(q: str, t: str) -> float:
    qn, tn = (_normalize(q), _normalize(t))
    if not qn or not tn:
        return 0.0
    if qn == tn:
        return 1.0
    if qn in tn or tn in qn:
        return 0.6
    return 0.0

def _best_token_sim(q: str, tag: str) -> float:
    qtok = _normalize(q)
    best = 0.0
    for tok in _tokens(tag):
        base = _string_match_sim(qtok, tok)
        if USE_WN_SIM:
            wn = wn_similarity(qtok, tok) or 0.0
            base = max(base, wn)
        if base > best:
            best = base
    return max(0.0, min(best, 1.0))

def _max_sim_to_tags(queries: List[str], tags: List[str]) -> float:
    mx = 0.0
    for q in queries:
        for t in tags:
            sim = _best_token_sim(q, t)
            if sim > mx:
                mx = sim
    return mx

def _guess_factor_category(f: str) -> str:
    f = _normalize(f)
    if not f:
        return 'UNKNOWN'
    if f in {'heated', 'unheated', 'sweet', 'bitter', 'decaf'}:
        return 'STATE/ATTRIBUTE'
    if f in {'dairy', 'espresso', 'tea', 'juice'}:
        return 'CATEGORY'
    if f in {'microwave_safe', 'heat_safe', 'container'}:
        return 'SAFETY/CONTAINER'
    if f in {'ceramic', 'glass', 'paper', 'metal'}:
        return 'MATERIAL'
    if f in {'breakfast', 'bedtime'}:
        return 'CONTEXT/INTENT'
    if f in {'size', 'small', 'large'}:
        return 'ATTRIBUTE'
    return 'OTHER'

def ask_factor(previous_context: str, user_utterance: str, obj_class: str, candidates: List[str], *, prev_failure: bool=False, extra_ban_words: Optional[List[str]]=None, recent_commands: Optional[List[str]]=None, recent_line_id: Optional[str]=None, cand_state_map: Optional[Dict[str, Dict[str, bool]]]=None, try_categories_to_avoid: Optional[List[str]]=None) -> Dict:
    hist_block = ''
    if recent_commands:
        show_hist = '\n'.join((f'- {h}' for h in recent_commands[-12:]))
        hist_block = f'\n## Recent correct commands in this group (most recent last)\n{show_hist}\n'
    cand_lines = []
    for c in candidates[:12]:
        near = _get_near_tags_for_candidate(c, obj_class, topk=NEAR_TAGS_TOPK)
        cand_lines.append(f'- {c}: [{', '.join(near)}]')
    cand_block = '\n'.join(cand_lines) if cand_lines else '(no candidates)'
    ban_line = ''
    if extra_ban_words:
        uniq_ban = sorted(set((_normalize(w) for w in extra_ban_words if w)))
        if uniq_ban:
            ban_line = 'Do NOT repeat any of these previously tried terms (or close paraphrases): ' + ', '.join(uniq_ban) + '\n'
    cat_line = ''
    if try_categories_to_avoid:
        cats = ', '.join(sorted(set(try_categories_to_avoid)))
        cat_line = f'Previously tried factor categories: {cats}. Choose a DIFFERENT category/angle this time.\n'
    fail_hint = 'Previous factor did not clearly separate candidates. Try a different angle.\n' if prev_failure else ''
    prompt = f'You help disambiguate which concrete instance the user meant **for the given action**.\nPropose ONE short factor (one word or short bigram) that best separates the intended target from other candidates.\nThe factor can be a STATE, CATEGORY, MATERIAL, FUNCTION, TYPICAL CONTEXT (e.g., breakfast), USER INTENT (e.g., bedtime),\nATTRIBUTE (e.g., sweet), or SAFETY/USAGE (e.g., microwave_safe). Avoid long phrases; keep it compact.\n{fail_hint}{ban_line}{cat_line}\nPrevious context: {previous_context or 'UnknownContext'}\nUser utterance: "{user_utterance}"\nClass: {obj_class}\n{hist_block}Candidates (with comparable tags):\n{cand_block}\n\nReply JSON only:\n{{"factor":"<word-or-bigram>","synonyms":["w1","w2"],"reason":"<why this fits and separates>"}}'
    print('\n[FACTOR] PROMPT:')
    print(prompt)
    if extra_ban_words:
        print('[FACTOR] banned terms:', sorted(set((_normalize(x) for x in extra_ban_words if x))))
    if try_categories_to_avoid:
        print('[FACTOR] category_hint:', sorted(set(try_categories_to_avoid)))
    data = _ask_llm_json(prompt, temperature=ASK_FACTOR_TEMP) or {}
    fac = _normalize(str(data.get('factor', '')).strip())
    syns = data.get('synonyms', [])
    if not isinstance(syns, list):
        syns = []
    syns = [_normalize(str(x)) for x in syns if isinstance(x, str)]
    reason = str(data.get('reason', '')).strip()
    print('\n[FACTOR] RAW JSON RESPONSE:')
    print(json.dumps({'factor': fac, 'synonyms': syns, 'reason': reason}, ensure_ascii=False))
    return {'factor': fac, 'synonyms': syns, 'reason': reason}

def pick_by_factor(user_sentence: str, kb_class: str, candidates: List[str], *, prev_context: Optional[str]=None, rounds: int=3, gap_threshold: float=0.05, abs_threshold: float=0.04, user_prefs: Optional[Dict[str, str]]=None, recent_commands: Optional[List[str]]=None, recent_line_id: Optional[str]=None, cand_state_map: Optional[Dict[str, Dict[str, bool]]]=None, feedback_logfile: Optional[str]=None, allow_llm_direct_fallback: bool=True, priors: Optional[Dict[str, float]]=None, predicate: Optional[str]=None, furniture: Optional[str]=None) -> Dict:
    if not candidates:
        return {'ok': False, 'pick': None, 'margin': 0.0, 'scores': [], 'factor': '', 'synonyms': [], 'reason': 'no candidates', 'hint': 'no-candidates', 'try_logs': []}
    PRIOR_LAMBDA = 0.6
    ACTION_CAT_WEIGHT = {'heated': {'good': {'STATE/ATTRIBUTE', 'CATEGORY', 'SAFETY/CONTAINER'}, 'bad': {'OTHER'}, 'w_good': 1.0, 'w_bad': 0.6}, 'inside::microwave': {'good': {'SAFETY/CONTAINER', 'STATE/ATTRIBUTE'}, 'bad': {'OTHER'}, 'w_good': 1.0, 'w_bad': 0.6}}

    def _action_weight(cat: str) -> float:
        key = None
        if (predicate or '').lower() == 'heated':
            key = 'heated'
        if (predicate or '').lower() == 'inside' and (furniture or '').lower() == 'microwave':
            key = 'inside::microwave'
        if key and key in ACTION_CAT_WEIGHT:
            pol = ACTION_CAT_WEIGHT[key]
            if cat in pol['good']:
                return pol['w_good']
            if cat in pol['bad']:
                return pol['w_bad']
        return 1.0
    ACTION_BANS = set()
    if (predicate or '').lower() in {'heated', 'inside'}:
        ACTION_BANS |= {'caffeine', 'caffeine_content', 'stimulant', 'energizing'}
    cand_tags: Dict[str, List[str]] = {c: _get_near_tags_for_candidate(c, kb_class, topk=NEAR_TAGS_TOPK) for c in candidates}
    print('[FACTOR] candidate tags:')
    for name, tags in cand_tags.items():
        print(f'  - {name}: {tags}')
    used_factors_and_syns = set()
    best_try = None
    try_logs: List[Dict] = []
    tried_categories: List[str] = []
    for r in range(1, rounds + 1):
        ban_words = sorted(used_factors_and_syns | ACTION_BANS)
        fac_obj = ask_factor(prev_context or 'UnknownContext', user_sentence, kb_class, candidates, prev_failure=r > 1, extra_ban_words=ban_words, recent_commands=recent_commands, recent_line_id=recent_line_id, cand_state_map=None, try_categories_to_avoid=tried_categories)
        fac = _normalize(fac_obj.get('factor') or '')
        syns = [_normalize(s) for s in fac_obj.get('synonyms') or [] if isinstance(s, str) and s.strip()]
        reason = fac_obj.get('reason') or ''
        cat = _guess_factor_category(fac)
        if cat != 'UNKNOWN':
            tried_categories.append(cat)
        if not fac or fac in used_factors_and_syns:
            used_factors_and_syns.add(fac)
            for s in syns:
                used_factors_and_syns.add(s)
            try_logs.append({'factor': fac, 'reason': 'invalid_or_repeated', 'top': '', 's1': 0, 's2': 0, 'margin_rel': 0.0, 'margin_abs': 0.0, 'category': cat})
            continue
        used_factors_and_syns.add(fac)
        for s in syns:
            used_factors_and_syns.add(s)
        queries = [fac] + syns
        cand_scores: List[Tuple[str, float]] = []
        for c in candidates:
            tags = cand_tags.get(c, [])
            sim = _max_sim_to_tags(queries, tags) if tags else 0.0
            prior = (priors or {}).get(c, 0.0)
            sim_fused = sim * (1.0 + PRIOR_LAMBDA * prior) * _action_weight(cat)
            cand_scores.append((c, max(0.0, min(sim_fused, 1.0))))
        cand_scores.sort(key=lambda x: x[1], reverse=True)
        if not cand_scores:
            try_logs.append({'factor': fac, 'reason': 'no_scores', 'top': '', 's1': 0, 's2': 0, 'margin_rel': 0.0, 'margin_abs': 0.0, 'category': cat})
            continue
        s1 = cand_scores[0][1]
        s2 = cand_scores[1][1] if len(cand_scores) > 1 else 0.0
        margin_rel = (s1 - s2) / (s2 if s2 > 1e-09 else 1.0) if len(cand_scores) > 1 else float('inf')
        margin_abs = s1 - s2
        print(f"[FACTOR] r={r} factor='{fac}' margin_rel={margin_rel:.3f} margin_abs={margin_abs:.3f} top={cand_scores[0][0]} (cat={cat})")
        if reason:
            print(f'[FACTOR] reason: {reason}')
        attempt = {'ok': margin_rel >= gap_threshold or margin_abs >= abs_threshold or margin_rel == float('inf'), 'pick': cand_scores[0][0], 'margin': float(margin_rel), 'scores': cand_scores, 'factor': fac, 'synonyms': syns[:], 'reason': reason, 'hint': '', 'category': cat, 'try_logs': [*try_logs, {'factor': fac, 'reason': reason, 'top': cand_scores[0][0], 's1': s1, 's2': s2, 'margin_rel': float(margin_rel), 'margin_abs': float(margin_abs), 'category': cat}]}
        best_try = attempt
        if attempt['ok']:
            if feedback_logfile:
                try:
                    with open(feedback_logfile, 'a', encoding='utf-8') as f:
                        f.write(json.dumps({'type': 'pick_by_factor_success', 'logs': attempt['try_logs']}, ensure_ascii=False) + '\n')
                except Exception:
                    pass
            return attempt
    if allow_llm_direct_fallback:
        print('[FACTOR] FALLBACK DIRECT  ask LLM to directly choose a candidate')
        pick, reason = choose_instance_no_ctx(user_sentence=user_sentence, kb_class=kb_class, candidates=candidates, recent_commands=recent_commands)
        direct_scores = [(c, 1.0 if c == pick else 0.0) for c in candidates]
        out = {'ok': True, 'pick': pick, 'margin': float('inf'), 'scores': direct_scores, 'factor': '(direct-llm)', 'synonyms': [], 'reason': f'[fallback-direct] {reason}', 'hint': 'factor-failed; direct LLM pick', 'try_logs': best_try['try_logs'] if best_try and 'try_logs' in best_try else []}
        if feedback_logfile:
            try:
                with open(feedback_logfile, 'a', encoding='utf-8') as f:
                    f.write(json.dumps({'type': 'pick_by_factor_direct', 'logs': out['try_logs']}, ensure_ascii=False) + '\n')
            except Exception:
                pass
        return out
    if best_try is None:
        out = {'ok': False, 'pick': candidates[0], 'margin': 0.0, 'scores': [], 'factor': '', 'synonyms': [], 'reason': 'no valid factor', 'hint': 'factor-failed; near-tags were empty', 'try_logs': try_logs[:]}
    else:
        best_try['ok'] = False
        best_try['hint'] = 'factor-failed; tried multiple categories'
        out = best_try
    if feedback_logfile:
        try:
            with open(feedback_logfile, 'a', encoding='utf-8') as f:
                f.write(json.dumps({'type': 'pick_by_factor_fail', 'logs': out['try_logs']}, ensure_ascii=False) + '\n')
        except Exception:
            pass
    return out

def choose_instance(user_sentence: str, kb_class: str, candidates: List[str], *, prev_command: Optional[str]=None, prev_command_nl: Optional[str]=None, prev_context: Optional[str]=None, ctx_scores_top3: Optional[List[Tuple[str, float]]]=None, cand_states: Optional[Dict[str, str]]=None, asp_template: Optional[str]=None, recent_commands: Optional[List[str]]=None) -> Tuple[str, str, str, float]:
    if not candidates:
        return ('unknown', 'no-candidates', 'UnknownContext', 0.0)
    hist_block = ''
    if recent_commands:
        show_hist = '\n'.join((f'- {h}' for h in recent_commands[-12:]))
        hist_block = f'\n## Recent correct commands in this group (most recent last)\n{show_hist}\n'
    prompt = f'You are an embodied household service robot that must resolve ambiguous references without bothering the user.\n\n## Recent interaction\n- Previous spoken command: "{prev_command_nl or 'N/A'}"\n- Previous executed ASP command: {prev_command or 'N/A'}\n- Previous dominant context: {prev_context or 'Unknown'}\n{hist_block}\n## New user utterance (fuzzy)\n"{user_sentence}"\n\n## Candidate objects\nAbstract class: **{kb_class}**\nConcrete instances:\n' + ''.join((f'- {c}\n' for c in candidates)) + f"""\n\n## Task\nChoose the **single** instance that most likely matches the user's intent and fill the template below.\n\nASP template:\n  {asp_template or 'N/A'}\n\nReply **JSON only**, no commentary:\n{{"pick":"<instance>","reason":"<short explanation>"}}"""
    data = _ask_llm_json(prompt) or {}
    if data is None:
        return ('unknown', 'LLM non-JSON', 'UnknownContext', 0.0)
    pick = str(data.get('pick', '')).strip().strip('\'"') or 'unknown'
    reason = str(data.get('reason', '')).strip() or 'no reason'
    pick_ctx = dominant_context(pick)
    sim_prev = scorer.ctx_similarity(pick_ctx, prev_context or '')
    return (pick, reason, pick_ctx, sim_prev)

def choose_action(user_sentence: str, all_actions: List[str], *, recent_commands: Optional[List[str]]=None) -> Tuple[str, str]:
    hist_block = ''
    if recent_commands:
        show_hist = '\n'.join((f'- {h}' for h in recent_commands[-10:]))
        hist_block = f'\n## Recent correct commands in this group\n{show_hist}\n'
    prompt = f'You are an embodied household robot.\n{hist_block}User utterance: "{user_sentence}"\n\nAvailable low-level actions: ' + ', '.join(all_actions) + '.\n\nReply JSON  {"pick":"<action>", "reason":"<short>"}'
    data = _ask_llm_json(prompt) or {}
    return (data.get('pick', 'unknown').strip(), data.get('reason', 'LLM did not explain').strip())

def choose_instance_no_ctx(user_sentence: str, kb_class: str, candidates: List[str], *, asp_template: Optional[str]=None, recent_commands: Optional[List[str]]=None, persona_text: Optional[str]=None) -> Tuple[str, str]:
    pick, _forced, reason = choose_instance_no_ctx_forced(user_sentence, kb_class, candidates, asp_template=asp_template, recent_commands=recent_commands, persona_text=persona_text)
    return (pick, reason)

def choose_instance_no_ctx_forced(user_sentence: str, kb_class: str, candidates: List[str], *, asp_template: Optional[str]=None, recent_commands: Optional[List[str]]=None, persona_text: Optional[str]=None) -> Tuple[str, str, str]:
    if not candidates:
        return ('unknown', 'unknown', 'no-candidates')
    hist_block = ''
    if recent_commands:
        show_hist = '\n'.join((f'- {h}' for h in recent_commands[-12:]))
        hist_block = f'\n## Recent correct commands in this group (most recent last)\n{show_hist}\n'
    persona_block = ''
    if persona_text:
        persona_block = f'\n## User behavioral profile\n{persona_text}\n'
    prompt = f'You are an embodied household service robot. Resolve the ambiguous reference below **without** using any positional or location assumptions.\n\n{hist_block}{persona_block}## User utterance (fuzzy)\n"{user_sentence}"\n\n## Candidate objects\nAbstract class: **{kb_class}**\nConcrete instances:\n' + ''.join((f'- {c}\n' for c in candidates)) + f'\n\n## Task\n1. You MUST select exactly one candidate as your best guess.\n2. Report confidence: high | medium | low.\n3. Use ask_human="yes" ONLY when confidence is low AND you truly cannot distinguish candidates.\n4. In most cases, prefer confidence=medium/high with a best guess instead of asking.\n3. Fill the ASP template below with that instance.\n\nASP template:\n  {asp_template or 'N/A'}\n\nReply **JSON only**, no commentary:\n{{"choice":"<instance>","confidence":"high|medium|low","ask_human":"yes|no","reason":"<short explanation>"}}'
    data = _ask_llm_json(prompt) or {}
    if data is None:
        return ('unknown', candidates[0], 'LLM non-JSON')
    raw_choice = str(data.get('choice', '')).strip().strip('\'"')
    if not raw_choice:
        raw_choice = str(data.get('pick', '')).strip().strip('\'"')
    if not raw_choice:
        raw_choice = str(data.get('forced_pick', '')).strip().strip('\'"')
    choice = raw_choice or 'unknown'
    raw_conf = str(data.get('confidence', '')).strip().lower()
    if not raw_conf:
        raw_conf = str(data.get('confidence_level', '')).strip().lower()
    reason = str(data.get('reason', '')).strip() or 'no reason'
    legacy_pick = str(data.get('pick', '')).strip().strip('\'"').lower()
    if legacy_pick in {'ask_human', 'ask', 'unknown', 'none', 'n/a'} and (not raw_conf):
        raw_conf = 'low'
    if choice not in candidates:
        choice = candidates[0]
    if raw_conf not in {'high', 'medium', 'low'}:
        raw_conf = LLM_CONF_DEFAULT
    ask_flag = str(data.get('ask_human', '')).strip().lower()
    ask_flag = ask_flag in {'yes', 'true', '1', 'ask', 'ask_human'}
    reason_l = reason.lower()
    uncertain_hit = any((kw in reason_l for kw in ('uncertain', 'unsure', 'not sure', 'ambiguous', 'unclear', 'insufficient', 'cannot distinguish', "can't distinguish", 'not enough information')))
    ask_gate_note = ''
    if raw_conf == 'low':
        if len(candidates) < LLM_ASK_MIN_CAND:
            raw_conf = 'medium'
            ask_gate_note = f'ask-gated:min_cands<{LLM_ASK_MIN_CAND}'
        elif LLM_REQUIRE_EXPLICIT_ASK and (not ask_flag):
            raw_conf = 'medium'
            ask_gate_note = 'ask-gated:missing_explicit_ask'
        elif LLM_REQUIRE_UNCERTAIN_REASON and (not uncertain_hit):
            raw_conf = 'medium'
            ask_gate_note = 'ask-gated:reason_not_uncertain'
    pick = 'ASK_HUMAN' if raw_conf == 'low' else choice
    if ask_gate_note:
        reason = f'{reason} ({ask_gate_note})'
    reason = f'{reason} (confidence={raw_conf})'
    forced_pick = choice
    return (pick, forced_pick, reason)

def llm_plan_actions_no_ctx(nl: str, filled_asp: str, *, all_items: List[str], all_furns: List[str], recent_commands: Optional[List[str]]=None) -> List[str]:
    MAX_SHOW = 2000
    items_txt = '\n'.join((f'- {x}' for x in all_items[:MAX_SHOW]))
    furns_txt = '\n'.join((f'- {x}' for x in all_furns[:MAX_SHOW]))
    hist_block = ''
    if recent_commands:
        show_hist = '\n'.join((f'- {h}' for h in recent_commands[-8:]))
        hist_block = f'\n## Recent correct commands in this group\n{show_hist}\n'
    prompt = f'You are controlling a household robot. You have NO state/context. You only know the universe of possible concrete instances.\n\n{hist_block}User utterance: "{nl}"\nDisambiguated ASP command: {filled_asp}\n\nUniverse of instances:\nItems:\n' + items_txt + '\n\nFurniture:\n' + furns_txt + '\n\nTask: Produce a minimal reasonable sequence of ASP-like actions to achieve the command.\nStrictly return JSON only with a single field `action_sequence` which is an array of strings.\nExample:\n{"action_sequence":["occurs(walk(agent1, living_room), 0)","occurs(open(microwave), 1)"]}'
    data = _ask_llm_json(prompt) or {}
    seq: List[str] = []
    if isinstance(data, dict):
        cand = data.get('action_sequence')
        if isinstance(cand, list):
            seq = [s for s in cand if isinstance(s, str)]
    return [s.strip() for s in seq if s and isinstance(s, str)]

def ask_min_label(kb_class: str, verb: str, rest_candidates: List[str]) -> dict:
    prompt = f'You help disambiguate an object class in a home task.\nReturn ONE minimal attribute that best separates the intended item from others.\nDo NOT name an instance. Reply JSON ONLY.\n\nClass: {kb_class}\nVerb: {verb}\nCandidates: {', '.join(rest_candidates[:20])}\n\nJSON shape: {{"attr":"<one_word>","value":"<one_word>"}}'
    data = _ask_llm_json(prompt) or {}
    attr = str(data.get('attr', '')).strip()
    val = str(data.get('value', '')).strip()
    return {'attr': attr, 'value': val}
