#!/usr/bin/env python3
import subprocess
import sys
import textwrap
import shutil
import json
import re
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
PY_EXE = sys.executable
ASP_FILE = ROOT / 'two_goals.sp'
FUZZY_DIR = ROOT / 'experiments' / 'fuzzy_sets'
RESULTS_ROOT = ROOT / 'experiments' / 'results'
PRIVATE_CONFIG_DIR = ROOT / 'runtime' / 'private'
DEFAULT_WEIGHT_PROFILE_PATH = ROOT / 'experiments' / 'configs' / 'weight_profile.json'
DEFAULT_RUN_PROFILE_PATH = ROOT / 'experiments' / 'configs' / 'run_profile.json'

def _resolve_profile_path(private_name: str, fallback: Path) -> Path:
    private_path = PRIVATE_CONFIG_DIR / private_name
    if private_path.exists():
        return private_path
    return fallback

WEIGHT_PROFILE_PATH = _resolve_profile_path('weight_profile.json', DEFAULT_WEIGHT_PROFILE_PATH)
RUN_PROFILE_PATH = _resolve_profile_path('run_profile.json', DEFAULT_RUN_PROFILE_PATH)
SUMMARY_DIR_NAME = '_summary_current'
_run_tag = os.environ.get('RESULTS_RUN_TAG')
if not _run_tag:
    _run_tag = datetime.now().strftime('%Y%m%d_%H%M%S')
_candidate_dir = RESULTS_ROOT / _run_tag
if _candidate_dir.exists():
    suffix = 1
    while (RESULTS_ROOT / f'{_run_tag}_{suffix}').exists():
        suffix += 1
    _run_tag = f'{_run_tag}_{suffix}'
RESULTS_DIR = RESULTS_ROOT / _run_tag
def _load_run_profile(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f'missing run profile: {path}')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError(f'run profile must be a JSON object: {path}')
        return data
    except Exception:
        raise

def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {'1', 'true', 'yes', 'on'}:
            return True
        if s in {'0', 'false', 'no', 'off'}:
            return False
    raise ValueError(f'invalid boolean value: {v!r}')

def _must(profile: dict, key: str):
    if key not in profile:
        raise KeyError(f'missing required key in run profile: {key}')
    return profile[key]

_RUN_PROFILE = _load_run_profile(RUN_PROFILE_PATH)
RUN_MODE = str(_must(_RUN_PROFILE, 'RUN_MODE'))
FORCE_NO_ASK_OPTIONS = list(_must(_RUN_PROFILE, 'FORCE_NO_ASK_OPTIONS'))
CONCEPT_STHEM_MODE_OPTIONS = list(_must(_RUN_PROFILE, 'CONCEPT_STHEM_MODE_OPTIONS'))
AUTO_STHEM_BY_LEVEL = _to_bool(_must(_RUN_PROFILE, 'AUTO_STHEM_BY_LEVEL'))
SEM_SOURCE_MODE = str(_must(_RUN_PROFILE, 'SEM_SOURCE_MODE'))
SEM_HYBRID_ALPHA_CTX = float(_must(_RUN_PROFILE, 'SEM_HYBRID_ALPHA_CTX'))
SEM_CALIBRATE = _to_bool(_must(_RUN_PROFILE, 'SEM_CALIBRATE'))
SEM_CALIBRATE_TAU = float(_must(_RUN_PROFILE, 'SEM_CALIBRATE_TAU'))
SEM_RELIABILITY_GATING = _to_bool(_must(_RUN_PROFILE, 'SEM_RELIABILITY_GATING'))
FACTOR_RELIABILITY_ADAPTIVE = _to_bool(_must(_RUN_PROFILE, 'FACTOR_RELIABILITY_ADAPTIVE'))
FACTOR_REL_BETA = float(_must(_RUN_PROFILE, 'FACTOR_REL_BETA'))
FACTOR_REL_MIN_GAP = float(_must(_RUN_PROFILE, 'FACTOR_REL_MIN_GAP'))
FACTOR_REL_MIN_RANGE = float(_must(_RUN_PROFILE, 'FACTOR_REL_MIN_RANGE'))
FACTOR_REL_GAP_SCALE = float(_must(_RUN_PROFILE, 'FACTOR_REL_GAP_SCALE'))
FACTOR_REL_RANGE_SCALE = float(_must(_RUN_PROFILE, 'FACTOR_REL_RANGE_SCALE'))
FACTOR_REL_GAP_WEIGHT = float(_must(_RUN_PROFILE, 'FACTOR_REL_GAP_WEIGHT'))
FACTOR_REL_RANGE_WEIGHT = float(_must(_RUN_PROFILE, 'FACTOR_REL_RANGE_WEIGHT'))
FACTOR_REL_MIN_SHARE = float(_must(_RUN_PROFILE, 'FACTOR_REL_MIN_SHARE'))
DIFFICULTY_DYNAMIC_WEIGHTS = _to_bool(_must(_RUN_PROFILE, 'DIFFICULTY_DYNAMIC_WEIGHTS'))
DIFF_DYN_MIN_GAP_L1 = float(_must(_RUN_PROFILE, 'DIFF_DYN_MIN_GAP_L1'))
DIFF_DYN_MIN_GAP_L2 = float(_must(_RUN_PROFILE, 'DIFF_DYN_MIN_GAP_L2'))
DIFF_DYN_MIN_GAP_L34 = float(_must(_RUN_PROFILE, 'DIFF_DYN_MIN_GAP_L34'))
DIFF_DYN_MARGIN = float(_must(_RUN_PROFILE, 'DIFF_DYN_MARGIN'))
DIFF_DYN_BOOST_L1 = float(_must(_RUN_PROFILE, 'DIFF_DYN_BOOST_L1'))
DIFF_DYN_BOOST_L2 = float(_must(_RUN_PROFILE, 'DIFF_DYN_BOOST_L2'))
DIFF_DYN_BOOST_L34 = float(_must(_RUN_PROFILE, 'DIFF_DYN_BOOST_L34'))
DIFF_DYN_L1_CONSENSUS_ENABLE = _to_bool(_must(_RUN_PROFILE, 'DIFF_DYN_L1_CONSENSUS_ENABLE'))
DIFF_DYN_L1_CONSENSUS_MIN_SEM_GAP = float(_must(_RUN_PROFILE, 'DIFF_DYN_L1_CONSENSUS_MIN_SEM_GAP'))
DIFF_DYN_L1_CONSENSUS_MIN_THEM_GAP = float(_must(_RUN_PROFILE, 'DIFF_DYN_L1_CONSENSUS_MIN_THEM_GAP'))
DIFF_DYN_L1_CONSENSUS_BOOST = float(_must(_RUN_PROFILE, 'DIFF_DYN_L1_CONSENSUS_BOOST'))
L1_SEM_SAME_BUCKET_NEUTRALIZE = _to_bool(_must(_RUN_PROFILE, 'L1_SEM_SAME_BUCKET_NEUTRALIZE'))
PRONOUN_SEM_NEUTRALIZE = _to_bool(_must(_RUN_PROFILE, 'PRONOUN_SEM_NEUTRALIZE'))
FACTOR_DISTRIBUTION_NORMALIZE = _to_bool(_must(_RUN_PROFILE, 'FACTOR_DISTRIBUTION_NORMALIZE'))
FACTOR_DISTRIBUTION_ZERO_POLICY = str(_must(_RUN_PROFILE, 'FACTOR_DISTRIBUTION_ZERO_POLICY'))
THEMATIC_MODE = str(_must(_RUN_PROFILE, 'THEMATIC_MODE'))
LOCK_CONCEPT_ONLY = _to_bool(_must(_RUN_PROFILE, 'LOCK_CONCEPT_ONLY'))
CONCEPT_MATCH_MODE = str(_must(_RUN_PROFILE, 'CONCEPT_MATCH_MODE'))
CONCEPT_ACTION_WEIGHT = float(_must(_RUN_PROFILE, 'CONCEPT_ACTION_WEIGHT'))
CONCEPT_ENV_WEIGHT = float(_must(_RUN_PROFILE, 'CONCEPT_ENV_WEIGHT'))
CONCEPT_TOPK = int(_must(_RUN_PROFILE, 'CONCEPT_TOPK'))
CONCEPT_NMIN = int(_must(_RUN_PROFILE, 'CONCEPT_NMIN'))
CONCEPT_ROOM_BACKOFF = float(_must(_RUN_PROFILE, 'CONCEPT_ROOM_BACKOFF'))
CONCEPT_PRED_BACKOFF = float(_must(_RUN_PROFILE, 'CONCEPT_PRED_BACKOFF'))
CONCEPT_ENV_WEAK_MARGIN = float(_must(_RUN_PROFILE, 'CONCEPT_ENV_WEAK_MARGIN'))
CONCEPT_ENV_WEAK_SCALE = float(_must(_RUN_PROFILE, 'CONCEPT_ENV_WEAK_SCALE'))
CONCEPT_ENV_MAX_SRC = int(_must(_RUN_PROFILE, 'CONCEPT_ENV_MAX_SRC'))
SALIENCE_UNIFORM_MIX_L1 = float(_must(_RUN_PROFILE, 'SALIENCE_UNIFORM_MIX_L1'))
SALIENCE_UNIFORM_MIX_L2 = float(_must(_RUN_PROFILE, 'SALIENCE_UNIFORM_MIX_L2'))
SALIENCE_UNIFORM_MIX_L3 = float(_must(_RUN_PROFILE, 'SALIENCE_UNIFORM_MIX_L3'))
SALIENCE_UNIFORM_MIX_L4 = float(_must(_RUN_PROFILE, 'SALIENCE_UNIFORM_MIX_L4'))
SALIENCE_UNIFORM_MIX_MIN_CANDS = int(_must(_RUN_PROFILE, 'SALIENCE_UNIFORM_MIX_MIN_CANDS'))
THEMATIC_BLEND_OBJECT = _to_bool(_must(_RUN_PROFILE, 'THEMATIC_BLEND_OBJECT'))
THEMATIC_BLEND_ALPHA = float(_must(_RUN_PROFILE, 'THEMATIC_BLEND_ALPHA'))
THEMATIC_BLEND_ALPHA_L3 = float(_must(_RUN_PROFILE, 'THEMATIC_BLEND_ALPHA_L3'))
THEMATIC_BLEND_ALPHA_L4 = float(_must(_RUN_PROFILE, 'THEMATIC_BLEND_ALPHA_L4'))
THEMATIC_CONCEPT_SAFE_FALLBACK = _to_bool(_must(_RUN_PROFILE, 'THEMATIC_CONCEPT_SAFE_FALLBACK'))
THEMATIC_CONCEPT_SAFE_MIN_TOP = float(_must(_RUN_PROFILE, 'THEMATIC_CONCEPT_SAFE_MIN_TOP'))
THEMATIC_CONCEPT_SAFE_MIN_GAP = float(_must(_RUN_PROFILE, 'THEMATIC_CONCEPT_SAFE_MIN_GAP'))
THEMATIC_CONCEPT_SAFE_ALPHA_OBJ = float(_must(_RUN_PROFILE, 'THEMATIC_CONCEPT_SAFE_ALPHA_OBJ'))
THEMATIC_L1_OBJECT_PRIOR = _to_bool(_must(_RUN_PROFILE, 'THEMATIC_L1_OBJECT_PRIOR'))
THEMATIC_L1_ALPHA_OBJ = float(_must(_RUN_PROFILE, 'THEMATIC_L1_ALPHA_OBJ'))
UNWEIGHTED_THREE_FACTOR = _to_bool(_must(_RUN_PROFILE, 'UNWEIGHTED_THREE_FACTOR'))

def _load_weight_profile() -> dict:
    if not WEIGHT_PROFILE_PATH.exists():
        raise FileNotFoundError(f'missing weight profile: {WEIGHT_PROFILE_PATH}')
    try:
        raw = json.loads(WEIGHT_PROFILE_PATH.read_text(encoding='utf-8'))
    except Exception:
        raise
    if not isinstance(raw, dict):
        raise ValueError(f'weight profile must be a JSON object: {WEIGHT_PROFILE_PATH}')
    out = {}
    for lv in ('L1', 'L2', 'L3', 'L4'):
        if lv not in raw or not isinstance(raw[lv], dict):
            raise KeyError(f'missing required level in weight profile: {lv}')
        row = raw[lv]
        sem = float(row['sem'])
        them = float(row['them'])
        sal = float(row['sal'])
        total = sem + them + sal
        if total <= 0:
            raise ValueError(f'invalid non-positive weight sum at level {lv}')
        out[lv] = {'sem': sem / total, 'them': them / total, 'sal': sal / total}
    return out

_WEIGHT_PROFILE = _load_weight_profile()
W_SEM_L1 = _WEIGHT_PROFILE['L1']['sem']
W_THEM_L1 = _WEIGHT_PROFILE['L1']['them']
W_SAL_L1 = _WEIGHT_PROFILE['L1']['sal']
W_SEM_L2 = _WEIGHT_PROFILE['L2']['sem']
W_THEM_L2 = _WEIGHT_PROFILE['L2']['them']
W_SAL_L2 = _WEIGHT_PROFILE['L2']['sal']
W_SEM_L3 = _WEIGHT_PROFILE['L3']['sem']
W_THEM_L3 = _WEIGHT_PROFILE['L3']['them']
W_SAL_L3 = _WEIGHT_PROFILE['L3']['sal']
W_SEM_L4 = _WEIGHT_PROFILE['L4']['sem']
W_THEM_L4 = _WEIGHT_PROFILE['L4']['them']
W_SAL_L4 = _WEIGHT_PROFILE['L4']['sal']
DISABLE_HISTORY_COUNTER_FOR_PROPOSED = _to_bool(_must(_RUN_PROFILE, 'DISABLE_HISTORY_COUNTER_FOR_PROPOSED'))
STAGE1_SEM_GUARD = _to_bool(_must(_RUN_PROFILE, 'STAGE1_SEM_GUARD'))
STAGE1_SEM_GUARD_LEVELS = str(_must(_RUN_PROFILE, 'STAGE1_SEM_GUARD_LEVELS'))
STAGE1_SEM_GUARD_MIN_THEM = float(_must(_RUN_PROFILE, 'STAGE1_SEM_GUARD_MIN_THEM'))
STAGE1_SEM_GUARD_RATIO = float(_must(_RUN_PROFILE, 'STAGE1_SEM_GUARD_RATIO'))
STAGE1_SEM_GUARD_MIN_CANDS = int(_must(_RUN_PROFILE, 'STAGE1_SEM_GUARD_MIN_CANDS'))
CLEAR_LEAD_RATIO_BASE = float(_must(_RUN_PROFILE, 'CLEAR_LEAD_RATIO_BASE'))
CLEAR_LEAD_RATIO_L1 = float(_must(_RUN_PROFILE, 'CLEAR_LEAD_RATIO_L1'))
CLEAR_LEAD_RATIO_L2 = float(_must(_RUN_PROFILE, 'CLEAR_LEAD_RATIO_L2'))
CLEAR_LEAD_RATIO_L3 = float(_must(_RUN_PROFILE, 'CLEAR_LEAD_RATIO_L3'))
CLEAR_LEAD_RATIO_L4 = float(_must(_RUN_PROFILE, 'CLEAR_LEAD_RATIO_L4'))
FORCE_NO_ASK_FALLBACK_POLICY_DEFAULT = str(_must(_RUN_PROFILE, 'FORCE_NO_ASK_FALLBACK_POLICY_DEFAULT'))
AUTO_FALLBACK_POLICY_BY_LEVEL = _to_bool(_must(_RUN_PROFILE, 'AUTO_FALLBACK_POLICY_BY_LEVEL'))
FORCE_NO_ASK_FALLBACK_POLICY_L1_L2 = str(_must(_RUN_PROFILE, 'FORCE_NO_ASK_FALLBACK_POLICY_L1_L2'))
FORCE_NO_ASK_FALLBACK_POLICY_L3_L4 = str(_must(_RUN_PROFILE, 'FORCE_NO_ASK_FALLBACK_POLICY_L3_L4'))
L3_CLASS_SHRINK = _to_bool(_must(_RUN_PROFILE, 'L3_CLASS_SHRINK'))
L3_DROP_NO_BUCKET = _to_bool(_must(_RUN_PROFILE, 'L3_DROP_NO_BUCKET'))
L3_KEEP_TOP2 = _to_bool(_must(_RUN_PROFILE, 'L3_KEEP_TOP2'))
L3_BUCKET_RATIO = float(_must(_RUN_PROFILE, 'L3_BUCKET_RATIO'))
EVAL_USE_GOLD_GROUP_HISTORY = _to_bool(_must(_RUN_PROFILE, 'EVAL_USE_GOLD_GROUP_HISTORY'))
SKIP_COMPLETED_RUNS = _to_bool(_must(_RUN_PROFILE, 'SKIP_COMPLETED_RUNS'))
CONTINUE_ON_ERROR = _to_bool(_must(_RUN_PROFILE, 'CONTINUE_ON_ERROR'))
WRITE_DETAILED_SUMMARY = _to_bool(_must(_RUN_PROFILE, 'WRITE_DETAILED_SUMMARY'))
MERGE_SUMMARY_WITH_EXISTING = _to_bool(_must(_RUN_PROFILE, 'MERGE_SUMMARY_WITH_EXISTING'))
KEEP_RUN_ARTIFACTS = _to_bool(_must(_RUN_PROFILE, 'KEEP_RUN_ARTIFACTS'))
PUBLISH_SUMMARY_TO_ROOT = _to_bool(_must(_RUN_PROFILE, 'PUBLISH_SUMMARY_TO_ROOT'))
COUNTER_ONLY_CANDIDATES = str(_must(_RUN_PROFILE, 'COUNTER_ONLY_CANDIDATES'))
COUNTER_MATCH_SCOPE = str(_must(_RUN_PROFILE, 'COUNTER_MATCH_SCOPE'))
RUN_MODES = list(_must(_RUN_PROFILE, 'RUN_MODES'))
MODE_ALIAS = dict(_must(_RUN_PROFILE, 'MODE_ALIAS'))
_counter_mode = str(COUNTER_ONLY_CANDIDATES).strip().lower()
if _counter_mode not in {'raw', 'filtered'}:
    print(f" COUNTER_ONLY_CANDIDATES={COUNTER_ONLY_CANDIDATES!r}  'raw'")
    _counter_mode = 'raw'
MODE_ALIAS['counter_only'] = 'most_frequent_filtered' if _counter_mode == 'filtered' else 'most_frequent_raw'

def _normalize_sthem_options(raw_opts: list[str] | None) -> list[str]:
    allowed = {'both', 'instruction', 'environment'}
    out: list[str] = []
    for x in raw_opts or []:
        m = str(x).strip().lower()
        if not m:
            continue
        if m not in allowed:
            print(f'  CONCEPT_STHEM_MODE: {x!r}: {sorted(allowed)}')
            continue
        if m not in out:
            out.append(m)
    return out or ['both']
STHEM_OPTIONS = _normalize_sthem_options(CONCEPT_STHEM_MODE_OPTIONS)

def _enforce_concept_only_config() -> None:
    global THEMATIC_MODE
    global THEMATIC_BLEND_OBJECT, THEMATIC_BLEND_ALPHA, THEMATIC_BLEND_ALPHA_L3, THEMATIC_BLEND_ALPHA_L4
    global THEMATIC_CONCEPT_SAFE_FALLBACK
    global THEMATIC_L1_OBJECT_PRIOR, THEMATIC_L1_ALPHA_OBJ
    global RUN_MODES
    if not LOCK_CONCEPT_ONLY:
        return
    THEMATIC_MODE = 'concept'
    THEMATIC_BLEND_OBJECT = False
    THEMATIC_BLEND_ALPHA = 0.0
    THEMATIC_BLEND_ALPHA_L3 = 0.0
    THEMATIC_BLEND_ALPHA_L4 = 0.0
    THEMATIC_CONCEPT_SAFE_FALLBACK = False
    THEMATIC_L1_OBJECT_PRIOR = False
    THEMATIC_L1_ALPHA_OBJ = 0.0
    RUN_MODES = [m for m in RUN_MODES if m != 'counter_only']
_enforce_concept_only_config()

def _apply_concept_lock_to_env(env: dict) -> None:
    if not LOCK_CONCEPT_ONLY:
        return
    env['THEMATIC_MODE'] = 'concept'
    env['THEMATIC_BLEND_OBJECT'] = '0'
    env['THEMATIC_BLEND_ALPHA'] = '0.0'
    env['THEMATIC_BLEND_ALPHA_L3'] = '0.0'
    env['THEMATIC_BLEND_ALPHA_L4'] = '0.0'
    env['THEMATIC_CONCEPT_SAFE_FALLBACK'] = '0'
    env['THEMATIC_L1_OBJECT_PRIOR'] = '0'
    env['THEMATIC_L1_ALPHA_OBJ'] = '0.0'

def _sthem_suffix(sthem_mode: str) -> str:
    m = str(sthem_mode or 'both').strip().lower() or 'both'
    if len(STHEM_OPTIONS) == 1 and m == 'both':
        return ''
    return f'__sthem_{m}'

def _sthem_mode_for_target(target_name: str, requested_mode: str) -> str:
    mode = str(requested_mode or 'both').strip().lower() or 'both'
    if not AUTO_STHEM_BY_LEVEL:
        return mode
    name = str(target_name or '').strip().lower()
    if '_l1' in name or '_l2' in name:
        return 'instruction'
    if '_l3' in name or '_l4' in name:
        return 'both'
    return mode

def _fallback_policy_for_target(target_name: str) -> str:
    default_policy = str(FORCE_NO_ASK_FALLBACK_POLICY_DEFAULT).strip().lower() or 'fused_top1'
    if not AUTO_FALLBACK_POLICY_BY_LEVEL:
        return default_policy
    name = str(target_name or '').strip().lower()
    if '_l1' in name or '_l2' in name:
        return str(FORCE_NO_ASK_FALLBACK_POLICY_L1_L2).strip().lower() or default_policy
    if '_l3' in name or '_l4' in name:
        return str(FORCE_NO_ASK_FALLBACK_POLICY_L3_L4).strip().lower() or default_policy
    return default_policy

def _sem_suffix() -> str:
    m = str(SEM_SOURCE_MODE or 'wordnet').strip().lower() or 'wordnet'
    if m == 'wordnet':
        return ''
    return f'__sem_{m}'
targets = ['user1_health_l1.txt', 'user1_health_l2.txt', 'user1_health_l3.txt', 'user1_health_l4.txt', 'user2_caffeine_l1.txt', 'user2_caffeine_l2.txt', 'user2_caffeine_l3.txt', 'user2_caffeine_l4.txt', 'user3_comfort_l1.txt', 'user3_comfort_l2.txt', 'user3_comfort_l3.txt', 'user3_comfort_l4.txt', 'user4_scholar_l1.txt', 'user4_scholar_l2.txt', 'user4_scholar_l3.txt', 'user4_scholar_l4.txt', 'user5_social_l1.txt', 'user5_social_l2.txt', 'user5_social_l3.txt', 'user5_social_l4.txt']
SINGLE_FILE_NAME = 'user1_health_l1.txt'
SINGLE_RESULT_DIR_NAME = None
SINGLE_FILE_LIST = ['user1_health_l1.txt', 'user1_health_l2.txt', 'user1_health_l3.txt', 'user1_health_l4.txt']
DISAMBIG_JSONL = ROOT / 'disambig_results.jsonl'
DISAMBIG_TXT = ROOT / 'disambig_results.txt'
RUN_SUMMARY_RECORDS: list[dict] = []

def _asp_file_looks_valid(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding='utf-8')
    except Exception:
        return False
    required_tokens = ('sorts', 'predicates', 'rules', '#action', '% ===== INITIAL CONDITIONS START =====', '% ===== INITIAL CONDITIONS END =====')
    if any((tok not in text for tok in required_tokens)):
        return False
    return text.count('\n') >= 250

def ensure_asp_file_integrity() -> bool:
    if _asp_file_looks_valid(ASP_FILE):
        return True
    print(f'  {ASP_FILE}  git HEAD ...')
    if ASP_FILE.exists():
        bad_backup = ASP_FILE.with_name(f'two_goals.sp.bad_autosave_{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak')
        try:
            shutil.copy2(ASP_FILE, bad_backup)
            print(f'[ASP] bad backup saved: {bad_backup}')
        except Exception as e:
            print(f'[ASP] backup bad file failed: {e}')
    try:
        res = subprocess.run(['git', 'show', 'HEAD:two_goals.sp'], cwd=ROOT, check=True, capture_output=True, text=True)
        ASP_FILE.write_text(res.stdout, encoding='utf-8')
    except Exception as e:
        print(f' {e}')
        return False
    ok = _asp_file_looks_valid(ASP_FILE)
    if ok:
        print(f'  git HEAD  {ASP_FILE}')
    else:
        print(f'  {ASP_FILE}')
    return ok

def truncate_global_logs() -> None:
    for p in (DISAMBIG_JSONL, DISAMBIG_TXT):
        try:
            if p.exists():
                p.write_text('', encoding='utf-8')
        except Exception:
            pass

def clear_run_outputs(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    for name in ('execute_log.txt', 'test_results.json', 'disambig_results.jsonl', 'disambig_results.txt', 'group_history.jsonl', 'group_history.txt'):
        p = run_dir / name
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    for p in run_dir.glob('tri_missing_report*'):
        try:
            if p.is_file():
                p.unlink()
        except Exception:
            pass

def is_run_completed(run_dir: Path) -> bool:
    return (run_dir / 'test_metrics' / 'test_summary.json').exists()
    for dname in ('metrics', 'test_metrics'):
        d = run_dir / dname
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)

def _user_id_from_target(target_name: str) -> str | None:
    m = re.match('^(.+?)_l\\d+\\.txt$', target_name)
    if not m:
        m = re.match('^(.+?)_round\\d+_', target_name)
    return m.group(1) if m else None

def show_execute_log(round_name: str, tail_lines: int=80) -> None:
    log_path = RESULTS_DIR / round_name / 'execute_log.txt'
    if not log_path.exists():
        print(f'[LOG] execute_log.txt {log_path}')
        return
    try:
        lines = log_path.read_text(encoding='utf-8').splitlines()
        tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
        print(f'\n[LOG] tail of {log_path} (last {len(tail)} lines):')
        print('\n'.join(tail))
    except Exception as e:
        print(f'[LOG]  {log_path} {e!r}')

def find_matching_fuzzy_files(pattern: str):
    if not FUZZY_DIR.exists():
        print(f' Fuzzy : {FUZZY_DIR}')
        return []
    matched = []
    for p in FUZZY_DIR.glob('*.txt'):
        if p.name.endswith(pattern):
            matched.append(p)
    return sorted(matched)

def check_long_asp(jsonl_path: Path, max_steps: int=15) -> None:
    if not jsonl_path.exists():
        print(f'  JSONL : {jsonl_path}')
        return
    long_lines = []
    try:
        with jsonl_path.open('r', encoding='utf-8') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except Exception:
                    continue
                steps = row.get('disambig_steps')
                try:
                    steps_i = int(steps) if steps is not None else 0
                except Exception:
                    steps_i = 0
                if steps_i > max_steps:
                    long_lines.append((row.get('line_id'), steps_i))
    except Exception as e:
        print(f'  {jsonl_path} : {e}')
        return
    if long_lines:
        print(f'  {len(long_lines)}  ASP  > {max_steps}')
        for lid, st in long_lines[:20]:
            print(f'   - line_id={lid}, disambig_steps={st}')
        if len(long_lines) > 20:
            print(f'    {len(long_lines) - 20} ')

def _load_test_results_json(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f'missing test results: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, list):
        raise ValueError('test_results.json must be a JSON list')
    return data


def _summarize_test_results_rows(results: list[dict]) -> dict:
    total = len(results)
    success_total = 0
    answered_total = 0
    answered_success = 0
    ask_total = 0
    hard_error_total = 0
    compare_total = 0
    compare_hist_answered = 0
    compare_hist_success = 0
    compare_llm_answered = 0
    compare_llm_success = 0
    alt_llm_total = 0
    alt_llm_success_total = 0
    derived_noask_success = 0
    derived_noask_hard_error = 0
    derived_noask_total = 0
    group_stats = defaultdict(lambda: {'total': 0, 'answered': 0, 'success_answered': 0, 'ask': 0, 'hard_error': 0})

    for row in results:
        success = bool(row.get('success', False))
        decision = row.get('decision')
        decision_u = decision.upper() if isinstance(decision, str) else ''
        group_id = row.get('group')
        if group_id is None:
            group_id = 0
        gs = group_stats[group_id]
        gs['total'] += 1
        if success:
            success_total += 1

        is_llm = decision_u.startswith('LLM')
        answered_labels = {'DIRECT', 'HISTORY', 'FALLBACK', 'RANDOM', 'HEURISTIC'}
        is_answered = decision_u in answered_labels or is_llm
        is_ask = decision_u == 'ASK_HUMAN'

        if is_answered:
            answered_total += 1
            gs['answered'] += 1
            if success:
                answered_success += 1
                gs['success_answered'] += 1
            else:
                hard_error_total += 1
                gs['hard_error'] += 1
        elif is_ask:
            ask_total += 1
            gs['ask'] += 1
        else:
            answered_total += 1
            hard_error_total += 1
            gs['answered'] += 1
            gs['hard_error'] += 1

        attempt_number = row.get('attempt_number')
        use_for_noask = True
        if attempt_number is not None:
            try:
                use_for_noask = int(attempt_number) == 1
            except Exception:
                use_for_noask = True
        if use_for_noask:
            derived_noask_total += 1
            if decision_u == 'ASK_HUMAN':
                derived_pred = row.get('fallback_prediction')
            else:
                derived_pred = row.get('predicted')
            if derived_pred == row.get('correct'):
                derived_noask_success += 1
            else:
                derived_noask_hard_error += 1

        if row.get('compare_stage1_failed'):
            compare_total += 1
            if row.get('alt_history_cmd') is not None:
                compare_hist_answered += 1
                if row.get('alt_history_success') is True:
                    compare_hist_success += 1
            if row.get('alt_llm_cmd') is not None:
                compare_llm_answered += 1
                if row.get('alt_llm_success') is True:
                    compare_llm_success += 1

        if 'alt_llm_success_full' in row:
            alt_llm_total += 1
            if row.get('alt_llm_success_full') is True:
                alt_llm_success_total += 1

    overall_acc = success_total / total if total > 0 else 0.0
    answer_acc = answered_success / answered_total if answered_total > 0 else 0.0
    abstain_rate = ask_total / total if total > 0 else 0.0
    hard_error_rate = hard_error_total / total if total > 0 else 0.0
    derived_noask_overall = derived_noask_success / derived_noask_total if derived_noask_total > 0 else 0.0
    derived_noask_hard_error_rate = derived_noask_hard_error / derived_noask_total if derived_noask_total > 0 else 0.0

    return {
        'total': total,
        'success_total': success_total,
        'overall_acc': overall_acc,
        'answered_total': answered_total,
        'answered_success': answered_success,
        'answer_acc': answer_acc,
        'ask_total': ask_total,
        'abstain_rate': abstain_rate,
        'hard_error_total': hard_error_total,
        'hard_error_rate': hard_error_rate,
        'derived_noask': {
            'total': derived_noask_total,
            'success': derived_noask_success,
            'overall_acc': derived_noask_overall,
            'hard_error_total': derived_noask_hard_error,
            'hard_error_rate': derived_noask_hard_error_rate,
        },
        'group_stats': group_stats,
        'compare': {
            'total_stage1_failed': compare_total,
            'history_answered': compare_hist_answered,
            'history_success': compare_hist_success,
            'history_acc': compare_hist_success / compare_hist_answered if compare_hist_answered > 0 else 0.0,
            'llm_answered': compare_llm_answered,
            'llm_success': compare_llm_success,
            'llm_acc': compare_llm_success / compare_llm_answered if compare_llm_answered > 0 else 0.0,
        },
        'alt_llm_pipeline': {
            'total': alt_llm_total,
            'success': alt_llm_success_total,
            'overall_acc': alt_llm_success_total / alt_llm_total if alt_llm_total > 0 else 0.0,
        },
    }


def _format_test_summary_markdown(summary: dict, input_path: Path, results: list[dict]) -> str:
    ts = datetime.utcnow().isoformat() + 'Z'
    lines = []
    lines.append('# Test Results Summary')
    lines.append('')
    lines.append(f'- Generated at: {ts}')
    lines.append(f'- Input file: `{input_path}`')
    lines.append(f"- Total samples: **{summary['total']}**")
    lines.append(f"- Success: **{summary['success_total']}**")
    lines.append(f"- Overall accuracy: **{summary['overall_acc']:.3f}**")
    lines.append('')
    lines.append('## Outcomes')
    lines.append('')
    lines.append(f"- Answered samples: **{summary['answered_total']}**")
    lines.append(f"- Answered success: **{summary['answered_success']}**")
    lines.append(f"- Answered-only accuracy: **{summary['answer_acc']:.3f}**")
    lines.append(f"- ASK_HUMAN count: **{summary['ask_total']}**")
    lines.append(f"- Abstention rate: **{summary['abstain_rate']:.3f}**")
    lines.append(f"- Hard errors: **{summary['hard_error_total']}**")
    lines.append(f"- Hard error rate: **{summary['hard_error_rate']:.3f}**")
    lines.append('')

    derived = summary.get('derived_noask', {})
    lines.append('## Derived noask')
    lines.append('')
    lines.append(f"- Total samples: **{derived.get('total', 0)}**")
    lines.append(f"- Success: **{derived.get('success', 0)}**")
    lines.append(f"- Overall accuracy: **{derived.get('overall_acc', 0.0):.3f}**")
    lines.append(f"- Hard error rate: **{derived.get('hard_error_rate', 0.0):.3f}**")
    lines.append('')

    compare = summary.get('compare', {})
    lines.append('## Stage1 Failed Compare')
    lines.append('')
    lines.append(f"- Samples: **{compare.get('total_stage1_failed', 0)}**")
    lines.append(f"- History answered: **{compare.get('history_answered', 0)}**, success: **{compare.get('history_success', 0)}**, acc: **{compare.get('history_acc', 0.0):.3f}**")
    lines.append(f"- LLM answered: **{compare.get('llm_answered', 0)}**, success: **{compare.get('llm_success', 0)}**, acc: **{compare.get('llm_acc', 0.0):.3f}**")
    lines.append('')

    alt_llm = summary.get('alt_llm_pipeline', {})
    lines.append('## Alt Pipeline')
    lines.append('')
    lines.append(f"- Total: **{alt_llm.get('total', 0)}**, success: **{alt_llm.get('success', 0)}**, overall acc: **{alt_llm.get('overall_acc', 0.0):.3f}**")
    lines.append('')

    lines.append('## Decision Types')
    lines.append('')
    decision_stats = {d: {'total': 0, 'success': 0} for d in ('DIRECT', 'HISTORY', 'LLM_ONLY', 'LLM_HISTORY', 'LLM', 'ASK_HUMAN')}
    decision_stats['_OTHER_'] = {'total': 0, 'success': 0}
    for row in results:
        key = row.get('decision')
        success = bool(row.get('success', False))
        key = key if key in decision_stats else '_OTHER_'
        decision_stats[key]['total'] += 1
        if success:
            decision_stats[key]['success'] += 1
    for key, info in decision_stats.items():
        if info['total'] == 0:
            continue
        label = 'OTHER' if key == '_OTHER_' else key
        acc = info['success'] / info['total']
        lines.append(f"- {label}: total={info['total']}, success={info['success']}, acc={acc:.3f}")
    lines.append('')

    lines.append('## By Group')
    lines.append('')
    for group_id in sorted(summary['group_stats'].keys()):
        gs = summary['group_stats'][group_id]
        total_g = gs['total']
        answered_g = gs['answered']
        success_g = gs['success_answered']
        ask_g = gs['ask']
        hard_g = gs['hard_error']
        acc_ans = success_g / answered_g if answered_g > 0 else 0.0
        abstain = ask_g / total_g if total_g > 0 else 0.0
        hard_rate = hard_g / total_g if total_g > 0 else 0.0
        lines.append(f'- Group {group_id}: total={total_g}, answered={answered_g}, success={success_g}, acc_answered={acc_ans:.3f}, ask={ask_g}, hard_error={hard_g}, abstain_rate={abstain:.3f}, hard_error_rate={hard_rate:.3f}')
    lines.append('')
    return '\n'.join(lines)


def summarize_test_results_for_dir(res_dir: Path, tag: str='') -> None:
    test_json = res_dir / 'test_results.json'
    if not test_json.exists():
        print(f' [{tag}] missing test results: {test_json}')
        return
    print(f' [{tag}] Summarizing test results in {res_dir} ...')
    try:
        results = _load_test_results_json(test_json)
        summary = _summarize_test_results_rows(results)
        out_dir = res_dir / 'test_metrics'
        out_dir.mkdir(parents=True, exist_ok=True)
        out_md = out_dir / 'test_summary.md'
        out_json = out_dir / 'test_summary.json'
        md_text = _format_test_summary_markdown(summary, test_json, results)
        out_md.write_text(md_text, encoding='utf-8')
        json_obj = {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'input_file': str(test_json),
            'summary': {k: v for k, v in summary.items() if k != 'group_stats'},
            'group_stats': {str(g): stats for g, stats in summary['group_stats'].items()},
        }
        out_json.write_text(json.dumps(json_obj, indent=2), encoding='utf-8')
    except Exception as e:
        print(f' [{tag}] failed to summarize test results: {e}')

def _parse_user_level(target_name: str) -> tuple[str | None, str | None]:
    user_id = _user_id_from_target(target_name)
    level = None
    m = re.search('_l(\\d+)\\.txt$', target_name)
    if m:
        level = f'l{m.group(1)}'
    return (user_id, level)

def _load_test_summary(res_dir: Path) -> dict | None:
    summary_path = res_dir / 'test_metrics' / 'test_summary.json'
    if not summary_path.exists():
        return None
    try:
        return json.loads(summary_path.read_text(encoding='utf-8'))
    except Exception:
        return None

def _row_from_summary(summary_json: dict, run_name: str, target_name: str, mode: str | None, sthem_mode: str | None, force_no_ask: bool, res_dir: Path) -> dict:
    summary = summary_json.get('summary', {}) or {}
    compare = summary.get('compare', {}) or {}
    alt_llm = summary.get('alt_llm_pipeline', {}) or {}
    derived_noask = summary.get('derived_noask', {}) or {}
    user_id, level = _parse_user_level(target_name)
    ask_mode = 'noask' if force_no_ask else 'ask'
    row = {'run_name': run_name, 'target_file': target_name, 'user_id': user_id or '', 'level': level or '', 'mode': mode or 'default', 'sthem_mode': sthem_mode or 'both', 'ask_mode': ask_mode, 'result_dir': str(res_dir), 'input_file': summary_json.get('input_file', ''), 'total': summary.get('total', 0), 'success_total': summary.get('success_total', 0), 'overall_acc': summary.get('overall_acc', 0.0), 'answered_total': summary.get('answered_total', 0), 'answered_success': summary.get('answered_success', 0), 'answer_acc': summary.get('answer_acc', 0.0), 'ask_total': summary.get('ask_total', 0), 'abstain_rate': summary.get('abstain_rate', 0.0), 'hard_error_total': summary.get('hard_error_total', 0), 'hard_error_rate': summary.get('hard_error_rate', 0.0), 'derived_noask_total': derived_noask.get('total', 0), 'derived_noask_success': derived_noask.get('success', 0), 'derived_noask_overall_acc': derived_noask.get('overall_acc', 0.0), 'derived_noask_hard_error_total': derived_noask.get('hard_error_total', 0), 'derived_noask_hard_error_rate': derived_noask.get('hard_error_rate', 0.0), 'compare_total_stage1_failed': compare.get('total_stage1_failed', 0), 'compare_history_answered': compare.get('history_answered', 0), 'compare_history_success': compare.get('history_success', 0), 'compare_history_acc': compare.get('history_acc', 0.0), 'compare_llm_answered': compare.get('llm_answered', 0), 'compare_llm_success': compare.get('llm_success', 0), 'compare_llm_acc': compare.get('llm_acc', 0.0), 'alt_llm_total': alt_llm.get('total', 0), 'alt_llm_success': alt_llm.get('success', 0), 'alt_llm_overall_acc': alt_llm.get('overall_acc', 0.0)}
    row['_group_stats'] = summary_json.get('group_stats', {})
    return row

def _aggregate_rows(rows: list[dict], key_fields: list[str]) -> list[dict]:
    agg: dict[tuple, dict] = {}
    sum_fields = ['total', 'success_total', 'answered_total', 'answered_success', 'ask_total', 'hard_error_total', 'derived_noask_total', 'derived_noask_success', 'derived_noask_hard_error_total', 'compare_total_stage1_failed', 'compare_history_answered', 'compare_history_success', 'compare_llm_answered', 'compare_llm_success', 'alt_llm_total', 'alt_llm_success']
    for row in rows:
        key = tuple((row.get(k, '') for k in key_fields))
        rec = agg.setdefault(key, {k: row.get(k, '') for k in key_fields})
        for f in sum_fields:
            rec[f] = rec.get(f, 0) + (row.get(f, 0) or 0)
    for rec in agg.values():
        total = rec.get('total', 0) or 0
        answered_total = rec.get('answered_total', 0) or 0
        rec['overall_acc'] = rec.get('success_total', 0) / total if total else 0.0
        rec['answer_acc'] = rec.get('answered_success', 0) / answered_total if answered_total else 0.0
        rec['abstain_rate'] = rec.get('ask_total', 0) / total if total else 0.0
        rec['hard_error_rate'] = rec.get('hard_error_total', 0) / total if total else 0.0
        d_total = rec.get('derived_noask_total', 0) or 0
        d_success = rec.get('derived_noask_success', 0) or 0
        d_hard = rec.get('derived_noask_hard_error_total', 0) or 0
        rec['derived_noask_overall_acc'] = d_success / d_total if d_total else 0.0
        rec['derived_noask_hard_error_rate'] = d_hard / d_total if d_total else 0.0
        hist_ans = rec.get('compare_history_answered', 0) or 0
        llm_ans = rec.get('compare_llm_answered', 0) or 0
        rec['compare_history_acc'] = rec.get('compare_history_success', 0) / hist_ans if hist_ans else 0.0
        rec['compare_llm_acc'] = rec.get('compare_llm_success', 0) / llm_ans if llm_ans else 0.0
        alt_total = rec.get('alt_llm_total', 0) or 0
        rec['alt_llm_overall_acc'] = rec.get('alt_llm_success', 0) / alt_total if alt_total else 0.0
    return list(agg.values())

def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fields})

def _write_md_table(path: Path, rows: list[dict], fields: list[str]) -> None:
    lines = []
    lines.append('| ' + ' | '.join(fields) + ' |')
    lines.append('| ' + ' | '.join(['---'] * len(fields)) + ' |')
    for row in rows:
        vals = []
        for k in fields:
            v = row.get(k, '')
            if isinstance(v, float):
                vals.append(f'{v:.3f}')
            else:
                vals.append(str(v))
        lines.append('| ' + ' | '.join(vals) + ' |')
    path.write_text('\n'.join(lines), encoding='utf-8')

def _coerce_number(val: str) -> object:
    if val is None:
        return val
    sval = str(val).strip()
    if sval == '':
        return ''
    try:
        if re.fullmatch('-?\\d+', sval):
            return int(sval)
        if re.fullmatch('-?\\d*\\.\\d+', sval):
            return float(sval)
    except Exception:
        return val
    return val

def _load_existing_summary_table(path: Path, fields: list[str]) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    try:
        with path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = {}
                for k in fields:
                    parsed[k] = _coerce_number(row.get(k, ''))
                rows.append(parsed)
    except Exception as e:
        print(f'[SUMMARY] Failed to read existing summary_table.csv: {e}')
    return rows

def collect_summary(run_name: str, target_name: str, mode: str | None, sthem_mode: str | None, force_no_ask: bool, res_dir: Path) -> None:
    summary_json = _load_test_summary(res_dir)
    if not summary_json:
        return
    row = _row_from_summary(summary_json, run_name, target_name, mode, sthem_mode, force_no_ask, res_dir)
    RUN_SUMMARY_RECORDS.append(row)

def write_detailed_summary(rows: list[dict]) -> None:
    if not rows:
        return
    out_dir = RESULTS_DIR / SUMMARY_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ['run_name', 'target_file', 'user_id', 'level', 'mode', 'sthem_mode', 'ask_mode', 'total', 'success_total', 'overall_acc', 'answered_total', 'answered_success', 'answer_acc', 'ask_total', 'abstain_rate', 'hard_error_total', 'hard_error_rate', 'derived_noask_total', 'derived_noask_success', 'derived_noask_overall_acc', 'derived_noask_hard_error_total', 'derived_noask_hard_error_rate', 'compare_total_stage1_failed', 'compare_history_answered', 'compare_history_success', 'compare_history_acc', 'compare_llm_answered', 'compare_llm_success', 'compare_llm_acc', 'alt_llm_total', 'alt_llm_success', 'alt_llm_overall_acc', 'result_dir', 'input_file']
    merged_rows = rows
    if MERGE_SUMMARY_WITH_EXISTING:
        existing_rows = _load_existing_summary_table(out_dir / 'summary_table.csv', fields)
        if existing_rows:
            merged = {r.get('run_name', ''): r for r in existing_rows}
            for r in rows:
                merged[r.get('run_name', '')] = r
            merged_rows = list(merged.values())
    _write_csv(out_dir / 'summary_table.csv', merged_rows, fields)
    _write_md_table(out_dir / 'summary_table.md', merged_rows, fields)
    agg_rows = _aggregate_rows(merged_rows, ['user_id', 'level', 'mode', 'sthem_mode', 'ask_mode'])
    agg_fields = ['user_id', 'level', 'mode', 'sthem_mode', 'ask_mode', 'total', 'success_total', 'overall_acc', 'answered_total', 'answered_success', 'answer_acc', 'ask_total', 'abstain_rate', 'hard_error_total', 'hard_error_rate', 'derived_noask_total', 'derived_noask_success', 'derived_noask_overall_acc', 'derived_noask_hard_error_total', 'derived_noask_hard_error_rate', 'compare_total_stage1_failed', 'compare_history_answered', 'compare_history_success', 'compare_history_acc', 'compare_llm_answered', 'compare_llm_success', 'compare_llm_acc', 'alt_llm_total', 'alt_llm_success', 'alt_llm_overall_acc']
    _write_csv(out_dir / 'summary_agg.csv', agg_rows, agg_fields)
    _write_md_table(out_dir / 'summary_agg.md', agg_rows, agg_fields)
    group_stats: dict = {}
    if MERGE_SUMMARY_WITH_EXISTING:
        existing_stats_path = out_dir / 'summary_group_stats.json'
        if existing_stats_path.exists():
            try:
                group_stats.update(json.loads(existing_stats_path.read_text(encoding='utf-8')))
            except Exception:
                pass
    for row in merged_rows:
        if row.get('_group_stats'):
            group_stats[row['run_name']] = row.get('_group_stats', {})
    (out_dir / 'summary_group_stats.json').write_text(json.dumps(group_stats, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[SUMMARY] Detailed summary written to {out_dir}')

def cleanup_results(keep_dir: Path) -> None:
    for p in RESULTS_DIR.iterdir():
        if p == keep_dir:
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
        except Exception:
            pass

def publish_summary_to_root(summary_dir: Path) -> None:
    if not summary_dir.exists():
        return
    root_summary = RESULTS_ROOT / SUMMARY_DIR_NAME
    try:
        if root_summary.exists():
            shutil.rmtree(root_summary)
        shutil.copytree(summary_dir, root_summary)
        print(f'[SUMMARY] Published to {root_summary}')
    except Exception as e:
        print(f' summary {e!r}')

def run_single_target(target_name: str, result_dir_name: str | None=None, disambig_mode: str | None=None, sthem_mode: str | None=None, force_no_ask: bool | None=None) -> bool:
    if not FUZZY_DIR.exists():
        print(f' Fuzzy : {FUZZY_DIR}')
        return False
    target_path = FUZZY_DIR / target_name
    if not target_path.exists():
        print(f' {target_path}')
        return False
    print(f'\n----- Running {target_path.name} -----')
    run_name = result_dir_name if result_dir_name else target_name.replace('.txt', '')
    res_dir = RESULTS_DIR / run_name
    res_dir.mkdir(parents=True, exist_ok=True)
    if SKIP_COMPLETED_RUNS and is_run_completed(res_dir):
        print(f'[SKIP] {run_name}')
        return True
    clear_run_outputs(res_dir)
    truncate_global_logs()
    env = os.environ.copy()
    effective_sthem_mode = _sthem_mode_for_target(target_path.name, sthem_mode or 'both')
    fallback_policy = _fallback_policy_for_target(target_path.name)
    resolved_mode = ''
    if disambig_mode:
        resolved_mode = MODE_ALIAS.get(disambig_mode, disambig_mode)
        env['DISAMBIG_MODE'] = resolved_mode
    else:
        env.pop('DISAMBIG_MODE', None)
    if effective_sthem_mode:
        env['CONCEPT_STHEM_MODE'] = str(effective_sthem_mode).strip().lower()
    env['CONCEPT_MATCH_MODE'] = str(CONCEPT_MATCH_MODE).strip().lower()
    env['CONCEPT_ACTION_WEIGHT'] = str(CONCEPT_ACTION_WEIGHT)
    env['CONCEPT_ENV_WEIGHT'] = str(CONCEPT_ENV_WEIGHT)
    env['CONCEPT_TOPK'] = str(CONCEPT_TOPK)
    env['CONCEPT_NMIN'] = str(CONCEPT_NMIN)
    env['CONCEPT_ROOM_BACKOFF'] = str(CONCEPT_ROOM_BACKOFF)
    env['CONCEPT_PRED_BACKOFF'] = str(CONCEPT_PRED_BACKOFF)
    env['CONCEPT_ENV_WEAK_MARGIN'] = str(CONCEPT_ENV_WEAK_MARGIN)
    env['CONCEPT_ENV_WEAK_SCALE'] = str(CONCEPT_ENV_WEAK_SCALE)
    env['CONCEPT_ENV_MAX_SRC'] = str(CONCEPT_ENV_MAX_SRC)
    env['SALIENCE_UNIFORM_MIX_L1'] = str(SALIENCE_UNIFORM_MIX_L1)
    env['SALIENCE_UNIFORM_MIX_L2'] = str(SALIENCE_UNIFORM_MIX_L2)
    env['SALIENCE_UNIFORM_MIX_L3'] = str(SALIENCE_UNIFORM_MIX_L3)
    env['SALIENCE_UNIFORM_MIX_L4'] = str(SALIENCE_UNIFORM_MIX_L4)
    env['SALIENCE_UNIFORM_MIX_MIN_CANDS'] = str(SALIENCE_UNIFORM_MIX_MIN_CANDS)
    env['SEM_SOURCE'] = str(SEM_SOURCE_MODE).strip().lower()
    env['SEM_HYBRID_ALPHA_CTX'] = str(SEM_HYBRID_ALPHA_CTX)
    env['SEM_CALIBRATE_ENABLED'] = '1' if SEM_CALIBRATE else '0'
    env['SEM_CALIBRATE_TAU'] = str(SEM_CALIBRATE_TAU)
    env['SEM_RELIABILITY_GATING'] = '1' if SEM_RELIABILITY_GATING else '0'
    env['FACTOR_RELIABILITY_ADAPTIVE'] = '1' if FACTOR_RELIABILITY_ADAPTIVE else '0'
    env['FACTOR_REL_BETA'] = str(FACTOR_REL_BETA)
    env['FACTOR_REL_MIN_GAP'] = str(FACTOR_REL_MIN_GAP)
    env['FACTOR_REL_MIN_RANGE'] = str(FACTOR_REL_MIN_RANGE)
    env['FACTOR_REL_GAP_SCALE'] = str(FACTOR_REL_GAP_SCALE)
    env['FACTOR_REL_RANGE_SCALE'] = str(FACTOR_REL_RANGE_SCALE)
    env['FACTOR_REL_GAP_WEIGHT'] = str(FACTOR_REL_GAP_WEIGHT)
    env['FACTOR_REL_RANGE_WEIGHT'] = str(FACTOR_REL_RANGE_WEIGHT)
    env['FACTOR_REL_MIN_SHARE'] = str(FACTOR_REL_MIN_SHARE)
    env['DIFFICULTY_DYNAMIC_WEIGHTS'] = '1' if DIFFICULTY_DYNAMIC_WEIGHTS else '0'
    env['DIFF_DYN_MIN_GAP_L1'] = str(DIFF_DYN_MIN_GAP_L1)
    env['DIFF_DYN_MIN_GAP_L2'] = str(DIFF_DYN_MIN_GAP_L2)
    env['DIFF_DYN_MIN_GAP_L34'] = str(DIFF_DYN_MIN_GAP_L34)
    env['DIFF_DYN_MARGIN'] = str(DIFF_DYN_MARGIN)
    env['DIFF_DYN_BOOST_L1'] = str(DIFF_DYN_BOOST_L1)
    env['DIFF_DYN_BOOST_L2'] = str(DIFF_DYN_BOOST_L2)
    env['DIFF_DYN_BOOST_L34'] = str(DIFF_DYN_BOOST_L34)
    env['DIFF_DYN_L1_CONSENSUS_ENABLE'] = '1' if DIFF_DYN_L1_CONSENSUS_ENABLE else '0'
    env['DIFF_DYN_L1_CONSENSUS_MIN_SEM_GAP'] = str(DIFF_DYN_L1_CONSENSUS_MIN_SEM_GAP)
    env['DIFF_DYN_L1_CONSENSUS_MIN_THEM_GAP'] = str(DIFF_DYN_L1_CONSENSUS_MIN_THEM_GAP)
    env['DIFF_DYN_L1_CONSENSUS_BOOST'] = str(DIFF_DYN_L1_CONSENSUS_BOOST)
    env['L1_SEM_SAME_BUCKET_NEUTRALIZE'] = '1' if L1_SEM_SAME_BUCKET_NEUTRALIZE else '0'
    env['PRONOUN_SEM_NEUTRALIZE'] = '1' if PRONOUN_SEM_NEUTRALIZE else '0'
    env['FACTOR_DISTRIBUTION_NORMALIZE'] = '1' if FACTOR_DISTRIBUTION_NORMALIZE else '0'
    env['FACTOR_DISTRIBUTION_ZERO_POLICY'] = str(FACTOR_DISTRIBUTION_ZERO_POLICY).strip().lower()
    env['THEMATIC_MODE'] = str(THEMATIC_MODE).strip().lower()
    env['THEMATIC_BLEND_OBJECT'] = '1' if THEMATIC_BLEND_OBJECT else '0'
    env['THEMATIC_BLEND_ALPHA'] = str(THEMATIC_BLEND_ALPHA)
    env['THEMATIC_BLEND_ALPHA_L3'] = str(THEMATIC_BLEND_ALPHA_L3)
    env['THEMATIC_BLEND_ALPHA_L4'] = str(THEMATIC_BLEND_ALPHA_L4)
    env['THEMATIC_CONCEPT_SAFE_FALLBACK'] = '1' if THEMATIC_CONCEPT_SAFE_FALLBACK else '0'
    env['THEMATIC_CONCEPT_SAFE_MIN_TOP'] = str(THEMATIC_CONCEPT_SAFE_MIN_TOP)
    env['THEMATIC_CONCEPT_SAFE_MIN_GAP'] = str(THEMATIC_CONCEPT_SAFE_MIN_GAP)
    env['THEMATIC_CONCEPT_SAFE_ALPHA_OBJ'] = str(THEMATIC_CONCEPT_SAFE_ALPHA_OBJ)
    env['THEMATIC_L1_OBJECT_PRIOR'] = '1' if THEMATIC_L1_OBJECT_PRIOR else '0'
    env['THEMATIC_L1_ALPHA_OBJ'] = str(THEMATIC_L1_ALPHA_OBJ)
    env['UNWEIGHTED_THREE_FACTOR'] = '1' if UNWEIGHTED_THREE_FACTOR else '0'
    env['W_SEM_L1'] = str(W_SEM_L1)
    env['W_THEM_L1'] = str(W_THEM_L1)
    env['W_SAL_L1'] = str(W_SAL_L1)
    env['W_SEM_L2'] = str(W_SEM_L2)
    env['W_THEM_L2'] = str(W_THEM_L2)
    env['W_SAL_L2'] = str(W_SAL_L2)
    env['W_SEM_L3'] = str(W_SEM_L3)
    env['W_THEM_L3'] = str(W_THEM_L3)
    env['W_SAL_L3'] = str(W_SAL_L3)
    env['W_SEM_L4'] = str(W_SEM_L4)
    env['W_THEM_L4'] = str(W_THEM_L4)
    env['W_SAL_L4'] = str(W_SAL_L4)
    disable_counter = DISABLE_HISTORY_COUNTER_FOR_PROPOSED and resolved_mode in {'three_factor_ask_only', 'three_factor'}
    env['DISABLE_HISTORY_COUNTER'] = '1' if disable_counter else '0'
    env['STAGE1_SEM_GUARD'] = '1' if STAGE1_SEM_GUARD else '0'
    env['STAGE1_SEM_GUARD_LEVELS'] = str(STAGE1_SEM_GUARD_LEVELS)
    env['STAGE1_SEM_GUARD_MIN_THEM'] = str(STAGE1_SEM_GUARD_MIN_THEM)
    env['STAGE1_SEM_GUARD_RATIO'] = str(STAGE1_SEM_GUARD_RATIO)
    env['STAGE1_SEM_GUARD_MIN_CANDS'] = str(STAGE1_SEM_GUARD_MIN_CANDS)
    env['CLEAR_LEAD_RATIO_BASE'] = str(CLEAR_LEAD_RATIO_BASE)
    env['CLEAR_LEAD_RATIO_L1'] = str(CLEAR_LEAD_RATIO_L1)
    env['CLEAR_LEAD_RATIO_L2'] = str(CLEAR_LEAD_RATIO_L2)
    env['CLEAR_LEAD_RATIO_L3'] = str(CLEAR_LEAD_RATIO_L3)
    env['CLEAR_LEAD_RATIO_L4'] = str(CLEAR_LEAD_RATIO_L4)
    env['L3_CLASS_SHRINK'] = '1' if L3_CLASS_SHRINK else '0'
    env['L3_DROP_NO_BUCKET'] = '1' if L3_DROP_NO_BUCKET else '0'
    env['L3_KEEP_TOP2'] = '1' if L3_KEEP_TOP2 else '0'
    env['L3_BUCKET_RATIO'] = str(L3_BUCKET_RATIO)
    env['EVAL_USE_GOLD_GROUP_HISTORY'] = '1' if EVAL_USE_GOLD_GROUP_HISTORY else '0'
    if force_no_ask is None:
        force_no_ask = False
    env['FORCE_NO_ASK'] = '1' if force_no_ask else '0'
    env['FORCE_NO_ASK_FALLBACK_POLICY'] = fallback_policy
    env['COUNTER_MATCH_SCOPE'] = str(COUNTER_MATCH_SCOPE).strip().lower()
    _apply_concept_lock_to_env(env)
    env['FUZZY_RUN_NAME'] = str(Path(RESULTS_DIR.name) / run_name)
    ret = subprocess.run([PY_EXE, 'test_fuzzy_sets.py', '--target', target_path.name], cwd=ROOT, stdin=subprocess.DEVNULL, env=env)
    if ret.returncode != 0:
        print(f' {target_path.name} ')
        return False
    moved_jsonl = res_dir / 'disambig_results.jsonl'
    moved_txt = res_dir / 'disambig_results.txt'
    if DISAMBIG_JSONL.exists():
        shutil.copyfile(DISAMBIG_JSONL, moved_jsonl)
    if DISAMBIG_TXT.exists():
        shutil.copyfile(DISAMBIG_TXT, moved_txt)
    check_long_asp(moved_jsonl, max_steps=15)
    summarize_test_results_for_dir(res_dir, tag=run_name)
    collect_summary(run_name, target_path.name, disambig_mode, effective_sthem_mode, force_no_ask or False, res_dir)
    show_execute_log(run_name, tail_lines=80)
    print('\n ')
    print(f'   {res_dir}/')
    print('     - test_results.json')
    print('     - disambig_results.(jsonl|txt)')
    print('     - test_metrics/test_summary.(md|json)')
    return True

def run_single_file():
    print('\n==================== SINGLE_FILE  ====================')
    base = SINGLE_FILE_NAME.replace('.txt', '')
    if RUN_MODES:
        for mode in RUN_MODES:
            for sthem_mode in STHEM_OPTIONS:
                for force_no_ask in FORCE_NO_ASK_OPTIONS:
                    sthem_eff = _sthem_mode_for_target(SINGLE_FILE_NAME, sthem_mode)
                    ask_suffix = 'noask' if force_no_ask else 'ask'
                    run_name = f'{base}__{mode}__{ask_suffix}{_sthem_suffix(sthem_eff)}{_sem_suffix()}'
                    ok = run_single_target(SINGLE_FILE_NAME, run_name, mode, sthem_eff, force_no_ask)
                    if not ok and (not CONTINUE_ON_ERROR):
                        break
    else:
        for sthem_mode in STHEM_OPTIONS:
            for force_no_ask in FORCE_NO_ASK_OPTIONS:
                sthem_eff = _sthem_mode_for_target(SINGLE_FILE_NAME, sthem_mode)
                ask_suffix = 'noask' if force_no_ask else 'ask'
                run_name = f'{SINGLE_RESULT_DIR_NAME or base}__{ask_suffix}{_sthem_suffix(sthem_eff)}{_sem_suffix()}'
                ok = run_single_target(SINGLE_FILE_NAME, run_name, None, sthem_eff, force_no_ask)
                if not ok and (not CONTINUE_ON_ERROR):
                    break

def run_pattern_mode():
    print('  fuzzy \n' + textwrap.indent('\n'.join(targets), '   '))
    modes = RUN_MODES if RUN_MODES else [None]
    for t in targets:
        fuzzy_files = find_matching_fuzzy_files(t)
        if not fuzzy_files:
            print(f"  '{t}'  fuzzy {FUZZY_DIR} ")
            continue
        print(' test_fuzzy_sets.py')
        print(textwrap.indent('\n'.join((str(p.name) for p in fuzzy_files)), '  - '))
        for mode in modes:
            for sthem_mode in STHEM_OPTIONS:
                for force_no_ask in FORCE_NO_ASK_OPTIONS:
                    mode_suffix = f'__{mode}' if mode else ''
                    ask_suffix = 'noask' if force_no_ask else 'ask'
                    sthem_suffix = _sthem_suffix(sthem_mode)
                    mode_label = f'{mode or 'default'}::{ask_suffix}::sthem={sthem_mode}'
                    print(f'\n==================== Pattern {t} ({mode_label}) ====================')
                    round_name = f'{t.replace('.txt', '')}{mode_suffix}__{ask_suffix}{sthem_suffix}{_sem_suffix()}'
                    res_dir = RESULTS_DIR / round_name
                    res_dir.mkdir(parents=True, exist_ok=True)
                    agg_jsonl = res_dir / 'disambig_results.jsonl'
                    agg_txt = res_dir / 'disambig_results.txt'
                    for pth in (agg_jsonl, agg_txt):
                        try:
                            pth.write_text('', encoding='utf-8')
                        except Exception:
                            pass
                    pattern_ok = True
                    for fpath in fuzzy_files:
                        print(f'\n----- Running {fpath.name} -----')
                        truncate_global_logs()
                        sthem_eff = _sthem_mode_for_target(fpath.name, sthem_mode)
                        fallback_policy = _fallback_policy_for_target(fpath.name)
                        env = os.environ.copy()
                        resolved_mode = ''
                        if mode:
                            resolved_mode = MODE_ALIAS.get(mode, mode)
                            env['DISAMBIG_MODE'] = resolved_mode
                        else:
                            env.pop('DISAMBIG_MODE', None)
                        env['CONCEPT_STHEM_MODE'] = sthem_eff
                        env['CONCEPT_MATCH_MODE'] = str(CONCEPT_MATCH_MODE).strip().lower()
                        env['CONCEPT_ACTION_WEIGHT'] = str(CONCEPT_ACTION_WEIGHT)
                        env['CONCEPT_ENV_WEIGHT'] = str(CONCEPT_ENV_WEIGHT)
                        env['CONCEPT_TOPK'] = str(CONCEPT_TOPK)
                        env['CONCEPT_NMIN'] = str(CONCEPT_NMIN)
                        env['CONCEPT_ENV_WEAK_MARGIN'] = str(CONCEPT_ENV_WEAK_MARGIN)
                        env['CONCEPT_ENV_WEAK_SCALE'] = str(CONCEPT_ENV_WEAK_SCALE)
                        env['CONCEPT_ENV_MAX_SRC'] = str(CONCEPT_ENV_MAX_SRC)
                        env['SEM_SOURCE'] = str(SEM_SOURCE_MODE).strip().lower()
                        env['SEM_HYBRID_ALPHA_CTX'] = str(SEM_HYBRID_ALPHA_CTX)
                        env['SEM_CALIBRATE_ENABLED'] = '1' if SEM_CALIBRATE else '0'
                        env['SEM_CALIBRATE_TAU'] = str(SEM_CALIBRATE_TAU)
                        env['SEM_RELIABILITY_GATING'] = '1' if SEM_RELIABILITY_GATING else '0'
                        env['FACTOR_RELIABILITY_ADAPTIVE'] = '1' if FACTOR_RELIABILITY_ADAPTIVE else '0'
                        env['FACTOR_REL_BETA'] = str(FACTOR_REL_BETA)
                        env['FACTOR_REL_MIN_GAP'] = str(FACTOR_REL_MIN_GAP)
                        env['FACTOR_REL_MIN_RANGE'] = str(FACTOR_REL_MIN_RANGE)
                        env['FACTOR_REL_GAP_SCALE'] = str(FACTOR_REL_GAP_SCALE)
                        env['FACTOR_REL_RANGE_SCALE'] = str(FACTOR_REL_RANGE_SCALE)
                        env['FACTOR_REL_GAP_WEIGHT'] = str(FACTOR_REL_GAP_WEIGHT)
                        env['FACTOR_REL_RANGE_WEIGHT'] = str(FACTOR_REL_RANGE_WEIGHT)
                        env['FACTOR_REL_MIN_SHARE'] = str(FACTOR_REL_MIN_SHARE)
                        env['DIFFICULTY_DYNAMIC_WEIGHTS'] = '1' if DIFFICULTY_DYNAMIC_WEIGHTS else '0'
                        env['DIFF_DYN_MIN_GAP_L1'] = str(DIFF_DYN_MIN_GAP_L1)
                        env['DIFF_DYN_MIN_GAP_L2'] = str(DIFF_DYN_MIN_GAP_L2)
                        env['DIFF_DYN_MIN_GAP_L34'] = str(DIFF_DYN_MIN_GAP_L34)
                        env['DIFF_DYN_MARGIN'] = str(DIFF_DYN_MARGIN)
                        env['DIFF_DYN_BOOST_L1'] = str(DIFF_DYN_BOOST_L1)
                        env['DIFF_DYN_BOOST_L2'] = str(DIFF_DYN_BOOST_L2)
                        env['DIFF_DYN_BOOST_L34'] = str(DIFF_DYN_BOOST_L34)
                        env['DIFF_DYN_L1_CONSENSUS_ENABLE'] = '1' if DIFF_DYN_L1_CONSENSUS_ENABLE else '0'
                        env['DIFF_DYN_L1_CONSENSUS_MIN_SEM_GAP'] = str(DIFF_DYN_L1_CONSENSUS_MIN_SEM_GAP)
                        env['DIFF_DYN_L1_CONSENSUS_MIN_THEM_GAP'] = str(DIFF_DYN_L1_CONSENSUS_MIN_THEM_GAP)
                        env['DIFF_DYN_L1_CONSENSUS_BOOST'] = str(DIFF_DYN_L1_CONSENSUS_BOOST)
                        env['L1_SEM_SAME_BUCKET_NEUTRALIZE'] = '1' if L1_SEM_SAME_BUCKET_NEUTRALIZE else '0'
                        env['PRONOUN_SEM_NEUTRALIZE'] = '1' if PRONOUN_SEM_NEUTRALIZE else '0'
                        env['FACTOR_DISTRIBUTION_NORMALIZE'] = '1' if FACTOR_DISTRIBUTION_NORMALIZE else '0'
                        env['FACTOR_DISTRIBUTION_ZERO_POLICY'] = str(FACTOR_DISTRIBUTION_ZERO_POLICY).strip().lower()
                        env['THEMATIC_MODE'] = str(THEMATIC_MODE).strip().lower()
                        env['THEMATIC_BLEND_OBJECT'] = '1' if THEMATIC_BLEND_OBJECT else '0'
                        env['THEMATIC_BLEND_ALPHA'] = str(THEMATIC_BLEND_ALPHA)
                        env['THEMATIC_BLEND_ALPHA_L3'] = str(THEMATIC_BLEND_ALPHA_L3)
                        env['THEMATIC_BLEND_ALPHA_L4'] = str(THEMATIC_BLEND_ALPHA_L4)
                        env['THEMATIC_CONCEPT_SAFE_FALLBACK'] = '1' if THEMATIC_CONCEPT_SAFE_FALLBACK else '0'
                        env['THEMATIC_CONCEPT_SAFE_MIN_TOP'] = str(THEMATIC_CONCEPT_SAFE_MIN_TOP)
                        env['THEMATIC_CONCEPT_SAFE_MIN_GAP'] = str(THEMATIC_CONCEPT_SAFE_MIN_GAP)
                        env['THEMATIC_CONCEPT_SAFE_ALPHA_OBJ'] = str(THEMATIC_CONCEPT_SAFE_ALPHA_OBJ)
                        env['THEMATIC_L1_OBJECT_PRIOR'] = '1' if THEMATIC_L1_OBJECT_PRIOR else '0'
                        env['THEMATIC_L1_ALPHA_OBJ'] = str(THEMATIC_L1_ALPHA_OBJ)
                        env['UNWEIGHTED_THREE_FACTOR'] = '1' if UNWEIGHTED_THREE_FACTOR else '0'
                        env['W_SEM_L1'] = str(W_SEM_L1)
                        env['W_THEM_L1'] = str(W_THEM_L1)
                        env['W_SAL_L1'] = str(W_SAL_L1)
                        env['W_SEM_L2'] = str(W_SEM_L2)
                        env['W_THEM_L2'] = str(W_THEM_L2)
                        env['W_SAL_L2'] = str(W_SAL_L2)
                        env['W_SEM_L3'] = str(W_SEM_L3)
                        env['W_THEM_L3'] = str(W_THEM_L3)
                        env['W_SAL_L3'] = str(W_SAL_L3)
                        env['W_SEM_L4'] = str(W_SEM_L4)
                        env['W_THEM_L4'] = str(W_THEM_L4)
                        env['W_SAL_L4'] = str(W_SAL_L4)
                        disable_counter = DISABLE_HISTORY_COUNTER_FOR_PROPOSED and resolved_mode in {'three_factor_ask_only', 'three_factor'}
                        env['DISABLE_HISTORY_COUNTER'] = '1' if disable_counter else '0'
                        env['STAGE1_SEM_GUARD'] = '1' if STAGE1_SEM_GUARD else '0'
                        env['STAGE1_SEM_GUARD_LEVELS'] = str(STAGE1_SEM_GUARD_LEVELS)
                        env['STAGE1_SEM_GUARD_MIN_THEM'] = str(STAGE1_SEM_GUARD_MIN_THEM)
                        env['STAGE1_SEM_GUARD_RATIO'] = str(STAGE1_SEM_GUARD_RATIO)
                        env['STAGE1_SEM_GUARD_MIN_CANDS'] = str(STAGE1_SEM_GUARD_MIN_CANDS)
                        env['CLEAR_LEAD_RATIO_BASE'] = str(CLEAR_LEAD_RATIO_BASE)
                        env['CLEAR_LEAD_RATIO_L1'] = str(CLEAR_LEAD_RATIO_L1)
                        env['CLEAR_LEAD_RATIO_L2'] = str(CLEAR_LEAD_RATIO_L2)
                        env['CLEAR_LEAD_RATIO_L3'] = str(CLEAR_LEAD_RATIO_L3)
                        env['CLEAR_LEAD_RATIO_L4'] = str(CLEAR_LEAD_RATIO_L4)
                        env['L3_CLASS_SHRINK'] = '1' if L3_CLASS_SHRINK else '0'
                        env['L3_DROP_NO_BUCKET'] = '1' if L3_DROP_NO_BUCKET else '0'
                        env['L3_KEEP_TOP2'] = '1' if L3_KEEP_TOP2 else '0'
                        env['L3_BUCKET_RATIO'] = str(L3_BUCKET_RATIO)
                        env['EVAL_USE_GOLD_GROUP_HISTORY'] = '1' if EVAL_USE_GOLD_GROUP_HISTORY else '0'
                        env['FORCE_NO_ASK'] = '1' if force_no_ask else '0'
                        env['FORCE_NO_ASK_FALLBACK_POLICY'] = fallback_policy
                        env['COUNTER_MATCH_SCOPE'] = str(COUNTER_MATCH_SCOPE).strip().lower()
                        _apply_concept_lock_to_env(env)
                        file_stem = fpath.name.replace('.txt', '')
                        run_name = f'{file_stem}{mode_suffix}__{ask_suffix}{_sthem_suffix(sthem_eff)}{_sem_suffix()}'
                        env['FUZZY_RUN_NAME'] = str(Path(RESULTS_DIR.name) / run_name)
                        file_dir = RESULTS_DIR / run_name
                        file_dir.mkdir(parents=True, exist_ok=True)
                        if SKIP_COMPLETED_RUNS and is_run_completed(file_dir):
                            print(f'[SKIP] {run_name}')
                            continue
                        clear_run_outputs(file_dir)
                        ret = subprocess.run([PY_EXE, 'test_fuzzy_sets.py', '--target', fpath.name], cwd=ROOT, stdin=subprocess.DEVNULL, env=env)
                        if ret.returncode != 0:
                            print(f" {fpath.name}  pattern '{t}'")
                            pattern_ok = False
                            if CONTINUE_ON_ERROR:
                                continue
                            break
                        file_jsonl = file_dir / 'disambig_results.jsonl'
                        file_txt = file_dir / 'disambig_results.txt'
                        if DISAMBIG_JSONL.exists():
                            shutil.copyfile(DISAMBIG_JSONL, file_jsonl)
                        if DISAMBIG_TXT.exists():
                            shutil.copyfile(DISAMBIG_TXT, file_txt)
                        check_long_asp(file_jsonl, max_steps=15)
                        summarize_test_results_for_dir(file_dir, tag=run_name)
                        collect_summary(run_name, fpath.name, mode, sthem_eff, force_no_ask, file_dir)
                        if DISAMBIG_JSONL.exists():
                            try:
                                with DISAMBIG_JSONL.open('r', encoding='utf-8') as src, agg_jsonl.open('a', encoding='utf-8') as dst:
                                    for ln in src:
                                        dst.write(ln)
                            except Exception as e:
                                print(f'  JSONL : {e}')
                        if DISAMBIG_TXT.exists():
                            try:
                                with DISAMBIG_TXT.open('r', encoding='utf-8') as src, agg_txt.open('a', encoding='utf-8') as dst:
                                    dst.write(f'\n===== FILE {fpath.name} =====\n')
                                    for ln in src:
                                        dst.write(ln)
                            except Exception as e:
                                print(f'  TXT : {e}')
                    if not pattern_ok:
                        continue
                    if not agg_jsonl.exists():
                        print(f" pattern '{t}'  JSONL ")
                        continue
                    check_long_asp(agg_jsonl, max_steps=15)
    print('\n PATTERN ')
    print('   -  pattern   experiments/results/<pattern_without_ext>/ ')
    print('        disambig_results.jsonl / .txt')
    print('   -  fuzzy   experiments/results/<file_stem>/ ')
    print('        execute_log.txt, test_results.json test_fuzzy_sets.py')
    print('        disambig_results.jsonl / .txt')
    print('        test_metrics/test_summary.(md|json) vs ASK_HUMAN ')
if __name__ == '__main__':
    if not ensure_asp_file_integrity():
        raise SystemExit(1)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f'[RUN] results_dir = {RESULTS_DIR}')
    print(f'[RUN] sthem_modes = {STHEM_OPTIONS}')
    print('[RUN] profiles loaded')
    print(f'[RUN] counter = candidates:{COUNTER_ONLY_CANDIDATES}, match_scope:{COUNTER_MATCH_SCOPE}')
    if LOCK_CONCEPT_ONLY:
        print('[RUN] concept lock = ON (pure concept thematic enforced)')
    print('[RUN] lead thresholds loaded')
    if RUN_MODE.upper() == 'SINGLE_FILE':
        run_single_file()
    elif RUN_MODE.upper() == 'SINGLE_FILES':
        if not SINGLE_FILE_LIST:
            print(' SINGLE_FILE_LIST ')
        for name in SINGLE_FILE_LIST:
            if RUN_MODES:
                base = name.replace('.txt', '')
                ok = True
                for mode in RUN_MODES:
                    for sthem_mode in STHEM_OPTIONS:
                        for force_no_ask in FORCE_NO_ASK_OPTIONS:
                            sthem_eff = _sthem_mode_for_target(name, sthem_mode)
                            ask_suffix = 'noask' if force_no_ask else 'ask'
                            run_name = f'{base}__{mode}__{ask_suffix}{_sthem_suffix(sthem_eff)}{_sem_suffix()}'
                            ok = run_single_target(name, run_name, mode, sthem_eff, force_no_ask)
                            if not ok:
                                print(f' {name} ')
                                if not CONTINUE_ON_ERROR:
                                    break
                    if not ok:
                        if not CONTINUE_ON_ERROR:
                            break
                if not ok:
                    if not CONTINUE_ON_ERROR:
                        break
            else:
                ok = True
                for sthem_mode in STHEM_OPTIONS:
                    for force_no_ask in FORCE_NO_ASK_OPTIONS:
                        sthem_eff = _sthem_mode_for_target(name, sthem_mode)
                        ask_suffix = 'noask' if force_no_ask else 'ask'
                        run_name = f'{name.replace('.txt', '')}__{ask_suffix}{_sthem_suffix(sthem_eff)}{_sem_suffix()}'
                        ok = run_single_target(name, run_name, None, sthem_eff, force_no_ask)
                        if not ok:
                            print(f' {name} ')
                            if not CONTINUE_ON_ERROR:
                                break
                if not ok:
                    if not CONTINUE_ON_ERROR:
                        break
    else:
        run_pattern_mode()
    if WRITE_DETAILED_SUMMARY:
        write_detailed_summary(RUN_SUMMARY_RECORDS)
        try:
            from generate_paper_tables import main as generate_tables
            generate_tables()
        except Exception as e:
            print(f' generate_paper_tables {e!r}')
        if PUBLISH_SUMMARY_TO_ROOT:
            publish_summary_to_root(RESULTS_DIR / SUMMARY_DIR_NAME)
        if not KEEP_RUN_ARTIFACTS:
            cleanup_results(RESULTS_DIR / SUMMARY_DIR_NAME)
