#!/usr/bin/env python3
import argparse
import sys
import re
import json
from pathlib import Path
import os
from config.config import SHOW_START_OUTPUT, SHOW_LAST_OUTPUT, SHOW_CHANGED_HOLDS_OUTPUT
from commands.context_io import clear_context_file
from commands.group_log import init_group_logs
from commands.thematic_fit import set_thematic_history_file
from kg.history_manager import set_active_history_file
clear_context_file()
parser = argparse.ArgumentParser()
parser.add_argument('--target', default='user1_health_l1.txt', help=' experiments/fuzzy_sets/  user1_health_l1.txt  user2_caffeine_round1_hypernym.txt')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
FS_DIR = ROOT / 'experiments' / 'fuzzy_sets'
MU_DIR = ROOT / 'experiments' / 'multi_user_dataset'
EVAL_USE_GOLD_GROUP_HISTORY = os.environ.get('EVAL_USE_GOLD_GROUP_HISTORY', '1').strip().lower() in {'1', 'true', 'yes', 'on'}
TARGET_FILE = args.target
TARGET_PATH = FS_DIR / TARGET_FILE
if not TARGET_PATH.exists():
    sys.stderr.write(f'[ERROR]  {TARGET_PATH}\n')
    sys.exit(1)
m_level = re.search('_l(\\d+)\\.txt$', TARGET_FILE)
if m_level:
    TARGET_TYPE = f'l{m_level.group(1)}'
elif 'hypernym' in TARGET_FILE:
    TARGET_TYPE = 'hypernym'
elif 'pronoun' in TARGET_FILE:
    TARGET_TYPE = 'pronoun'
elif 'attribute' in TARGET_FILE:
    TARGET_TYPE = 'attribute'
else:
    TARGET_TYPE = 'unknown'
run_name = os.environ.get('FUZZY_RUN_NAME')
if run_name:
    RES_DIR = ROOT / 'experiments' / 'results' / run_name
else:
    RES_DIR = ROOT / 'experiments' / 'results' / TARGET_FILE.replace('.txt', '')
RES_DIR.mkdir(parents=True, exist_ok=True)
os.environ['DISAMBIG_RUN_NAME'] = run_name or TARGET_FILE.replace('.txt', '')
LOG_FILE = RES_DIR / 'execute_log.txt'
OUT_JSON = RES_DIR / 'test_results.json'
init_group_logs(str(RES_DIR), reset=True)
m_user = re.match('^(.+)_round(\\d+)_', TARGET_FILE)
if not m_user:
    m_user = re.match('^(.+)_l\\d+\\.txt$', TARGET_FILE)
user_id = m_user.group(1) if m_user else None
ext_active = os.environ.get('ACTIVE_HISTORY_FILE', '').strip()
ext_them = os.environ.get('THEMATIC_HISTORY_FILE', '').strip()
if ext_active and Path(ext_active).exists():
    set_active_history_file(ext_active)
    if ext_them and Path(ext_them).exists():
        set_thematic_history_file(ext_them)
        print(f'[CFG] Active/Thematic history overridden by env: {ext_active} / {ext_them}')
    else:
        set_thematic_history_file(ext_active)
        print(f'[CFG] Active/Thematic history overridden by env: {ext_active}')
elif m_user:
    hist_path = MU_DIR / f'{user_id}_history.txt'
    if hist_path.exists():
        set_active_history_file(str(hist_path))
        set_thematic_history_file(str(hist_path))
        print(f'[CFG] Active history & thematic history set to {hist_path}')
    else:
        set_active_history_file(None)
        print(f'[WARN] history file {hist_path} not found; using default HISTORY_FILE.')
else:
    set_active_history_file(None)
    print(f'[INFO] TARGET_FILE {TARGET_FILE} has no user prefix; using default HISTORY_FILE.')

class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        self.flush()

    def flush(self):
        for s in self.streams:
            if hasattr(s, 'flush'):
                s.flush()
log_fp = open(LOG_FILE, 'w', encoding='utf-8', buffering=1)
sys.stdout = Tee(sys.__stdout__, log_fp)
sys.stderr = Tee(sys.__stderr__, log_fp)
from kg.kg_updater import revert_kg_to_backup, update_kg_from_asp_outputs
from asp.file_manager import extract_initial_conditions, remove_initial_conditions_from_asp, insert_initial_conditions_to_asp
from asp.goals import execute_user_goal
from main import pipeline, add_group_history, clear_group_history, replace_group_history
from commands.confidence_scorer import set_repeat_pool
from commands.command_planner import plan
MULTIWORD_MAP = {'water glass': 'water_glass', 'hot milk': 'hot_milk', 'hot espresso': 'hot_espresso', 'hot chicken': 'hot_chicken', 'classic novel': 'classic_novel', 'fantasy novel': 'fantasy_novel', 'sci fi novel': 'sci_fi_novel', 'sci-fi novel': 'sci_fi_novel', 'superhero comic': 'superhero_comic', 'graphic memoir': 'graphic_memoir', 'computer science textbook': 'computer_science_textbook', 'computer-science textbook': 'computer_science_textbook', 'physics textbook': 'physics_textbook', 'kitchen table': 'kitchen_table', 'kitchen counter': 'kitchen_counter', 'coffee table': 'coffee_table', 'dish bowl': 'dish_bowl', 'audio amplifier': 'audio_amplifier', 'tv stand': 'tv_stand', 'TV stand': 'tv_stand', 'table lamp': 'table_lamp', 'study desk': 'study_desk', 'computer desk': 'computer_desk', 'hot drink': 'hot_drink', 'cold drink': 'cold_drink', 'study item': 'study_item', 'snack': 'snack'}

def replace_multiwords(s: str) -> str:
    for k, v in MULTIWORD_MAP.items():
        s = re.sub(f'\\b{k}\\b', v, s, flags=re.IGNORECASE)
    return s
ASP_CMD_RE = re.compile('^\\s*([a-zA-Z_][\\w]*)\\s*\\(([^)]*)\\)\\s*$')

def extract_obj_from_asp(asp_cmd: str | None) -> str | None:
    if not asp_cmd:
        return None
    m = ASP_CMD_RE.match(str(asp_cmd).strip())
    if not m:
        return None
    pred = m.group(1)
    args = [a.strip() for a in m.group(2).split(',') if a.strip()]
    if not args:
        return None
    if pred in {'inside', 'on', 'has'}:
        return args[-1]
    if pred == 'heated':
        return args[0]
    return args[-1]

def analyse_stage_error(asp_gold: str, pred_cmd: str, success: bool, detail: dict) -> dict:
    gold_obj = extract_obj_from_asp(asp_gold)
    pred_obj = extract_obj_from_asp(pred_cmd)
    decision = str(detail.get('decision') or '')
    scores = detail.get('scores') or []
    stage1_top1 = None
    stage1_top2 = None
    m1 = None
    m2 = None
    lead_ratio = None
    if scores:
        sorted_scores = sorted(scores, key=lambda d: float(d.get('mean', 0.0)), reverse=True)
        if sorted_scores:
            stage1_top1 = sorted_scores[0].get('name')
            try:
                m1 = float(sorted_scores[0].get('mean', 0.0))
            except Exception:
                m1 = None
        if len(sorted_scores) > 1:
            stage1_top2 = sorted_scores[1].get('name')
            try:
                m2 = float(sorted_scores[1].get('mean', 0.0))
            except Exception:
                m2 = None
        if m1 is not None and m2 not in (None, 0.0):
            lead_ratio = (m1 - m2) / max(m2, 1e-09)
    stage1_ok = bool(gold_obj and stage1_top1 == gold_obj)
    if success:
        if decision == 'HISTORY':
            error_src = 'OK_STAGE2_REDUNDANT' if stage1_ok else 'OK_FIXED_BY_HISTORY'
        else:
            error_src = 'OK'
    elif decision == 'HISTORY':
        error_src = 'ERR_STAGE2_OVERRIDE_CORRECT_STAGE1' if stage1_ok else 'ERR_STAGE1_AND_STAGE2'
    elif decision == 'DIRECT':
        error_src = 'ERR_STAGE1'
    else:
        error_src = 'ERR_OTHER'
    if not scores or not gold_obj:
        stage1_status = 'NO_STAGE1_OR_GOLD'
    elif stage1_ok and success and (decision != 'HISTORY'):
        stage1_status = 'S1_CORRECT_USED'
    elif stage1_ok and success and (decision == 'HISTORY'):
        stage1_status = 'S1_CORRECT_BUT_ESCALATED'
    elif stage1_ok and (not success) and (decision == 'HISTORY'):
        stage1_status = 'S1_CORRECT_BUT_OVERRIDDEN_WRONG_S2'
    elif not stage1_ok and success and (decision == 'HISTORY'):
        stage1_status = 'S1_WRONG_BUT_FIXED_S2'
    elif not stage1_ok and (not success) and (decision == 'HISTORY'):
        stage1_status = 'S1_WRONG_AND_S2_WRONG'
    elif not stage1_ok and (not success) and (decision == 'DIRECT'):
        stage1_status = 'S1_WRONG_USED'
    else:
        stage1_status = 'OTHER'
    return {'gold_obj': gold_obj, 'pred_obj': pred_obj, 'stage1_top1': stage1_top1, 'stage1_top2': stage1_top2, 'stage1_m1': m1, 'stage1_m2': m2, 'stage1_lead_ratio': lead_ratio, 'stage1_ok': stage1_ok, 'error_source': error_src, 'stage1_status': stage1_status}

def predicate_to_template(pred: str) -> str:
    m = re.match('^\\s*([a-zA-Z_][\\w]*)\\s*\\(([^)]*)\\)\\s*$', pred)
    if not m:
        return pred
    name, args_str = (m.group(1), m.group(2))
    args = [a.strip() for a in args_str.split(',') if a.strip()]
    if name in {'inside', 'on'} and len(args) >= 2:
        return f'{name}({args[0]}, __OBJ__)'
    if name == 'has' and len(args) >= 2:
        args[1] = '__OBJ__'
        return f'{name}({', '.join(args)})'
    if name == 'heated' and len(args) >= 1:
        args[0] = '__OBJ__'
        return f'{name}({', '.join(args)})'
    if name in {'open', 'closed', 'switched_on', 'switched_off'} and len(args) >= 1:
        args[0] = '__FURN__'
        return f'{name}({', '.join(args)})'
    return pred

def dump_results(res: list):
    OUT_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding='utf-8')
try:
    fuzz_log = json.loads((FS_DIR / 'fuzz_log.json').read_text(encoding='utf-8'))
except FileNotFoundError:
    fuzz_log = []

def _norm_text(s: str) -> str:
    return ' '.join(str(s).strip().lower().split())

def find_fuzz_entry(round_num: int, group: int, line: int, target_type: str, nl_original: str):
    norm_curr = _norm_text(nl_original)
    cand_idx = [e for e in fuzz_log if e.get('type') == target_type and e.get('group') == group and (e.get('line') == line) and (round_num == 0 or e.get('round') == round_num)]
    if cand_idx:
        for e in cand_idx:
            if _norm_text(e.get('fuzzy', '')) == norm_curr:
                return e
        return None
    return None
file_path = FS_DIR / TARGET_FILE
if not file_path.exists():
    print(f'[ERROR]  {file_path}', file=sys.stderr)
    sys.exit(1)
results: list[dict] = []
m_round = re.search('round(\\d+)_', file_path.name)
if m_round:
    round_num = int(m_round.group(1))
elif re.search('_l\\d+\\.txt$', file_path.name):
    round_num = 0
else:
    print(f'[WARN]  round file={file_path.name}round_num  0')
    round_num = 0
text = file_path.read_text(encoding='utf-8')
blocks = re.findall('\\[\\s*(.*?)\\s*\\]', text, flags=re.DOTALL)
all_group_cmds: list[list[str]] = []
for blk in blocks:
    cmds = [l.split('|', 1)[0].strip() for l in blk.splitlines() if '|' in l]
    all_group_cmds.append(cmds)
for grp_idx, blk in enumerate(blocks, start=1):
    print(f'Round{round_num} Group{grp_idx}')
    clear_group_history(f'[{grp_idx}]')
    pool = [cmd for i, group in enumerate(all_group_cmds, start=1) if i != grp_idx for cmd in group]
    set_repeat_pool(pool)
    revert_kg_to_backup()
    prev_cmd_asp = None
    prev_cmd_nl = None
    group_gold_history_lines: list[str] = []
    lines = [l.strip() for l in blk.splitlines() if '|' in l]
    for line_idx, raw in enumerate(lines, start=1):
        if '|' not in raw:
            continue
        asp_gold, nl_raw = [p.strip() for p in raw.split('|', 1)]
        nl_original = nl_raw
        nl = replace_multiwords(nl_raw)
        print(f' [{grp_idx}-{line_idx}] {asp_gold} | {nl}')
        extract_initial_conditions()
        remove_initial_conditions_from_asp()
        insert_initial_conditions_to_asp()
        executed_touched = []
        ran_plan = False
        entry = find_fuzz_entry(round_num=round_num, group=grp_idx, line=line_idx, target_type=TARGET_TYPE, nl_original=nl_original)
        if entry is None and round_num == 0 and (line_idx > 1):
            entry = {'type': TARGET_TYPE, 'original': nl_original, 'fuzzy': nl_original, 'predicate': asp_gold, 'fallback': True}
        if entry:
            print(f' {entry['original']}  {entry['fuzzy']}')
            asp_tmpl = predicate_to_template(entry['predicate'])
            pred_cmd, conf, detail, touched_dummy, is_clear = pipeline(nl, prev_cmd_asp, asp_template=asp_tmpl, prev_command_nl=prev_cmd_nl, line_id=f'[{grp_idx}-{line_idx}]', run_planner=False)
            success = pred_cmd == asp_gold
            stage_info = analyse_stage_error(asp_gold=asp_gold, pred_cmd=pred_cmd, success=success, detail=detail)
            if success:
                plan_res_pred = plan(pred_cmd)
                executed_touched = plan_res_pred.get('touched', [])
                ran_plan = True
                prev_cmd_asp = pred_cmd
                prev_cmd_nl = nl
            else:
                print(f' [WARN]  KG/context   {asp_gold}')
                plan_res_gold = plan(asp_gold)
                executed_touched = plan_res_gold.get('touched', [])
                ran_plan = True
                prev_cmd_asp = asp_gold
                prev_cmd_nl = nl
            results.append({'round': round_num, 'group': grp_idx, 'line': line_idx, 'attempt_number': 1, 'type': entry['type'], 'original': entry['original'], 'fuzzy': entry['fuzzy'], 'correct': asp_gold, 'predicted': pred_cmd, 'success': success, 'confidence': conf, 'steps': [], 'cost': None, 'cost_score': None, 'repeat_hits': None, 'semantic': None, 'prev_ctx': detail.get('prev_ctx'), 'curr_ctx': detail.get('curr_ctx'), 'confidence_expr': None, 'obj_candidates': detail.get('obj_candidates'), 'furn_candidates': detail.get('furn_candidates'), 'obj_reason': detail.get('decision_reason'), 'decision': detail.get('decision'), 'fallback_prediction': detail.get('fallback_prediction'), 'eval_error_source': stage_info.get('error_source'), 'stage1_status': stage_info.get('stage1_status'), 'gold_obj': stage_info.get('gold_obj'), 'pred_obj': stage_info.get('pred_obj'), 'stage1_top1': stage_info.get('stage1_top1'), 'stage1_top2': stage_info.get('stage1_top2'), 'stage1_m1': stage_info.get('stage1_m1'), 'stage1_m2': stage_info.get('stage1_m2'), 'stage1_lead_ratio': stage_info.get('stage1_lead_ratio'), 'compare_stage1_failed': bool(detail.get('compare', {}).get('stage1_failed')), 'alt_history_cmd': detail.get('compare', {}).get('asp_history'), 'alt_llm_cmd': detail.get('compare', {}).get('asp_llm'), 'alt_llm_reason': detail.get('compare', {}).get('llm_reason'), 'alt_llm_decision': detail.get('compare', {}).get('llm_decision'), 'alt_history_success': detail.get('compare', {}).get('asp_history') == asp_gold if detail.get('compare', {}).get('asp_history') else False, 'alt_llm_success': detail.get('compare', {}).get('asp_llm') == asp_gold if detail.get('compare', {}).get('asp_llm') else False, 'alt_llm_cmd_full': detail.get('compare', {}).get('asp_llm') if detail.get('compare', {}).get('stage1_failed') else pred_cmd, 'alt_llm_success_full': detail.get('compare', {}).get('asp_llm') == asp_gold if detail.get('compare', {}).get('stage1_failed') else success})
            dump_results(results)
        else:
            try:
                add_group_history(f'[{grp_idx}-{line_idx}]', asp_gold, nl)
            except Exception as e:
                print(' [WARN] add_group_history ', e)
            try:
                execute_user_goal(asp_gold)
                ran_plan = True
                prev_cmd_asp = asp_gold
                prev_cmd_nl = nl
            except Exception as e:
                print(' [WARN] ', e)
        if ran_plan:
            try:
                update_kg_from_asp_outputs(start_holds_file=SHOW_START_OUTPUT, last_holds_file=SHOW_LAST_OUTPUT, changed_holds_file=SHOW_CHANGED_HOLDS_OUTPUT, changed_names_file=None, touched_items=executed_touched, auto_find_changed_names=True)
            except Exception as e:
                print(' [WARN] KG ', e)
        else:
            print(' [WARN] KG ')
        with LOG_FILE.open('a', encoding='utf-8') as f:
            f.write('=======\n')
        if EVAL_USE_GOLD_GROUP_HISTORY:
            group_gold_history_lines.append(f'{asp_gold} | {nl}')
            try:
                replace_group_history(grp_idx, group_gold_history_lines)
            except Exception as e:
                print(f'[WARN] replace_group_history failed at [{grp_idx}-{line_idx}]: {e}')
set_repeat_pool([])
dump_results(results)
print(f'  {OUT_JSON} {LOG_FILE}')
