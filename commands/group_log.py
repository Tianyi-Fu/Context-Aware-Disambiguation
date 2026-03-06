from __future__ import annotations
import json, time, re
from pathlib import Path
from typing import Optional
_JSONL_PATH: Optional[Path] = None
_TXT_PATH: Optional[Path] = None
_ID_RE = re.compile('\\[\\s*(\\d+)\\s*-\\s*(\\d+)\\s*\\]')

def init_group_logs(base_dir: str, reset: bool=True) -> None:
    global _JSONL_PATH, _TXT_PATH
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    _JSONL_PATH = base / 'group_history.jsonl'
    _TXT_PATH = base / 'group_history.txt'
    if reset:
        _JSONL_PATH.write_text('', encoding='utf-8')
        _TXT_PATH.write_text('', encoding='utf-8')
    print(f'[INIT] logs reset={reset} @ {_JSONL_PATH} / {_TXT_PATH}')

def _ensure_inited():
    if _JSONL_PATH is None or _TXT_PATH is None:
        raise RuntimeError('group_log not initialized. Call init_group_logs(base_dir, reset=...) first.')

def _parse_line_id(line_id: str) -> tuple[str, int]:
    m = _ID_RE.search(line_id or '')
    if not m:
        raise ValueError(f'Bad line_id: {line_id!r}')
    return (m.group(1), int(m.group(2)))

def _fmt_line(asp_cmd: str, nl: str) -> str:
    return f'{asp_cmd:<30s} | {nl}'

def log_line(line_id: str, asp_cmd: str, nl: str) -> None:
    _ensure_inited()
    g, idx = _parse_line_id(line_id)
    line = _fmt_line(asp_cmd, nl)
    with _JSONL_PATH.open('a', encoding='utf-8') as f:
        f.write(json.dumps({'type': 'entry', 'group': g, 'idx': idx, 'line': line, 'ts': int(time.time())}, ensure_ascii=False) + '\n')
    with _TXT_PATH.open('a', encoding='utf-8') as f:
        f.write(f'[{g}-{idx}] {line}\n')

def print_and_log(line_id: str, asp_cmd: str, nl: str) -> None:
    header = f'{line_id} {asp_cmd:<30s} | {nl}'
    print(header)
    log_line(line_id, asp_cmd, nl)

def log_from_header_printed(header_line: str) -> None:
    _ensure_inited()
    m = _ID_RE.search(header_line)
    if not m:
        raise ValueError(f'Header line does not contain [g-k]: {header_line!r}')
    g, idx = (m.group(1), int(m.group(2)))
    parts = header_line.split('|', 1)
    if len(parts) != 2:
        raise ValueError(f"Header line missing '|': {header_line!r}")
    left = parts[0]
    asp_cmd = left[left.find(']') + 1:].strip()
    nl = parts[1].strip()
    log_line(f'[{g}-{idx}]', asp_cmd, nl)
