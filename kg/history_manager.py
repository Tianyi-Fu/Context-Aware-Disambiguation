import os
import shutil
from typing import List
from config.config import HISTORY_FILE, BACKUP_FILE
ACTIVE_HISTORY_FILE: str = HISTORY_FILE

def set_active_history_file(path: str | None) -> None:
    global ACTIVE_HISTORY_FILE
    if path and os.path.exists(path):
        ACTIVE_HISTORY_FILE = path
        print(f'[HIST] ACTIVE_HISTORY_FILE set to {ACTIVE_HISTORY_FILE}')
    else:
        ACTIVE_HISTORY_FILE = HISTORY_FILE
        print(f'[HIST] ACTIVE_HISTORY_FILE reset to default {HISTORY_FILE}')

def get_active_history_file() -> str:
    return ACTIVE_HISTORY_FILE

def _iter_history_lines(path: str) -> List[str]:
    lines: List[str] = []
    if not os.path.exists(path):
        return lines
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('[') or line.startswith(']'):
                lines.append(line)
                continue
            if line.startswith('===') and 'New Session' not in line:
                continue
            if '|' not in line and line != '=== New Session ===':
                continue
            lines.append(line)
    return lines

def load_user_history() -> List[str]:
    path = get_active_history_file()
    return _iter_history_lines(path)

def load_all_sessions():
    path = get_active_history_file()
    if not os.path.exists(path):
        return []
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    raw_blocks = content.split('=== New Session ===')
    sessions = []
    for block in raw_blocks:
        block_lines = []
        for line in block.strip().splitlines():
            l = line.strip()
            if l and l not in ('[', ']') and ('|' in l):
                block_lines.append(l)
        if block_lines:
            sessions.append(block_lines)
    return sessions

def append_to_history(asp_command: str, user_command: str) -> None:
    path = get_active_history_file()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(f'{asp_command} | {user_command}\n')

def reset_history_from_backup() -> None:
    if os.path.exists(BACKUP_FILE):
        shutil.copyfile(BACKUP_FILE, HISTORY_FILE)
        print('[INFO] User history has been reset from backup.')
    else:
        print(f'[WARN] Backup file not found: {BACKUP_FILE}. Skipping reset.')

def extract_asp_command(line: str) -> str:
    return line.split('|', 1)[0].strip()
