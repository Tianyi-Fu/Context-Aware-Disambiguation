import os
import sys
import subprocess

from config.config import (
    PROJECT_ROOT,
    SPARC_JAR_PATH,
    ASP_FILE,
    OCCURS_OUTPUT,
    SHOW_START_OUTPUT,
    SHOW_LAST_OUTPUT,
    SHOW_CHANGED_HOLDS_OUTPUT,
    SHOW_CHANGED_HOLDS_NAME_OUTPUT,
    SHOW_CHANGED_HOLDS_NAME_OUTPUT_USER,
    OPERATED_OUTPUT,
)

SHOW_OPERATED_HOLDS_NAME_OUTPUT = OPERATED_OUTPUT

def _ensure_cwd():
    if os.getcwd() != PROJECT_ROOT:
        os.chdir(PROJECT_ROOT)
        print(f"[CTX] chdir -> {PROJECT_ROOT}")

def update_n_in_file(asp_file, n):
    try:
        with open(asp_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(asp_file, 'w', encoding='utf-8') as f:
            for line in lines:
                if line.startswith("#const n ="):
                    f.write(f"#const n = {n}.\n")
                else:
                    f.write(line)
    except Exception as e:
        print(f"[ERR] update_n_in_file: {e}")

def remove_generated_display_statements(asp_file):
    try:
        with open(asp_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        with open(asp_file, 'w', encoding='utf-8') as f:
            for line in lines:
                if line.strip() == "% Automatically generated display statement" or line.startswith("display"):
                    continue
                f.write(line)
        remove_blank_lines_from_file(asp_file)
    except Exception as e:
        print(f"[ERR] remove_generated_display_statements: {e}")

def add_display_to_asp_file(asp_file, display_predicate):
    try:
        with open(asp_file, 'r', encoding='utf-8') as f:
            asp_code = f.read()
        asp_code = "\n".join([line for line in asp_code.splitlines() if not line.startswith("display")])
        asp_code += f"\n\n% Automatically generated display statement\n"
        asp_code += f"display {display_predicate}.\n"
        with open(asp_file, 'w', encoding='utf-8') as f:
            f.write(asp_code)
    except Exception as e:
        print(f"[ERR] add_display_to_asp_file: {e}")

def run_sparc_with_output_to_file(sparc_jar_path, asp_file, output_filename):
    os.makedirs(os.path.dirname(output_filename), exist_ok=True)
    cmd = ["java", "-jar", sparc_jar_path, asp_file, "-A"]
    with open(output_filename, "w", encoding="utf-8") as out:
        proc = subprocess.run(cmd, stdout=out, stderr=subprocess.PIPE, cwd=PROJECT_ROOT)
    if proc.returncode != 0:
        err = proc.stderr.decode(errors="ignore")
        print(f"[ERR] SPARC: {err}")
        raise RuntimeError(f"SPARC failed with code {proc.returncode}")

def check_output_file(output_filename, required_content=""):
    try:
        if os.path.exists(output_filename):
            with open(output_filename, 'r', encoding='utf-8') as f:
                contents = f.read()
            if required_content:
                return required_content in contents
            return bool(contents.strip())
        return False
    except Exception as e:
        print(f"[ERR] check_output_file: {e}")
        return False

def print_file_contents(filename, predicate):
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                contents = f.read().strip()
            if contents:
                print(f"\n----- {predicate.upper()} OUTPUT -----\n")
                print(contents)
    except Exception as e:
        print(f"[ERR] print_file_contents: {e}")

def remove_blank_lines_from_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        cleaned = [line for line in lines if line.strip()]
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned)
    except Exception as e:
        print(f"[ERR] remove_blank_lines_from_file: {e}")

def _outputs_map():
    return {
        "occurs": OCCURS_OUTPUT,
        "show_operated_holds_name": SHOW_OPERATED_HOLDS_NAME_OUTPUT,
        "show_changed_holds": SHOW_CHANGED_HOLDS_OUTPUT,
        "show_changed_holds_name": SHOW_CHANGED_HOLDS_NAME_OUTPUT,
        "show_start_holds": SHOW_START_OUTPUT,
        "show_last_holds": SHOW_LAST_OUTPUT,
    }

def get_minimal_n(max_n=15):
    _ensure_cwd()

    display_env = os.environ.get("DISPLAY_PREDICATES", "")
    if display_env.strip():
        display_preds = [s.strip() for s in display_env.split(",") if s.strip()]
    else:
        display_preds = ["occurs", "show_operated_holds_name", "show_changed_holds",
                         "show_changed_holds_name", "show_start_holds", "show_last_holds"]

    out_map = _outputs_map()

    n = 1
    while True:
        if n > max_n:
            raise RuntimeError(f"Solution not found with n <= {max_n}.")
        print(f"Trying with n={n}")
        update_n_in_file(ASP_FILE, n)

        add_display_to_asp_file(ASP_FILE, "occurs")
        run_sparc_with_output_to_file(SPARC_JAR_PATH, ASP_FILE, out_map["occurs"])

        if check_output_file(out_map["occurs"], required_content="occurs"):
            print(f"Solution found with n={n}")
            break
        else:
            if os.path.exists(out_map["occurs"]):
                try: os.remove(out_map["occurs"])
                except Exception: pass
            remove_generated_display_statements(ASP_FILE)
            n += 1

    return n

def main():
    _ensure_cwd()

    display_env = os.environ.get("DISPLAY_PREDICATES", "")
    if display_env.strip():
        display_preds = [s.strip() for s in display_env.split(",") if s.strip()]
    else:
        display_preds = ["occurs", "show_operated_holds_name", "show_changed_holds",
                         "show_changed_holds_name", "show_start_holds", "show_last_holds"]

    out_map = _outputs_map()

    n = get_minimal_n(max_n=15)
    print(f"Solution found with n={n}")
    print_file_contents(out_map["occurs"], "occurs")

    for pred in display_preds:
        if pred == "occurs":
            continue
        add_display_to_asp_file(ASP_FILE, pred)
        run_sparc_with_output_to_file(SPARC_JAR_PATH, ASP_FILE, out_map[pred])
        if pred in {"show_start_holds", "show_last_holds"}:
            print(f"[INFO] saved {pred} to {out_map[pred]} (not printing to reduce log noise)")
        else:
            print_file_contents(out_map[pred], pred)
        remove_generated_display_statements(ASP_FILE)

    print(f"The minimal value of n for the solution is: {n}")

if __name__ == "__main__":
    main()
