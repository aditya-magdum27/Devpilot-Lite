"""tools.py - small, safe helpers the agent is allowed to use.

Every filesystem helper checks paths itself (not only the UI).
The model never gets a shell, a Python REPL, or network access.
"""
import difflib
import fnmatch
import io
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

MAX_TOTAL_CONTEXT_CHARS = 60_000  # total text we send to the model per prompt
MAX_CHANGED_FILES = 5
OUTPUT_LIMIT = 6_000  # keep only the tail of test output

EXCLUDED_DIRS = {
    ".git", ".hg", ".svn", "venv", ".venv", "env", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox", ".idea", ".vscode",
    "dist", "build", ".runs", "site-packages",
}
SECRET_PATTERNS = (
    "*.pem", "*.key", "*.p12", "*.pfx", "id_rsa*", "id_ed25519*", "*credentials*",
    "*secret*", ".netrc", "*.sqlite", "*.db",
)
ALLOWED_SUFFIXES = {
    ".py", ".md", ".txt", ".toml", ".cfg", ".ini", ".json", ".yaml", ".yml",
    ".html", ".css", ".js", ".ts",
}


class PathError(ValueError):
    """Raised when a path or file is not allowed."""


# ---------------------------------------------------------------- path safety
def is_secret_name(name: str) -> bool:
    n = name.lower()
    return n.startswith(".env") or any(fnmatch.fnmatch(n, p) for p in SECRET_PATTERNS)


def is_test_file(name: str) -> bool:
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py"))


def validate_relative_path(base_dir, rel_path) -> Path:
    """Return the safe absolute path for rel_path inside base_dir, or raise PathError."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise PathError("Empty file path.")
    rel = rel_path.strip().replace("\\", "/")
    if "\x00" in rel:
        raise PathError("Invalid character in path.")
    if rel.startswith("/") or (len(rel) > 1 and rel[1] == ":"):
        raise PathError(f"Absolute paths are not allowed: {rel_path}")
    parts = PurePosixPath(rel).parts
    if not parts or ".." in parts:
        raise PathError(f"Path traversal is not allowed: {rel_path}")
    if any(p in EXCLUDED_DIRS for p in parts[:-1]) or parts[-1] in EXCLUDED_DIRS:
        raise PathError(f"Path is inside an excluded folder: {rel_path}")
    if is_secret_name(parts[-1]):
        raise PathError(f"Secret/credential files are not allowed: {rel_path}")
    base = Path(base_dir).resolve()
    target = (base / rel).resolve()  # resolve() also follows symlinks
    if target == base or base not in target.parents:
        raise PathError(f"Path is outside the working folder: {rel_path}")
    if target.suffix.lower() not in ALLOWED_SUFFIXES:
        raise PathError(f"Unsupported file type '{target.suffix}': {rel_path}")
    return target


def resolve_repository(workspace_dir, name) -> Path:
    """A repository must be a direct sub-folder of the workspace directory."""
    ws = Path(workspace_dir).resolve()
    if not isinstance(name, str) or not name or name.startswith(".") or any(c in name for c in "/\\:"):
        raise PathError(f"Invalid repository name: {name!r}")
    repo = (ws / name).resolve()
    if repo.parent != ws or not repo.is_dir():
        raise PathError(f"Repository must be a folder directly inside the workspace: {name!r}")
    return repo


def list_workspace_repos(workspace_dir) -> list[str]:
    ws = Path(workspace_dir)
    if not ws.is_dir():
        return []
    return sorted(p.name for p in ws.iterdir() if p.is_dir() and not p.name.startswith((".", "__")))


# ------------------------------------------------------------ upload a project (.zip)
MAX_ZIP_BYTES = 20 * 1024 * 1024
MAX_UNZIPPED_BYTES = 100 * 1024 * 1024
MAX_ZIP_FILES = 2000
_SKIP_PARTS = {".git", "node_modules", "__pycache__", ".venv", "venv", "ml-env", ".pytest_cache"}


def save_uploaded_zip(workspace_dir, data: bytes, filename: str) -> str:
    """Safely unzip an uploaded project into workspace/<name>. Returns the folder name."""
    if len(data) > MAX_ZIP_BYTES:
        raise PathError("Zip is too big (max 20 MB).")
    base = re.sub(r"[^A-Za-z0-9_-]", "-", Path(filename).stem).strip("-")[:40] or "project"
    ws = Path(workspace_dir).resolve()
    ws.mkdir(exist_ok=True)
    name, n = base, 2
    while (ws / name).exists():
        name, n = f"{base}-{n}", n + 1
    dest = ws / name
    try:
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise PathError("This is not a valid zip file.")
        with zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            if len(infos) > MAX_ZIP_FILES or sum(i.file_size for i in infos) > MAX_UNZIPPED_BYTES:
                raise PathError("Zip has too many files or is too large after unzipping.")
            dest.mkdir()
            for i in infos:
                parts = Path(i.filename).parts
                if i.filename.startswith(("/", "\\")) or ".." in parts or ":" in i.filename:
                    raise PathError(f"Unsafe path in zip: {i.filename}")
                if any(p in _SKIP_PARTS for p in parts) or parts[-1].lower().startswith(".env"):
                    continue  # skip junk folders and secret files
                target = (dest / i.filename).resolve()
                if dest not in target.parents:
                    raise PathError(f"Unsafe path in zip: {i.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(i))
        kids = list(dest.iterdir())
        if not kids:
            raise PathError("Zip had no usable files.")
        if len(kids) == 1 and kids[0].is_dir():  # zip had one top folder -> move contents up
            inner = kids[0]
            for item in inner.iterdir():
                shutil.move(str(item), str(dest / item.name))
            inner.rmdir()
    except Exception:
        shutil.rmtree(dest, ignore_errors=True)
        raise
    return name


# ------------------------------------------------------------ inspect / read
def inspect_repository(repo_dir, max_file_bytes: int) -> dict:
    """Walk the repo and return usable text files plus what was excluded (and why)."""
    repo = Path(repo_dir).resolve()
    files, excluded = [], []
    for root, dirs, names in os.walk(repo, followlinks=False):
        rel_root = Path(root).relative_to(repo)
        keep = []
        for d in sorted(dirs):
            if d in EXCLUDED_DIRS or d.endswith(".egg-info") or (Path(root) / d).is_symlink():
                excluded.append({"path": (rel_root / d).as_posix() + "/", "reason": "cache/dependency/venv folder or symlink"})
            else:
                keep.append(d)
        dirs[:] = keep
        for n in sorted(names):
            rel = (rel_root / n).as_posix()
            full = Path(root) / n
            if is_secret_name(n):
                excluded.append({"path": rel, "reason": "secret/credential file"})
            elif full.is_symlink():
                excluded.append({"path": rel, "reason": "symlink"})
            elif full.suffix.lower() not in ALLOWED_SUFFIXES:
                excluded.append({"path": rel, "reason": "unsupported/binary file type"})
            elif full.stat().st_size > max_file_bytes:
                excluded.append({"path": rel, "reason": "file too large"})
            else:
                files.append({"path": rel, "size": full.stat().st_size, "is_test": is_test_file(n)})
    return {"files": files, "excluded": excluded}


def list_relevant_files(files: list[dict], task: str, max_files: int) -> list[dict]:
    """Rank files by simple keyword overlap with the task; keep at most max_files."""
    words = {w for w in "".join(c.lower() if c.isalnum() else " " for c in task).split() if len(w) >= 3}

    def score(f):
        path = f["path"].lower()
        s = sum(2 for w in words if w in path)
        s += 2 if f["is_test"] else 0
        s += 1 if path.endswith(".py") else 0
        return s

    return sorted(files, key=lambda f: (-score(f), f["path"]))[:max_files]


def read_text_file(base_dir, rel_path, max_bytes: int) -> str:
    """Read one UTF-8 text file (size-limited). Line endings are normalised to \\n."""
    target = validate_relative_path(base_dir, rel_path)
    if not target.is_file():
        raise PathError(f"Not a file: {rel_path}")
    if target.stat().st_size > max_bytes:
        raise PathError(f"File too large (> {max_bytes} bytes): {rel_path}")
    data = target.read_bytes()
    if b"\x00" in data:
        raise PathError(f"Binary file: {rel_path}")
    try:
        return data.decode("utf-8").replace("\r\n", "\n")
    except UnicodeDecodeError as exc:
        raise PathError(f"Not UTF-8 text: {rel_path}") from exc


# ------------------------------------------------------- diffs and proposals
def generate_unified_diff(path: str, old: str, new: str, is_new: bool = False) -> str:
    if old == new:
        return ""
    lines = list(difflib.unified_diff(
        old.splitlines(), new.splitlines(),
        fromfile="/dev/null" if is_new else f"a/{path}", tofile=f"b/{path}", lineterm="",
    ))
    if not lines:
        return f"--- a/{path}\n+++ b/{path}\n(only end-of-file/whitespace differences)\n"
    return "\n".join(lines) + "\n"


def propose_file_changes(base_dir, changes: list[dict], max_file_bytes: int = 100_000) -> list[dict]:
    """Validate model-suggested changes and build proposals (nothing is written here)."""
    if len(changes) > MAX_CHANGED_FILES:
        raise PathError(f"Too many files changed ({len(changes)}). Limit is {MAX_CHANGED_FILES}.")
    base = Path(base_dir).resolve()
    proposals, seen = [], set()
    for change in changes:
        target = validate_relative_path(base, change.get("path"))
        rel = target.relative_to(base).as_posix()
        if rel in seen:
            raise PathError(f"Same file proposed twice: {rel}")
        seen.add(rel)
        content = change.get("content")
        if not isinstance(content, str) or "\x00" in content:
            raise PathError(f"Invalid content for {rel}.")
        if len(content.encode("utf-8")) > max_file_bytes:
            raise PathError(f"Proposed content too large for {rel}.")
        content = content.replace("\r\n", "\n")
        is_new = not target.exists()
        original = "" if is_new else read_text_file(base, rel, max_file_bytes)
        if content == original:
            continue  # no-op change
        proposals.append({
            "path": rel, "original": original, "new": content, "is_new": is_new,
            "diff": generate_unified_diff(rel, original, content, is_new), "applied": False,
        })
    return proposals


def merge_proposals(existing: list[dict], new: list[dict]) -> list[dict]:
    """Combine a repair's proposals with earlier ones; diffs always compare to the ORIGINAL file."""
    by_path = {p["path"]: dict(p) for p in existing}
    for p in new:
        old = by_path.get(p["path"])
        original = old["original"] if old else p["original"]
        is_new = old["is_new"] if old else p["is_new"]
        by_path[p["path"]] = dict(
            p, original=original, is_new=is_new, applied=False,
            diff=generate_unified_diff(p["path"], original, p["new"], is_new),
        )
    return list(by_path.values())


def apply_proposed_changes(base_dir, proposals: list[dict]) -> list[str]:
    """Write not-yet-applied proposals into base_dir (re-checking every path). Returns paths written."""
    written = []
    for p in proposals:
        if p.get("applied"):
            continue
        target = validate_relative_path(base_dir, p["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(p["new"], encoding="utf-8", newline="\n")
        p["applied"] = True
        written.append(p["path"])
    return written


def build_patch(proposals: list[dict]) -> str:
    """All diffs in one text, usable with `git apply` or `patch -p1` inside the original repo."""
    return "".join(p["diff"] for p in proposals if p.get("diff"))


# ------------------------------------------------------------ working copies
def create_working_copy(repo_dir, runs_dir, max_file_bytes: int) -> Path:
    """Copy permitted files into a fresh folder under runs_dir. The original is never touched."""
    repo = Path(repo_dir).resolve()
    runs = Path(runs_dir).resolve()
    runs.mkdir(parents=True, exist_ok=True)
    dest = runs / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    dest.mkdir()
    for f in inspect_repository(repo, max_file_bytes)["files"]:
        dst = dest / f["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / f["path"], dst)
    return dest


def _inside_runs(runs_dir, path) -> Path:
    runs = Path(runs_dir).resolve()
    target = Path(path).resolve()
    if target.parent != runs or not target.is_dir():
        raise PathError("Only managed working copies can be deleted.")
    return target


def discard_working_copy(runs_dir, path) -> None:
    shutil.rmtree(_inside_runs(runs_dir, path), ignore_errors=True)


def cleanup_old_runs(runs_dir, keep: int = 10) -> None:
    """Retention policy: keep the newest `keep` working copies, delete older managed ones."""
    runs = Path(runs_dir)
    if not runs.is_dir():
        return
    folders = sorted(p for p in runs.iterdir() if p.is_dir())
    for old in folders[:-keep] if keep > 0 else folders:
        shutil.rmtree(_inside_runs(runs_dir, old), ignore_errors=True)


# ------------------------------------------------------------- run tests
def _approved_pytest(targets: list[str]) -> list[str]:
    return [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *targets]


APPROVED_COMMANDS = {"pytest": _approved_pytest}  # the only command the agent can run
_SAFE_ENV_KEYS = ("PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT",
                  "TEMP", "TMP", "HOME", "USERPROFILE", "LANG", "LC_ALL")


def _safe_env() -> dict:
    """Minimal environment for test runs, so API keys are not visible to the code under test."""
    env = {k: os.environ[k] for k in _SAFE_ENV_KEYS if k in os.environ}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _to_text(value) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", "replace") if isinstance(value, bytes) else value


def find_test_targets(work_dir, max_file_bytes: int = 1_000_000) -> list[str]:
    """Find test files ourselves (the model never picks the command or its arguments)."""
    found = [f["path"] for f in inspect_repository(work_dir, max_file_bytes)["files"] if f["is_test"]]
    return sorted(found, key=lambda p: (p.count("/"), p))[:10]


def run_approved_tests(work_dir, timeout_seconds: int, command_key: str = "pytest") -> dict:
    """Run an approved test command with subprocess (argument list, no shell, with timeout)."""
    if command_key not in APPROVED_COMMANDS:
        raise PathError(f"Command not approved: {command_key}")
    result = {"ran": False, "command": "", "exit_code": None, "stdout": "", "stderr": "",
              "elapsed": 0.0, "timed_out": False, "note": ""}
    targets = find_test_targets(work_dir)
    if not targets:
        result["note"] = "No pytest test files (test_*.py or *_test.py) found. Nothing was run."
        return result
    argv = APPROVED_COMMANDS[command_key](targets)
    result["command"] = " ".join(["python"] + argv[1:])
    result["ran"] = True
    start = time.perf_counter()
    try:
        done = subprocess.run(argv, cwd=str(work_dir), capture_output=True, text=True, shell=False,
                              timeout=timeout_seconds, env=_safe_env(), encoding="utf-8", errors="replace")
        result.update(exit_code=done.returncode, stdout=done.stdout, stderr=done.stderr)
    except subprocess.TimeoutExpired as exc:
        result.update(timed_out=True, stdout=_to_text(exc.stdout), stderr=_to_text(exc.stderr),
                      note=f"Tests timed out after {timeout_seconds}s and were stopped.")
    result["elapsed"] = round(time.perf_counter() - start, 2)
    result["stdout"] = result["stdout"][-OUTPUT_LIMIT:]
    result["stderr"] = result["stderr"][-OUTPUT_LIMIT:]
    return result