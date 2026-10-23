"""Tests use a fake LLM, so no paid API is needed."""
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent
import prompts
import tools
from config import APP_DIR, Config, load_config

FIXTURE = APP_DIR / "workspace" / "sample-project"
BUG = "return b - a"
FIX = "return a - b"


# ------------------------------------------------------------------ helpers
def tree_hash(folder: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(folder.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            h.update(p.relative_to(folder).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


class FakeLLM:
    """Answers by prompt type. `fixes` is the list of calculator.py contents to return, in order."""
    def __init__(self, fixes):
        self.fixes = list(fixes)
        self.calls = {"plan": 0, "implement": 0, "repair": 0, "summary": 0}

    def invoke(self, messages):
        system = messages[0].content
        if system == prompts.PLAN_SYSTEM:
            self.calls["plan"] += 1
            body = {"problem_understanding": "subtract looks wrong", "evidence": ["test_subtract fails"],
                    "suspected_cause": "operands swapped", "files_to_modify": ["calculator.py"],
                    "steps": ["fix subtract"], "tests": ["pytest"], "risks": []}
        elif system in (prompts.IMPLEMENT_SYSTEM, prompts.REPAIR_SYSTEM):
            key = "implement" if system == prompts.IMPLEMENT_SYSTEM else "repair"
            self.calls[key] += 1
            content = self.fixes.pop(0) if len(self.fixes) > 1 else self.fixes[0]
            body = {"summary": f"{key} change", "changes": [{"path": "calculator.py", "content": content}]}
        else:
            self.calls["summary"] += 1
            return SimpleNamespace(content="Fake summary of the change.")
        return SimpleNamespace(content=json.dumps(body))


@pytest.fixture()
def env(tmp_path):
    ws = tmp_path / "workspace"
    shutil.copytree(FIXTURE, ws / "sample-project")
    cfg = Config(provider="openai", model="m", api_key="k", workspace_dir=ws, runs_dir=ws / ".runs",
                 timeout_seconds=30)
    return SimpleNamespace(cfg=cfg, ws=ws, repo=ws / "sample-project",
                           good=(FIXTURE / "calculator.py").read_text().replace(BUG, FIX),
                           bad=(FIXTURE / "calculator.py").read_text().replace(BUG, "return 0"),
                           bad2=(FIXTURE / "calculator.py").read_text().replace(BUG, "return 1"),
                           bad3=(FIXTURE / "calculator.py").read_text().replace(BUG, "return 2"))


def run(env, llm, mode=agent.MODE_IMPLEMENT, **cfg_changes):
    cfg = replace(env.cfg, **cfg_changes)
    return agent.run_agent(llm, cfg, "Fix the failing subtract test", "sample-project", mode)


# ------------------------------------------------------------------ basics
def test_app_imports_and_loads_without_api_key(monkeypatch):
    from streamlit.testing.v1 import AppTest
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY"):
        monkeypatch.setenv(name, "")
    at = AppTest.from_file(str(APP_DIR / "app.py"), default_timeout=30).run()
    assert not at.exception
    assert at.title[0].value == "DevPilot Lite"
    assert any("not configured" in w.value.lower() for w in at.warning)


def test_config_handles_missing_api_key(monkeypatch):
    for name in ("LLM_PROVIDER", "LLM_MODEL", "LLM_API_KEY", "LANGSMITH_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    cfg = load_config(load_env_file=False)
    assert not cfg.ready and any("LLM_API_KEY" in p for p in cfg.problems())
    secret = replace(cfg, api_key="sk-very-secret-value")
    assert "sk-very-secret-value" not in repr(secret)


def test_path_traversal_rejected(tmp_path):
    for bad in ("../x.py", "a/../../x.py", "/etc/passwd.py", "C:\\x.py", "..\\x.py", "", ".env", "a.exe"):
        with pytest.raises(tools.PathError):
            tools.validate_relative_path(tmp_path, bad)
    for bad in ("..", "../sample-project", "a/b", ".runs"):
        with pytest.raises(tools.PathError):
            tools.resolve_repository(tmp_path, bad)
    assert tools.validate_relative_path(tmp_path, "pkg/ok.py").name == "ok.py"


def test_secret_files_excluded(tmp_path):
    repo = tmp_path / "r"
    (repo / "node_modules").mkdir(parents=True)
    (repo / "__pycache__").mkdir()
    for name in (".env", "id_rsa", "server.pem", "my_credentials.json", "node_modules/x.js", "__pycache__/a.pyc", "pic.png"):
        (repo / name).write_text("secret")
    (repo / "a.py").write_text("x = 1\n")
    inv = tools.inspect_repository(repo, 1000)
    assert [f["path"] for f in inv["files"]] == ["a.py"]
    assert len(inv["excluded"]) >= 6
    copy = tools.create_working_copy(repo, tmp_path / "runs", 1000)
    assert sorted(p.name for p in copy.rglob("*") if p.is_file()) == ["a.py"]
    with pytest.raises(tools.PathError):
        tools.read_text_file(repo, ".env", 1000)


def test_working_copy_leaves_original_untouched(env):
    before = tree_hash(env.repo)
    copy = tools.create_working_copy(env.repo, env.cfg.runs_dir, 100_000)
    assert copy.parent == env.cfg.runs_dir.resolve() and (copy / "calculator.py").exists()
    (copy / "calculator.py").write_text("changed")
    assert tree_hash(env.repo) == before
    tools.discard_working_copy(env.cfg.runs_dir, copy)
    assert not copy.exists() and env.repo.exists()
    with pytest.raises(tools.PathError):
        tools.discard_working_copy(env.cfg.runs_dir, env.repo)  # only managed copies can be deleted


def test_proposed_paths_validated(env):
    wc = tools.create_working_copy(env.repo, env.cfg.runs_dir, 100_000)
    for bad in ({"path": "../evil.py", "content": "x"}, {"path": "/abs.py", "content": "x"},
                {"path": ".env", "content": "x"}, {"path": "run.exe", "content": "x"}):
        with pytest.raises(tools.PathError):
            tools.propose_file_changes(wc, [bad])
    with pytest.raises(tools.PathError):
        tools.propose_file_changes(wc, [{"path": "big.py", "content": "x" * 200}], max_file_bytes=100)
    ok = tools.propose_file_changes(wc, [{"path": "calculator.py", "content": env.good},
                                         {"path": "notes.md", "content": "hi\n"}])
    assert {p["path"] for p in ok} == {"calculator.py", "notes.md"}
    assert tools.propose_file_changes(wc, [{"path": "calculator.py", "content": (wc / "calculator.py").read_text()}]) == []


def test_unified_diff_correct():
    diff = tools.generate_unified_diff("calculator.py", "def f():\n    return b - a\n", "def f():\n    return a - b\n")
    assert diff.startswith("--- a/calculator.py\n+++ b/calculator.py\n")
    assert "-    return b - a" in diff and "+    return a - b" in diff
    assert tools.generate_unified_diff("x.py", "same\n", "same\n") == ""
    assert tools.generate_unified_diff("n.py", "", "hi\n", is_new=True).startswith("--- /dev/null")


def test_approved_tests_run_with_timeout(tmp_path):
    ok = tmp_path / "ok"
    ok.mkdir()
    (ok / "test_a.py").write_text("def test_a():\n    assert 1 + 1 == 2\n")
    r = tools.run_approved_tests(ok, timeout_seconds=30)
    assert r["ran"] and r["exit_code"] == 0 and r["command"].startswith("python -m pytest")

    slow = tmp_path / "slow"
    slow.mkdir()
    (slow / "test_slow.py").write_text("import time\n\ndef test_slow():\n    time.sleep(20)\n")
    r = tools.run_approved_tests(slow, timeout_seconds=2)
    assert r["timed_out"] and r["exit_code"] is None and r["elapsed"] < 15

    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "a.py").write_text("x = 1\n")
    r = tools.run_approved_tests(empty, timeout_seconds=5)
    assert not r["ran"] and r["exit_code"] is None
    with pytest.raises(tools.PathError):
        tools.run_approved_tests(ok, 5, command_key="bash")


def test_tests_cannot_see_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "sk-should-not-leak")
    (tmp_path / "test_env.py").write_text("import os\n\ndef test_no_key():\n    assert 'LLM_API_KEY' not in os.environ\n")
    assert tools.run_approved_tests(tmp_path, 30)["exit_code"] == 0


# ------------------------------------------------------------------ graph behaviour
def test_routing_functions():
    base = {"repair_attempts": 0, "max_repair_attempts": 1, "tests_ran": True}
    assert agent.route_after_validate({**base, "test_exit_code": 0}) == "report"
    assert agent.route_after_validate({**base, "test_exit_code": 1}) == "repair"
    assert agent.route_after_validate({**base, "test_exit_code": 1, "repair_attempts": 1}) == "report"
    assert agent.route_after_validate({**base, "test_exit_code": None, "tests_ran": False}) == "report"
    assert agent.route_after_plan({"execution_mode": agent.MODE_PLAN_ONLY}) == "report"
    assert agent.route_after_plan({"execution_mode": agent.MODE_IMPLEMENT}) == "implement"
    assert agent.route_after_analyze({"error": "x"}) == "report"


def test_first_fix_passes_without_repair(env):
    llm = FakeLLM([env.good])
    state = run(env, llm)
    assert state["final_status"] == "PASSED" and state["test_exit_code"] == 0
    assert state["repair_attempts"] == 0 and llm.calls["repair"] == 0
    assert "Status: PASSED" in state["final_report"]


def test_failed_test_triggers_repair_and_passes(env):
    before = tree_hash(env.repo)
    llm = FakeLLM([env.bad, env.good])
    state = run(env, llm)
    assert llm.calls["repair"] == 1 and state["repair_attempts"] == 1
    assert [h["exit_code"] for h in state["test_history"]][0] != 0
    assert state["test_exit_code"] == 0 and state["final_status"] == "PASSED"
    assert "Yes - 1" in state["final_report"]
    diff = "".join(p["diff"] for p in state["proposed_changes"])
    assert "-    return b - a" in diff and "+    return a - b" in diff  # diff is against the ORIGINAL
    assert tree_hash(env.repo) == before  # original never modified


def test_repair_loop_terminates_and_reports_failure(env):
    llm = FakeLLM([env.bad, env.bad2])
    state = run(env, llm)
    assert llm.calls["repair"] == 1 and state["repair_attempts"] == 1  # exactly one retry, then stop
    assert len(state["test_history"]) == 2  # tests were really re-run after the repair
    assert state["final_status"] == "FAILED" and state["test_exit_code"] != 0
    assert "Status: FAILED" in state["final_report"] and "Status: PASSED" not in state["final_report"]
    llm2 = FakeLLM([env.bad, env.bad2, env.bad3])
    state2 = run(env, llm2, max_repair_attempts=2)
    assert llm2.calls["repair"] == 2 and state2["final_status"] == "FAILED"
    llm3 = FakeLLM([env.bad, env.bad2])
    state3 = run(env, llm3, max_repair_attempts=0)
    assert llm3.calls["repair"] == 0 and state3["final_status"] == "FAILED"


def test_plan_only_changes_nothing(env):
    before = tree_hash(env.repo)
    llm = FakeLLM([env.good])
    state = run(env, llm, mode=agent.MODE_PLAN_ONLY)
    assert state["final_status"] == "PLAN_ONLY" and state["plan"]
    assert llm.calls["implement"] == 0 and not state["test_history"] and not state["proposed_changes"]
    assert tree_hash(env.repo) == before
    assert not env.cfg.runs_dir.exists() or not any(env.cfg.runs_dir.iterdir())
    assert "no files were changed" in state["final_report"].lower()


def test_no_tests_is_reported_as_unverified(env):
    (env.repo / "test_calculator.py").unlink()
    state = run(env, FakeLLM([env.good]))
    assert state["final_status"] == "UNVERIFIED" and state["test_exit_code"] is None
    assert "Status: UNVERIFIED" in state["final_report"] and "Status: PASSED" not in state["final_report"]
    assert state["repair_attempts"] == 0


def test_path_escape_from_model_is_rejected(env):
    class Evil(FakeLLM):
        def invoke(self, messages):
            if messages[0].content == prompts.IMPLEMENT_SYSTEM:
                return SimpleNamespace(content=json.dumps({"summary": "x", "changes": [{"path": "../../evil.py", "content": "x"}]}))
            return super().invoke(messages)
    state = run(env, Evil([env.good]))
    assert state["final_status"] == "ERROR" and "traversal" in state["error"].lower()
    assert not (env.ws.parent / "evil.py").exists()


def test_bad_task_and_bad_repo_give_clear_errors(env):
    cfg = env.cfg
    state = agent.run_agent(FakeLLM([env.good]), cfg, "  ", "sample-project", agent.MODE_IMPLEMENT)
    assert state["final_status"] == "ERROR" and "describe" in state["error"].lower()
    state = agent.run_agent(FakeLLM([env.good]), cfg, "fix things", "../..", agent.MODE_IMPLEMENT)
    assert state["final_status"] == "ERROR" and "invalid repository" in state["error"].lower()


def test_malformed_model_output_gets_one_retry_then_error(env):
    class Junk(FakeLLM):
        def invoke(self, messages):
            if messages[0].content == prompts.PLAN_SYSTEM:
                self.calls["plan"] += 1
                return SimpleNamespace(content="sorry, no json here")
            return super().invoke(messages)
    llm = Junk([env.good])
    state = run(env, llm)
    assert llm.calls["plan"] == 2 and state["final_status"] == "ERROR"


def test_repair_that_changes_nothing_stops_cleanly(env):
    state = run(env, FakeLLM([env.bad]))  # repair returns identical code -> no change
    assert state["repair_attempts"] == 1 and state["final_status"] == "FAILED"
    assert "no change" in state["error"].lower()
