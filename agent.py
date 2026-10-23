
"""agent.py - the LangGraph workflow.

analyze -> make_plan -> (Plan Only? report) -> implement -> validate
validate: pass -> report | fail + retry left -> repair -> validate
| fail, no retry -> report
"""

import functools
import json
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

import prompts
import tools
from config import Config, redact

MODE_PLAN_ONLY = "Plan Only"
MODE_IMPLEMENT = "Implement & Test"


class AgentState(TypedDict, total=False):
    user_task: str
    repository_path: str
    execution_mode: str
    relevant_files: list
    excluded_files: list
    file_contents: dict
    plan: dict
    proposed_changes: list
    change_summary: str
    working_copy: str
    applied_files: list
    test_command: str
    test_output: str
    test_exit_code: object
    tests_ran: bool
    test_history: list
    validation_summary: str
    repair_attempts: int
    max_repair_attempts: int
    current_stage: str
    activity_log: list
    final_report: str
    final_status: str
    error: str


# ------------------------------------------------------------------
# LLM helpers


class LLMOutputError(Exception):
    pass


class PlanOutput(BaseModel):
    problem_understanding: str = ""
    evidence: list[str] = Field(default_factory=list)
    suspected_cause: str = ""
    files_to_modify: list[str] = Field(default_factory=list)
    steps: list[str] = Field(default_factory=list)
    tests: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class FileChange(BaseModel):
    path: str
    content: str


class ChangeOutput(BaseModel):
    summary: str = ""
    changes: list[FileChange] = Field(default_factory=list)


def build_llm(cfg: Config):
    """Create the chat model for the configured provider."""
    problems = cfg.problems()
    if problems:
        raise ValueError(
            "Model is not configured: " + " ".join(problems)
        )

    # Groq
    if cfg.provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=cfg.model,
            api_key=cfg.api_key,
            temperature=0,
            max_retries=2,
        )

    # OpenAI and OpenAI-compatible APIs
    if cfg.provider == "openai":
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": cfg.model,
            "api_key": cfg.api_key,
        }

        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url

        return ChatOpenAI(**kwargs)

    # Anthropic
    if cfg.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs = {
            "model": cfg.model,
            "api_key": cfg.api_key,
            "max_tokens": 8000,
        }

        if cfg.base_url:
            kwargs["base_url"] = cfg.base_url

        return ChatAnthropic(**kwargs)

    raise ValueError(
        f"Unsupported LLM provider: {cfg.provider}"
    )


def _content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str)
            else block.get("text", "")
            for block in content
            if isinstance(block, (str, dict))
        )

    return str(content)


def extract_json(text: str) -> dict:
    """Extract one JSON object from a model reply."""
    text = text.strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        text,
        re.S,
    )

    if fenced:
        return json.loads(fenced.group(1))

    raise ValueError(
        "No valid JSON object found in the model reply."
    )


def ask_json(llm, system: str, user: str, schema):
    """Call the model and validate its JSON response."""
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=user),
    ]

    last_error = ""

    for _ in range(2):
        reply = _content_to_text(
            llm.invoke(messages).content
        )

        try:
            return schema.model_validate(
                extract_json(reply)
            )

        except ValueError as exc:
            last_error = str(exc)[:300]

            messages += [
                AIMessage(content=reply[:4000]),
                HumanMessage(
                    content=(
                        "That reply was not valid for the required "
                        f"JSON format ({last_error}). "
                        "Reply again with ONLY the JSON object."
                    )
                ),
            ]

    raise LLMOutputError(
        "The model did not return valid JSON twice. "
        f"Last problem: {last_error}"
    )


# ------------------------------------------------------------------
# Small helpers


def _add(log: list, stage: str, message: str) -> None:
    log.append(
        f"{datetime.now():%H:%M:%S} [{stage}] {message}"
    )


def _guard(stage: str, cfg: Config):
    """Capture node errors and store them in the agent state."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state: AgentState):
            try:
                return fn(state)

            except Exception as exc:
                message = redact(
                    f"{stage} step failed: "
                    f"{type(exc).__name__}: {exc}",
                    cfg,
                )[:1500]

                log = list(state.get("activity_log", []))
                _add(log, stage, "ERROR - " + message[:300])

                return {
                    "error": message,
                    "current_stage": stage,
                    "activity_log": log,
                }

        return wrapper

    return decorator


def _current_contents(state: AgentState, cfg: Config) -> dict:
    """Read relevant files from the isolated working copy."""
    wc = state["working_copy"]

    paths = list(
        dict.fromkeys(
            list(state.get("relevant_files", []))
            + [
                item["path"]
                for item in state.get("proposed_changes", [])
            ]
        )
    )

    out = {}
    total = 0

    for path in paths:
        try:
            content = tools.read_text_file(
                wc, path, cfg.max_file_bytes
            )
        except tools.PathError:
            continue

        if total + len(content) <= tools.MAX_TOTAL_CONTEXT_CHARS:
            out[path] = content
            total += len(content)

    return out


def initial_state(
    task: str,
    repository: str,
    mode: str,
    cfg: Config,
) -> AgentState:
    return AgentState(
        user_task=task,
        repository_path=repository,
        execution_mode=mode,
        relevant_files=[],
        excluded_files=[],
        file_contents={},
        plan={},
        proposed_changes=[],
        change_summary="",
        working_copy="",
        applied_files=[],
        test_command="",
        test_output="",
        test_exit_code=None,
        tests_ran=False,
        test_history=[],
        validation_summary="",
        repair_attempts=0,
        max_repair_attempts=cfg.max_repair_attempts,
        current_stage="Starting",
        activity_log=[],
        final_report="",
        final_status="",
        error="",
    )


# ------------------------------------------------------------------
# Routing


def route_after_analyze(state: AgentState) -> str:
    return "report" if state.get("error") else "plan"


def route_after_plan(state: AgentState) -> str:
    if (
        state.get("error")
        or state.get("execution_mode") == MODE_PLAN_ONLY
    ):
        return "report"

    return "implement"


def route_after_implement(state: AgentState) -> str:
    return "report" if state.get("error") else "validate"


def route_after_validate(state: AgentState) -> str:
    if state.get("error") or state.get("test_exit_code") == 0:
        return "report"

    if (
        state.get("tests_ran")
        and state.get("repair_attempts", 0)
        < state.get("max_repair_attempts", 0)
    ):
        return "repair"

    return "report"


def route_after_repair(state: AgentState) -> str:
    return "report" if state.get("error") else "validate"


# ------------------------------------------------------------------
# Final report


def compute_status(state: AgentState) -> str:
    if (
        state.get("execution_mode") == MODE_PLAN_ONLY
        and not state.get("error")
    ):
        return "PLAN_ONLY"

    if state.get("error") and not state.get("tests_ran"):
        return "ERROR"

    if (
        state.get("tests_ran")
        and state.get("test_exit_code") == 0
    ):
        return "PASSED"

    if state.get("tests_ran"):
        return "FAILED"

    return "UNVERIFIED"


STATUS_TEXT = {
    "PASSED": "Tests ran and passed (exit code 0).",
    "FAILED": "Tests ran and did NOT pass. The task is not complete.",
    "UNVERIFIED": "No tests were run, so the changes are NOT verified.",
    "PLAN_ONLY": "Plan Only mode: no files were changed and no code was run.",
    "ERROR": "The agent stopped because of an error. The task is not complete.",
}


def _bullets(items) -> str:
    return (
        "\n".join(f"- {item}" for item in items)
        if items
        else "- (none)"
    )


def build_final_report(
    state: AgentState,
    status: str,
    narrative: str,
) -> str:
    plan = state.get("plan") or {}

    changed = [
        item
        for item in state.get("proposed_changes", [])
        if item.get("diff")
    ]

    history = state.get("test_history", [])

    lines = [
        "## DevPilot Lite - Execution Report",
        f"**Status: {status}** - {STATUS_TEXT[status]}",
        "",
        "### Task received",
        state.get("user_task", ""),
        "",
        "### Files inspected",
        _bullets(state.get("relevant_files", [])),
        "",
        "### Plan followed",
    ]

    if plan:
        lines += [
            plan.get("problem_understanding", ""),
            "",
            "Suspected cause (hypothesis): "
            + (plan.get("suspected_cause") or "none given"),
            "",
            "Steps:",
            _bullets(plan.get("steps", [])),
        ]
    else:
        lines.append("- (no plan was produced)")

    lines += [
        "",
        "### Files changed (in the working copy only)",
        _bullets([
            f"{item['path']} "
            f"({'new file' if item['is_new'] else 'modified'})"
            for item in changed
        ]),
        "",
        "### Summary of changes",
        narrative
        or state.get("change_summary")
        or (
            "No changes were made."
            if not changed
            else "(no summary)"
        ),
        "",
        "### Tests actually executed",
    ]

    if history:
        lines += [
            f"- Run {item['run']}: `{item['command']}` "
            f"-> exit code {item['exit_code']}"
            f"{' (timed out)' if item['timed_out'] else ''}, "
            f"{item['elapsed']}s"
            for item in history
        ]
    else:
        lines.append(
            "- None. "
            + (
                state.get("validation_summary")
                or "No tests were run."
            )
        )

    lines += [
        "",
        "### Actual test results",
        state.get("validation_summary") or "(no test results)",
        "",
        "### Repair attempted",
    ]

    attempts = state.get("repair_attempts", 0)

    lines.append(
        f"Yes - {attempts} of "
        f"{state.get('max_repair_attempts', 0)} allowed."
        if attempts
        else "No."
    )

    issues = []

    if state.get("error"):
        issues.append(state["error"])

    if status == "FAILED":
        issues.append(
            "Tests are still failing after the last run. "
            "Review the diff and test output."
        )

    if status == "UNVERIFIED":
        issues.append("Changes were not verified by tests.")

    if any(
        tools.is_test_file(Path(item["path"]).name)
        for item in changed
    ):
        issues.append(
            "Test files were modified. Check that the tests "
            "were not weakened."
        )

    if changed:
        issues.append(
            "Changes exist only in the temporary working copy; "
            "the original project was not modified."
        )

    lines += [
        "",
        "### Remaining issues",
        _bullets(issues),
    ]

    return "\n".join(lines)


# ------------------------------------------------------------------
# LangGraph workflow


def build_graph(llm, cfg: Config):
    """Build the agent workflow using LangGraph."""

    @_guard("Analyze", cfg)
    def analyze(state: AgentState):
        task = (state.get("user_task") or "").strip()

        if len(task) < 5:
            raise ValueError(
                "Please describe the coding task (at least a few words)."
            )

        if len(task) > 4000:
            raise ValueError(
                "The task is too long (limit 4000 characters)."
            )

        repo = tools.resolve_repository(
            cfg.workspace_dir,
            state.get("repository_path"),
        )

        log = list(state.get("activity_log", []))

        _add(
            log,
            "Analyze",
            f"Task received. Repository: {repo.name}. "
            f"Mode: {state['execution_mode']}.",
        )

        inventory = tools.inspect_repository(
            repo, cfg.max_file_bytes
        )

        if not inventory["files"]:
            raise ValueError(
                "No supported text files found in this repository."
            )

        _add(
            log,
            "Analyze",
            f"Found {len(inventory['files'])} usable files; "
            f"excluded {len(inventory['excluded'])} "
            "(secrets, caches, binaries, large files).",
        )

        chosen = tools.list_relevant_files(
            inventory["files"],
            task,
            cfg.max_files,
        )

        contents = {}
        total = 0

        for item in chosen:
            try:
                content = tools.read_text_file(
                    repo,
                    item["path"],
                    cfg.max_file_bytes,
                )
            except tools.PathError as exc:
                _add(
                    log,
                    "Analyze",
                    f"Skipped {item['path']}: {exc}",
                )
                continue

            if total + len(content) > tools.MAX_TOTAL_CONTEXT_CHARS:
                _add(
                    log,
                    "Analyze",
                    f"Skipped {item['path']}: "
                    "context size limit reached.",
                )
                continue

            contents[item["path"]] = content
            total += len(content)

        _add(
            log,
            "Analyze",
            "Read files: " + ", ".join(contents),
        )

        working_copy = ""

        if state["execution_mode"] == MODE_IMPLEMENT:
            wc = tools.create_working_copy(
                repo,
                cfg.runs_dir,
                cfg.max_file_bytes,
            )

            working_copy = str(wc)

            _add(
                log,
                "Analyze",
                f"Created isolated working copy: .runs/{wc.name}",
            )

        return {
            "relevant_files": list(contents),
            "file_contents": contents,
            "excluded_files": [
                f"{item['path']} ({item['reason']})"
                for item in inventory["excluded"]
            ],
            "working_copy": working_copy,
            "current_stage": "Analyze",
            "activity_log": log,
        }

    @_guard("Plan", cfg)
    def make_plan(state: AgentState):
        log = list(state["activity_log"])

        plan = ask_json(
            llm,
            prompts.PLAN_SYSTEM,
            prompts.build_plan_message(
                state["user_task"],
                state["file_contents"],
            ),
            PlanOutput,
        )

        _add(
            log,
            "Plan",
            f"Plan created: {len(plan.steps)} steps; "
            "files to modify: "
            f"{', '.join(plan.files_to_modify) or 'none'}.",
        )

        if state["execution_mode"] == MODE_PLAN_ONLY:
            _add(
                log,
                "Plan",
                "Plan Only mode - stopping before any change.",
            )

        return {
            "plan": plan.model_dump(),
            "current_stage": "Plan",
            "activity_log": log,
        }

    @_guard("Implement", cfg)
    def implement(state: AgentState):
        log = list(state["activity_log"])

        reply = ask_json(
            llm,
            prompts.IMPLEMENT_SYSTEM,
            prompts.build_implement_message(
                state["user_task"],
                state["plan"],
                state["file_contents"],
            ),
            ChangeOutput,
        )

        proposals = tools.propose_file_changes(
            state["working_copy"],
            [
                change.model_dump()
                for change in reply.changes
            ],
            cfg.max_file_bytes,
        )

        if not proposals:
            raise ValueError(
                "The model did not propose any change that differs "
                "from the current files."
            )

        _add(
            log,
            "Implement",
            "Proposed changes (not yet applied): "
            + ", ".join(item["path"] for item in proposals),
        )

        return {
            "proposed_changes": proposals,
            "change_summary": reply.summary,
            "current_stage": "Implement",
            "activity_log": log,
        }

    @_guard("Validate", cfg)
    def validate(state: AgentState):
        log = list(state["activity_log"])
        wc = Path(state["working_copy"])
        proposals = deepcopy(state["proposed_changes"])

        for item in proposals:
            tools.validate_relative_path(wc, item["path"])

        written = tools.apply_proposed_changes(
            wc, proposals
        )

        applied = sorted(
            set(state.get("applied_files", []))
            | set(written)
        )

        _add(
            log,
            "Validate",
            "Applied to working copy: "
            + (", ".join(written) or "nothing new"),
        )

        result = tools.run_approved_tests(
            wc, cfg.timeout_seconds
        )

        output = result["stdout"]

        if result["stderr"].strip():
            output += "\n[stderr]\n" + result["stderr"]

        history = list(state.get("test_history", []))

        if result["ran"]:
            history.append({
                "run": len(history) + 1,
                "command": result["command"],
                "exit_code": result["exit_code"],
                "timed_out": result["timed_out"],
                "elapsed": result["elapsed"],
            })

            code = result["exit_code"]

            if result["timed_out"]:
                summary = result["note"]
            elif code == 0:
                summary = (
                    f"Tests passed: exit code 0 "
                    f"in {result['elapsed']}s."
                )
            elif code == 5:
                summary = (
                    "pytest found no tests to run "
                    f"(exit code 5) in {result['elapsed']}s."
                )
            else:
                summary = (
                    f"Tests FAILED: exit code {code} "
                    f"in {result['elapsed']}s."
                )

            _add(
                log,
                "Validate",
                f"Ran `{result['command']}` -> {summary}",
            )

        else:
            summary = (
                result["note"]
                + " The changes are NOT verified."
            )

            _add(log, "Validate", summary)

        if any(
            tools.is_test_file(Path(item["path"]).name)
            for item in proposals
            if item["diff"]
        ):
            summary += (
                " Note: test files were modified - "
                "check the tests were not weakened."
            )

        return {
            "proposed_changes": proposals,
            "applied_files": applied,
            "test_command": result["command"],
            "test_output": output or result["note"],
            "test_exit_code": result["exit_code"],
            "tests_ran": result["ran"],
            "test_history": history,
            "validation_summary": summary,
            "current_stage": "Validate",
            "activity_log": log,
        }

    def repair(state: AgentState):
        attempt = state.get("repair_attempts", 0) + 1
        log = list(state["activity_log"])

        try:
            _add(
                log,
                "Repair",
                f"Tests failed. Repair attempt {attempt} "
                f"of {state['max_repair_attempts']}.",
            )

            reply = ask_json(
                llm,
                prompts.REPAIR_SYSTEM,
                prompts.build_repair_message(
                    state["user_task"],
                    state["test_command"],
                    state["test_output"],
                    _current_contents(state, cfg),
                    state.get("change_summary", ""),
                ),
                ChangeOutput,
            )

            new = tools.propose_file_changes(
                state["working_copy"],
                [
                    change.model_dump()
                    for change in reply.changes
                ],
                cfg.max_file_bytes,
            )

            if not new:
                raise ValueError(
                    "The repair step proposed no change, "
                    "so the failing tests were left as they are."
                )

            _add(
                log,
                "Repair",
                "Proposed correction: "
                + ", ".join(item["path"] for item in new),
            )

            return {
                "repair_attempts": attempt,
                "proposed_changes": tools.merge_proposals(
                    state["proposed_changes"], new
                ),
                "change_summary": (
                    state.get("change_summary", "")
                    + " Repair: "
                    + reply.summary
                ).strip(),
                "current_stage": "Repair",
                "activity_log": log,
            }

        except Exception as exc:
            message = redact(
                f"Repair step failed: "
                f"{type(exc).__name__}: {exc}",
                cfg,
            )[:1500]

            _add(log, "Repair", "ERROR - " + message[:300])

            return {
                "repair_attempts": attempt,
                "error": message,
                "current_stage": "Repair",
                "activity_log": log,
            }

    def write_report(state: AgentState):
        log = list(state.get("activity_log", []))
        status = compute_status(state)

        changed = [
            item
            for item in state.get("proposed_changes", [])
            if item.get("diff")
        ]

        narrative = ""

        if changed and status != "PLAN_ONLY":
            try:
                facts = {
                    "status": status,
                    "task": state.get("user_task"),
                    "files_changed": [
                        item["path"] for item in changed
                    ],
                    "tests_run": state.get("test_history", []),
                    "repair_attempts": state.get(
                        "repair_attempts", 0
                    ),
                    "error": state.get("error", ""),
                    "model_change_summary": state.get(
                        "change_summary", ""
                    ),
                    "diff": "".join(
                        item["diff"] for item in changed
                    )[:6000],
                }

                reply = llm.invoke([
                    SystemMessage(
                        content=prompts.FINAL_SUMMARY_SYSTEM
                    ),
                    HumanMessage(
                        content=prompts.build_final_summary_message(
                            facts
                        )
                    ),
                ])

                narrative = _content_to_text(
                    reply.content
                ).strip()[:1500]

            except Exception:
                # Use the recorded summary if the LLM summary fails.
                narrative = ""

        _add(log, "Report", f"Final status: {status}.")

        return {
            "final_status": status,
            "final_report": build_final_report(
                state, status, narrative
            ),
            "current_stage": "Complete",
            "activity_log": log,
        }

    graph = StateGraph(AgentState)

    graph.add_node("analyze", analyze)
    graph.add_node("make_plan", make_plan)
    graph.add_node("implement", implement)
    graph.add_node("validate", validate)
    graph.add_node("repair", repair)
    graph.add_node("write_report", write_report)

    graph.add_edge(START, "analyze")

    graph.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            "plan": "make_plan",
            "report": "write_report",
        },
    )

    graph.add_conditional_edges(
        "make_plan",
        route_after_plan,
        {
            "implement": "implement",
            "report": "write_report",
        },
    )

    graph.add_conditional_edges(
        "implement",
        route_after_implement,
        {
            "validate": "validate",
            "report": "write_report",
        },
    )

    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "repair": "repair",
            "report": "write_report",
        },
    )

    graph.add_conditional_edges(
        "repair",
        route_after_repair,
        {
            "validate": "validate",
            "report": "write_report",
        },
    )

    graph.add_edge("write_report", END)

    return graph.compile()


def run_agent(
    llm,
    cfg: Config,
    task: str,
    repository: str,
    mode: str,
) -> AgentState:
    """Run the complete graph and return its final state."""
    return build_graph(llm, cfg).invoke(
        initial_state(task, repository, mode, cfg)
    )
