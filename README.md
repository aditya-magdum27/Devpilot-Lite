# DevPilot Lite - Autonomous Software Engineering Agent (small version)

DevPilot Lite is a small AI agent with a Streamlit dashboard. You give it a coding task for a
project, and it **reads the code, makes a plan, changes the code in a safe copy, runs the tests,
and tries one repair if the tests fail**. It then writes an honest report.

## Problem statement
Fixing a bug by hand means: read code, think, edit, run tests, repeat. DevPilot Lite automates
that loop, but with safety rules: the agent can only touch a temporary copy, only run approved
test commands, and never claims success unless the tests really passed.

## Features
- Plan Only mode (plan, no changes) and Implement & Test mode.
- Real LangGraph workflow with conditional routing and a **bounded** repair loop (default 1 retry).
- Diff view for every changed file, real test output, real pass/fail status.
- Safe: isolated working copy, path checks, secret files excluded, no shell for the model.
- Download of the proposed changes as a `.patch` file.
- Works with OpenAI or Anthropic models (one configured in `.env`). LangSmith tracing is optional.

## Architecture
| File | Job |
|---|---|
| `app.py` | Streamlit UI only |
| `agent.py` | LangGraph state, nodes, routing, final report |
| `tools.py` | Safe helpers (read files, diffs, apply changes, run tests) |
| `prompts.py` | Short prompts that ask for JSON and honest answers |
| `config.py` | Reads `.env`, checks settings, hides secrets |
| `workspace/sample-project/` | Demo project with one deliberate bug |
| `workspace/.runs/` | Temporary working copies (created automatically) |
| `tests/` | The app's own tests (fake LLM, no paid API) |

## Workflow
```
START -> analyze -> make_plan -+-> (Plan Only) ------------------------> write_report -> END
                               +-> implement -> validate -+-> tests pass --------------> write_report
                                                          +-> fail, retry left -> repair -> validate
                                                          +-> fail, no retry left --------> write_report
```
Any error in any step jumps straight to `write_report`, so the run always ends with a report.

1. **analyze** - checks the task and repo, skips secrets/caches/binaries/big files, reads a limited number of relevant files, creates the working copy.
2. **make_plan** - model writes a plan: understanding, evidence (facts), suspected cause (hypothesis), files, steps, tests, risks.
3. **implement** - model returns complete new content for changed files. Paths are validated and diffs are built. Nothing is written yet.
4. **validate** - re-checks paths, writes the files into the working copy, runs `python -m pytest` (from an approved list) with a timeout, records exit code and output.
5. **repair** - if tests failed and the retry budget allows, the model gets the real failure output plus the current code and proposes a correction. Then `validate` runs again.
6. **write_report** - builds the report from recorded facts. The model only writes the short "Summary of changes" paragraph, and it is told the real status.

## Why LangGraph?
The task is not a straight line: it branches (Plan Only or Implement), loops (repair), and must stop safely.
LangGraph lets us write this as explicit nodes and conditional edges, so the flow is easy to see, test and explain.

## Why LangChain?
LangChain gives one common chat-model interface (`ChatOpenAI`, `ChatAnthropic`) and message classes.
We switch provider with an environment variable and no code change.

## Tools available to the agent
`inspect_repository`, `list_relevant_files`, `read_text_file`, `validate_relative_path`,
`propose_file_changes`, `apply_proposed_changes`, `generate_unified_diff`, `run_approved_tests`.
The model does not call these directly. The graph nodes call them. The model only returns text/JSON.
It has no shell, no Python REPL, no network tool.

## Conditional routing and repair loop
Routing functions in `agent.py` (`route_after_validate`, etc.) look at the state:
exit code 0 -> report. Non-zero and `repair_attempts < max_repair_attempts` -> repair. Otherwise -> report.
`repair` always adds 1 to `repair_attempts`, so the loop cannot run forever.

## File isolation
Each run copies permitted files to `workspace/.runs/<timestamp>-<id>/`. Only that copy is changed or tested.
The original project is never modified or deleted. The newest 10 copies are kept; older ones are deleted
(only inside `.runs/`). The "End review session" button deletes the current copy.
To use a result: download the `.patch` and run `git apply devpilot-changes.patch` inside your project, or copy the edits by hand.

## Limitations and safety notes
- **Tests are real code execution.** `pytest` runs the working copy's code in a normal subprocess. It has a timeout and a stripped environment (API keys are not passed), but it is **not a sandbox** (not Docker). Use trusted projects only.
- The model can make mistakes. Always read the diff. If a test file was changed, the report warns you.
- Only text files of common types are read or edited. Max 5 changed files per proposal, size limits apply.
- Test discovery uses pytest files only (`test_*.py`, `*_test.py`). No tests found = status UNVERIFIED.
- Streamlit stops a running script if you click or edit widgets during a run. Wait for it to finish.
- Not production software: no login, database, or deployment.

## Setup on Windows (PowerShell)
Use Python 3.10 or newer (3.12 was tested).
```powershell
cd devpilot-lite
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```
If PowerShell blocks the activate script: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate again.
If `py -3.12` is not found, use `py -3` or `python`.

## Environment configuration (`.env`)
```
LLM_PROVIDER=openai          # or: anthropic
LLM_MODEL=<model name from your provider's docs>
LLM_API_KEY=<your key>
LLM_BASE_URL=                # optional, only for OpenAI-compatible servers
```
Other values (`MAX_REPAIR_ATTEMPTS`, `EXECUTION_TIMEOUT_SECONDS`, `MAX_FILES_TO_INSPECT`, `MAX_FILE_BYTES`) have safe defaults.
LangSmith: set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...` (optional).
The app opens even without a key and shows setup help.

## Launch
```powershell
streamlit run app.py
```

## Run the tests
```powershell
python -m pytest                 # the app's own tests (fake LLM, no API key needed)
cd workspace\sample-project
python -m pytest                 # the sample project: 1 FAILED + 4 passed on purpose
cd ..\..
```

## The sample project's bug
In `workspace/sample-project/calculator.py`, `subtract(a, b)` returns `b - a` instead of `a - b`.
`test_subtract` fails because of it. The dashboard does not contain this answer. The agent must find it from the code and test output.
The repair step only runs if the first fix does not make the tests pass. The unit tests in `tests/` prove this path with a fake model.

## How to demo in a campus interview
1. Start the app. Show the sidebar (model status, key hidden).
2. Run the sample's tests in a terminal to show 1 failure.
3. In the dashboard choose `sample-project`, **Plan Only**, task: "Investigate why test_subtract fails." Show the plan: **evidence** vs **suspected cause**.
4. Run again with **Implement & Test**. Show the activity log, the diff, the real pytest output and the PASSED status.
5. Show `workspace/sample-project/calculator.py` is still unchanged (the agent edited only `.runs/...`).
6. Open `agent.py`: show the conditional edges and `route_after_validate`. Say how the retry limit works.
7. Run `python -m pytest` to show the safety tests (traversal, secrets, loop ends, no fake pass).
