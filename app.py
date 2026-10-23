"""app.py - Streamlit dashboard for DevPilot Lite. All agent logic lives in agent.py."""
import re
from pathlib import Path

import streamlit as st

import agent
import tools
from config import load_config, redact, setup_tracing

st.set_page_config(page_title="DevPilot Lite", page_icon="🧭", layout="wide")
st.markdown("""<style>
.block-container{padding-top:2rem;max-width:1200px}
.pill{display:inline-block;padding:2px 12px;margin:0 6px 6px 0;border-radius:999px;
      border:1px solid #30363d;color:#8b949e;font-size:.85rem}
.pill.done{border-color:#2ea043;color:#3fb950}
.pill.now{border-color:#4F8BF9;color:#4F8BF9;font-weight:600}
</style>""", unsafe_allow_html=True)

STAGES = ["Analyze", "Plan", "Implement", "Validate", "Repair", "Report"]
cfg = load_config()
setup_tracing(cfg)

for key, default in (("result", None), ("running", False), ("start_requested", False), ("job", {})):
    st.session_state.setdefault(key, default)


# ------------------------------------------------------------------ callbacks
def request_start():
    """Runs before the page re-renders. Ignores clicks while a run is active."""
    if st.session_state.running:
        return
    st.session_state.job = {
        "repo": st.session_state.get("repo_select"),
        "task": st.session_state.get("task_text", ""),
        "mode": st.session_state.get("mode_select", agent.MODE_IMPLEMENT),
    }
    st.session_state.running = True
    st.session_state.start_requested = True


def discard_copy():
    wc = (st.session_state.result or {}).get("working_copy")
    if wc:
        try:
            tools.discard_working_copy(cfg.runs_dir, wc)
            st.session_state.result["working_copy"] = ""
            st.session_state.result["discarded"] = True
        except tools.PathError as exc:
            st.session_state.result["error"] = str(exc)


# ------------------------------------------------------------------ helpers
def stage_pills(result: dict) -> None:
    log = result.get("activity_log", [])
    visited = {m.group(1) for line in log if (m := re.search(r"\[(\w+)\]", line))}
    current = result.get("current_stage", "")
    html = ""
    for stage in STAGES:
        css = "pill"
        if stage in visited:
            css += " done"
        if stage == current or (stage == "Report" and current == "Complete"):
            css += " now"
        html += f'<span class="{css}">{stage}</span>'
    st.markdown(html, unsafe_allow_html=True)


def run_job(job: dict) -> None:
    """Run the graph and show REAL events as each node finishes."""
    last = {}
    try:
        if not cfg.ready:
            raise ValueError("Model is not configured. " + " ".join(cfg.problems()))
        tools.cleanup_old_runs(cfg.runs_dir, keep=10)
        graph = agent.build_graph(agent.build_llm(cfg), cfg)
        state = agent.initial_state(job["task"], job["repo"], job["mode"], cfg)
        with st.status("Agent running...", expanded=True) as status:
            stage_box, log_box = st.empty(), st.empty()
            for last in graph.stream(state, stream_mode="values"):
                stage_box.markdown(f"**Current stage:** {last.get('current_stage', '')}")
                log_box.code("\n".join(last.get("activity_log", [])), language=None)
            status.update(label="Agent finished", state="complete")
        st.session_state.result = last
    except Exception as exc:  # noqa: BLE001
        st.session_state.result = {
            **last, "error": redact(f"{type(exc).__name__}: {exc}", cfg)[:1500],
            "final_status": "ERROR", "current_stage": last.get("current_stage", "Starting"),
        }
    finally:
        st.session_state.running = False


def render_plan(plan: dict) -> None:
    if not plan:
        st.info("No plan was produced.")
        return
    st.markdown("**Problem understanding**\n\n" + (plan.get("problem_understanding") or "-"))
    st.markdown("**Evidence found in the repository** (facts seen in files)")
    st.markdown("\n".join(f"- {e}" for e in plan.get("evidence", [])) or "- none")
    st.markdown("**Suspected cause** (hypothesis, not proven)\n\n" + (plan.get("suspected_cause") or "none"))
    for title, key in (("Files to modify", "files_to_modify"), ("Steps", "steps"),
                       ("Tests to add or run", "tests"), ("Potential risks", "risks")):
        st.markdown(f"**{title}**")
        st.markdown("\n".join(f"- {i}" for i in plan.get(key, [])) or "- none")


# ------------------------------------------------------------------ header + sidebar
st.title("DevPilot Lite")
st.caption("Autonomous Software Engineering Agent")

with st.sidebar:
    st.subheader("Model configuration")
    if cfg.ready:
        st.success("Model configured")
        st.write(f"Provider: `{cfg.provider}`")
        st.write(f"Model: `{cfg.model}`")
        st.write("API key: set (hidden)")
        if cfg.base_url:
            st.write("Custom base URL: set")
    else:
        st.warning("Model not configured")
        for problem in cfg.problems():
            st.write("- " + problem)
    st.write("LangSmith tracing: " + ("on" if cfg.tracing_enabled else "off"))
    st.write(f"Repair attempts: {cfg.max_repair_attempts} | Test timeout: {cfg.timeout_seconds}s")

if not cfg.ready:
    st.warning("The app is running, but the LLM is not configured yet, so the agent cannot start.")
    with st.expander("Setup instructions", expanded=True):
        st.markdown(
            "1. Copy `.env.example` to `.env` (PowerShell: `Copy-Item .env.example .env`).\n"
            "2. Fill `LLM_PROVIDER` (`openai` or `anthropic`), `LLM_MODEL` and `LLM_API_KEY`.\n"
            "3. Stop the app (Ctrl+C) and run `streamlit run app.py` again."
        )

# ------------------------------------------------------------------ controls
repos = tools.list_workspace_repos(cfg.workspace_dir)
busy = st.session_state.running
left, right = st.columns([1, 2])
with left:
    with st.expander("Upload your project (.zip)"):
        up_n = st.session_state.setdefault("upload_n", 0)
        up = st.file_uploader("Zip file of your project", type=["zip"], disabled=busy, key=f"zip_up_{up_n}")
        if up is not None and st.button("Add project", disabled=busy):
            try:
                new_name = tools.save_uploaded_zip(cfg.workspace_dir, up.getvalue(), up.name)
                st.session_state.repo_select = new_name
                st.session_state.upload_n = up_n + 1  # clears the uploader
                st.rerun()
            except tools.PathError as e:
                st.error(str(e))
    st.selectbox("Repository (inside workspace/)", repos, key="repo_select", disabled=busy)
    st.selectbox("Execution mode", [agent.MODE_PLAN_ONLY, agent.MODE_IMPLEMENT], index=1, key="mode_select", disabled=busy)
    st.text_input("Maximum repair attempts", value=str(cfg.max_repair_attempts), disabled=True,
                  help="Fixed for this version. Set MAX_REPAIR_ATTEMPTS in .env to change it.")
with right:
    st.text_area("Coding task", key="task_text", height=170, disabled=busy,
                 placeholder="Describe what the agent should do, for example: investigate the failing tests and fix the cause.")
if not repos:
    st.error("No repositories found. Upload a project zip using the box above.")
st.button("Start Agent", type="primary", on_click=request_start,
          disabled=busy or not cfg.ready or not repos)

# ------------------------------------------------------------------ run (if requested)
if st.session_state.start_requested:
    st.session_state.start_requested = False
    run_job(st.session_state.job)
    st.rerun()  # redraw with the button enabled again and the saved result

# ------------------------------------------------------------------ results
result = st.session_state.result
if result:
    st.divider()
    status = result.get("final_status", "")
    stage_pills(result)
    c1, c2, c3 = st.columns(3)
    c1.metric("Workflow stage", result.get("current_stage", "-"))
    c2.metric("Final status", status or "-")
    c3.metric("Repair attempts used", f"{result.get('repair_attempts', 0)} / {result.get('max_repair_attempts', 0)}")

    if result.get("error"):
        st.error(result["error"])
    if status == "PASSED":
        st.success("Tests passed (exit code 0).")
    elif status == "FAILED":
        st.error(f"Tests failed (exit code {result.get('test_exit_code')}). The task is NOT complete.")
    elif status == "UNVERIFIED":
        st.warning("No tests were run, so the changes are not verified.")
    elif status == "PLAN_ONLY":
        st.info("Plan Only mode: no files were changed and no code was run.")

    tabs = st.tabs(["Plan", "Files inspected", "Proposed changes & diff", "Test output", "Final report", "Activity log"])
    with tabs[0]:
        render_plan(result.get("plan", {}))
    with tabs[1]:
        files = result.get("relevant_files", [])
        st.markdown("\n".join(f"- `{f}`" for f in files) if files else "No files were inspected.")
        with st.expander("Excluded from inspection (secrets, caches, large or binary files)"):
            st.markdown("\n".join(f"- `{e}`" for e in result.get("excluded_files", [])) or "Nothing excluded.")
    with tabs[2]:
        changed = [p for p in result.get("proposed_changes", []) if p.get("diff")]
        if not changed:
            st.info("No changes were proposed.")
        for p in changed:
            added = sum(1 for l in p["diff"].splitlines() if l.startswith("+") and not l.startswith("+++"))
            removed = sum(1 for l in p["diff"].splitlines() if l.startswith("-") and not l.startswith("---"))
            with st.expander(f"{p['path']}  ({'new' if p['is_new'] else 'modified'}, +{added} / -{removed})", expanded=True):
                st.code(p["diff"], language="diff")
        if changed:
            st.download_button("Download proposed changes (.patch)", tools.build_patch(changed),
                               file_name="devpilot-changes.patch", mime="text/x-diff")
            st.caption("Changes exist only in a temporary working copy. To use them, run `git apply devpilot-changes.patch` "
                       "inside your project folder, or copy the edits by hand. The original is never modified.")
    with tabs[3]:
        if result.get("tests_ran"):
            st.write(f"Command: `{result.get('test_command')}`")
            st.write(f"Exit code: **{result.get('test_exit_code')}**")
            st.write(result.get("validation_summary", ""))
            for h in result.get("test_history", []):
                st.write(f"- Run {h['run']}: exit code {h['exit_code']}{' (timed out)' if h['timed_out'] else ''} in {h['elapsed']}s")
            st.code(result.get("test_output", ""), language=None)
        else:
            st.info(result.get("validation_summary") or "No tests were run in this session.")
    with tabs[4]:
        st.markdown(result.get("final_report") or "No final report (the run stopped early). See the error above.")
    with tabs[5]:
        st.code("\n".join(result.get("activity_log", [])) or "(empty)", language=None)

    if result.get("working_copy"):
        st.caption(f"Working copy: `.runs/{Path(result['working_copy']).name}` (older copies are cleaned automatically; keeps the newest 10).")
        st.button("End review session and delete this working copy", on_click=discard_copy)
    elif result.get("discarded"):
        st.caption("Working copy deleted. The original project was never changed.")