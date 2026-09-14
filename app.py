"""
PHASE 5: Streamlit chat interface.
PHASE 6: persistent thread selection.
PHASE 7: "+ New Chat".
PHASE 8: conversation titles.
PHASE 10: this is what you manually run TEST 1 / TEST 2 / TEST 3 against.

CRITICAL DESIGN RULE (per spec): st.session_state is used ONLY to remember
which thread_id is currently selected in THIS browser tab, for THIS UI
session. It is never the source of truth for conversation content. Every
render of the message list reads directly from LangGraph's checkpointed
state via graph.get_messages(app, thread_id) -- so even if session_state
were wiped (hard refresh, new browser tab), selecting the same thread_id
again from the sidebar reloads the exact same conversation from SQLite.
"""
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

import graph
import thread_store

st.set_page_config(page_title="PolicyGPT", layout="wide")

# ---------------------------------------------------------------------
# Build the graph once per Streamlit server process (cached across reruns
# via st.session_state guard + module-level singleton in graph.py itself).
# ---------------------------------------------------------------------
app = graph.build_app()

# ---------------------------------------------------------------------
# session_state holds ONLY the UI selection (which thread is open), never
# the messages themselves.
# ---------------------------------------------------------------------
if "current_thread_id" not in st.session_state:
    st.session_state.current_thread_id = None

# On a hard refresh, session_state.current_thread_id resets to None even
# though the conversations still exist in SQLite. Fall back to the most
# recently created persisted thread so the user isn't dropped into a blank
# state with their history invisible (they can still switch threads from
# the sidebar either way).
existing_threads = thread_store.list_threads()
if st.session_state.current_thread_id is None and existing_threads:
    st.session_state.current_thread_id = existing_threads[0]["thread_id"]

# ---------------------------------------------------------------------
# Sidebar: persistent thread list, read fresh from SQLite every rerun.
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("PolicyGPT")

    if st.button("+ New Chat", use_container_width=True):
        # PHASE 7: a new thread_id is generated ONLY here, on an explicit
        # user click -- never as a side effect of a plain Streamlit rerun.
        new_id = thread_store.create_thread(title="New Chat")
        st.session_state.current_thread_id = new_id
        st.rerun()

    st.markdown("---")
    st.caption("Conversations")

    threads = thread_store.list_threads()
    if not threads:
        st.caption("No conversations yet — click **+ New Chat** to start.")
    for t in threads:
        label = t["title"] or "New Chat"
        is_active = t["thread_id"] == st.session_state.current_thread_id
        if st.button(("➤ " if is_active else "") + label,
                     key=f"thread-{t['thread_id']}", use_container_width=True):
            st.session_state.current_thread_id = t["thread_id"]
            st.rerun()

    st.markdown("---")
    with st.expander("Architecture notes"):
        st.markdown(
            "- **session_state**: only remembers which thread is open in this tab\n"
            "- **LangGraph state**: the actual message history (`messages` field)\n"
            "- **SQLite checkpoint**: where that state is durably stored, "
            "in `data/checkpoints.sqlite`\n"
            "- **thread_id**: stable key linking every invoke to the same conversation\n"
            "- **PolicyGPT**: from-scratch decoder-only transformer — "
            "not retrained per message, only used for inference"
        )

# ---------------------------------------------------------------------
# Main pane
# ---------------------------------------------------------------------
thread_id = st.session_state.current_thread_id

if thread_id is None:
    st.info("Click **+ New Chat** in the sidebar to start your first conversation.")
    st.stop()

# Always read messages fresh from the LangGraph/SQLite checkpoint -- NEVER
# from a locally-held list. This is what makes refresh-safe behavior work:
# this line is the entire "restore history" mechanism, and it runs on
# every single rerun regardless of why the rerun happened.
messages = graph.get_messages(app, thread_id)

for msg in messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.write(msg.content)

user_input = st.chat_input("Ask PolicyGPT about fiscal, monetary, or education policy...")

if user_input:
    is_first_message = len(messages) == 0
    with st.chat_message("user"):
        st.write(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = graph.send_message(app, thread_id, user_input)
        st.write(reply.content)

    if is_first_message:
        # PHASE 8: auto-title the thread from the first thing the user asked
        thread_store.set_title(thread_id, thread_store.title_from_first_message(user_input))

    st.rerun()
