"""
PHASE 2: LangGraph StateGraph with a message-based state.
PHASE 3: SQLite checkpointer added.
PHASE 4: verify a thread persists after an application restart (see the
         __main__ block below -- run this file twice with THREAD_ID set
         to the same value and watch the message count grow).

State design, per the spec: NOT a global Python list, NOT session_state.
`messages` uses LangGraph's `add_messages` reducer, so returning just the
new AIMessage from the node appends it to whatever's already checkpointed
for that thread_id -- it does not overwrite the history.
"""
import sqlite3
from typing import TypedDict, Annotated

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from context import build_context
from model import generate

DB_PATH = "data/checkpoints.sqlite"


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def policygpt_node(state: ChatState) -> ChatState:
    """Reads the full (checkpoint-restored) message history, builds a
    bounded context string, calls the from-scratch transformer, and
    returns only the NEW AIMessage -- the add_messages reducer handles
    appending it to what's already in the checkpoint."""
    prompt = build_context(state["messages"])
    completion = generate(prompt, max_new_tokens=150, temperature=0.7)
    return {"messages": [AIMessage(content=completion.strip())]}


_CONN = None
_APP = None


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        # check_same_thread=False: Streamlit's execution model can hand
        # control across threads (reruns, widget callbacks), and SQLite's
        # default same-thread check would otherwise raise on a connection
        # created in one thread and used from another.
        _CONN = sqlite3.connect(db_path, check_same_thread=False)
    return _CONN


def build_app(db_path: str = DB_PATH):
    """Compiles the StateGraph with a SqliteSaver checkpointer. This is the
    single source of truth for conversation state -- NOT session_state,
    NOT an in-memory list."""
    global _APP
    if _APP is not None:
        return _APP
    conn = get_connection(db_path)
    checkpointer = SqliteSaver(conn)

    workflow = StateGraph(ChatState)
    workflow.add_node("policygpt", policygpt_node)
    workflow.set_entry_point("policygpt")
    workflow.add_edge("policygpt", END)

    _APP = workflow.compile(checkpointer=checkpointer)
    return _APP


def get_messages(app, thread_id: str) -> list[BaseMessage]:
    """Reads persisted messages for a thread WITHOUT invoking the model --
    used by Streamlit on every rerun to render history from SQLite,
    independent of session_state."""
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = app.get_state(config)
    if snapshot and snapshot.values:
        return snapshot.values.get("messages", [])
    return []


def send_message(app, thread_id: str, text: str) -> AIMessage:
    """Invokes the graph: loads checkpoint for thread_id, adds the human
    message, runs the node, saves the updated checkpoint."""
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke({"messages": [HumanMessage(content=text)]}, config=config)
    return result["messages"][-1]


if __name__ == "__main__":
    # PHASE 4 TEST: run this script, note the message count, then run it
    # again (a brand new `python graph.py` process) with the SAME
    # THREAD_ID -- the count should keep growing, proving persistence
    # across actual process restarts, not just within one run.
    import os
    thread_id = os.environ.get("THREAD_ID", "phase4-test-thread")
    app = build_app()

    existing = get_messages(app, thread_id)
    print(f"thread '{thread_id}' currently has {len(existing)} messages before this run")

    reply = send_message(app, thread_id, "What is fiscal policy?")
    print("AI reply:", reply.content[:200])

    after = get_messages(app, thread_id)
    print(f"thread '{thread_id}' now has {len(after)} messages after this run")
    print("Run this script again with the same THREAD_ID to confirm it keeps growing.")
