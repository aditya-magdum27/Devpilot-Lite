"""
PHASE 6: persistent thread selection (list threads from storage, not memory).
PHASE 7: "+ New Chat" creates a new thread_id.
PHASE 8: conversation titles.

LangGraph's checkpointer stores conversation STATE per thread_id, but it
has no concept of a human-readable "title" or a simple "list all my
conversations" query built for a sidebar UI. So we keep one extra, tiny
sqlite table -- `threads` -- in the SAME database file
(data/checkpoints.sqlite), separate from LangGraph's own checkpoint
tables. This table is ONLY metadata (thread_id, title, created_at) --
the actual conversation messages still live exclusively in LangGraph's
checkpoint tables, this file never duplicates that.
"""
import sqlite3
import uuid
from datetime import datetime, timezone

DB_PATH = "data/checkpoints.sqlite"


def _get_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS threads (
            thread_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def create_thread(title: str = "New Chat", db_path: str = DB_PATH) -> str:
    """PHASE 7: called only when the user clicks '+ New Chat' -- never on
    every Streamlit rerun. Returns a fresh, stable thread_id."""
    conn = _get_conn(db_path)
    thread_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO threads (thread_id, title, created_at) VALUES (?, ?, ?)",
        (thread_id, title, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    conn.close()
    return thread_id


def list_threads(db_path: str = DB_PATH) -> list[dict]:
    """PHASE 6: the sidebar reads this on every rerun -- it hits SQLite,
    not any in-memory Python list, so it survives a full app restart."""
    conn = _get_conn(db_path)
    rows = conn.execute(
        "SELECT thread_id, title, created_at FROM threads ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [{"thread_id": r[0], "title": r[1], "created_at": r[2]} for r in rows]


def set_title(thread_id: str, title: str, db_path: str = DB_PATH) -> None:
    """PHASE 8: called once, after the first human message on a thread, to
    replace the default 'New Chat' label with something derived from what
    the user actually asked."""
    conn = _get_conn(db_path)
    conn.execute("UPDATE threads SET title = ? WHERE thread_id = ?", (title, thread_id))
    conn.commit()
    conn.close()


def title_from_first_message(text: str, max_len: int = 40) -> str:
    text = text.strip().replace("\n", " ")
    return text[:max_len] + ("..." if len(text) > max_len else "")


if __name__ == "__main__":
    # quick manual test of the thread store in isolation
    import os
    test_db = "data/_thread_store_test.sqlite"
    if os.path.exists(test_db):
        os.remove(test_db)

    t1 = create_thread("New Chat", db_path=test_db)
    t2 = create_thread("New Chat", db_path=test_db)
    set_title(t1, title_from_first_message("What is fiscal policy?"), db_path=test_db)
    set_title(t2, title_from_first_message("What is education policy?"), db_path=test_db)

    threads = list_threads(db_path=test_db)
    for t in threads:
        print(t)
    assert len(threads) == 2
    print("OK: thread store create/list/title all working")
    os.remove(test_db)
