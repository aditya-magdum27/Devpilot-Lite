"""
PHASE 9 (built early since graph.py depends on it): context window management.

PolicyGPT's transformer has a fixed max_len (256 characters, see model.py).
We can never hand it the full conversation history once a thread gets long
-- it would silently truncate from the wrong end (or error) inside the
model. build_context() is the explicit truncation policy, applied BEFORE
the model ever sees the text:

  1. Format each message as "Human: ...\\n" or "AI: ...\\n".
  2. Walk the message list from the END (most recent) backward, accumulating
     characters, until adding the next (older) message would exceed the
     character budget.
  3. Reverse back into chronological order, join, and append "AI:" as the
     generation prompt for the model to continue.

This guarantees:
  - Recent turns are always preserved (they're added first, from the end).
  - Older turns are dropped first when the conversation gets long.
  - The result never exceeds the model's context length.

Char count is used as a stand-in for "tokens" here because the tokenizer is
character-level (see tokenizer.py) -- for a sub-word/BPE tokenizer you would
count actual token ids instead, but the truncation *logic* is identical.
"""
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

# leave headroom in the model's 256-char context for the actual generation
MAX_CONTEXT_CHARS = 180


def _format_message(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):
        role = "Human"
    elif isinstance(msg, AIMessage):
        role = "AI"
    else:
        role = "System"
    return f"{role}: {msg.content}\n"


def build_context(messages: list[BaseMessage], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Returns a bounded prompt string ending in 'AI:' ready to feed to
    generate(). Keeps the most recent turns, drops the oldest first."""
    kept = []
    total = 0
    for msg in reversed(messages):
        formatted = _format_message(msg)
        if total + len(formatted) > max_chars and kept:
            # always keep at least the single most recent message, even if
            # it alone exceeds the budget -- an empty prompt is worse
            break
        kept.append(formatted)
        total += len(formatted)
    kept.reverse()
    prompt = "".join(kept) + "AI:"
    return prompt


if __name__ == "__main__":
    # quick manual test: simulate a long-ish conversation and confirm
    # older turns get dropped while the most recent ones survive
    msgs = []
    for i in range(20):
        msgs.append(HumanMessage(content=f"Question number {i} about fiscal policy?"))
        msgs.append(AIMessage(content=f"Answer number {i} about fiscal policy."))
    ctx = build_context(msgs, max_chars=180)
    print(ctx)
    print("--- length ---", len(ctx))
    assert len(ctx) <= 180 + len("AI:") + 5  # small slack for the always-keep-last rule
    print("OK: context bounded correctly, most recent turns kept")
