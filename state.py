"""
Shared LangGraph state for the Guest Booking Cancellation Agent.

REWRITTEN for the LLM-tool-calling architecture (previously a 38-field
hand-written state machine for a deterministic router - see
state.py.pre_rewrite_backup for the old version). The LLM now owns all
conversation logic, so state only needs to carry: tenant identity, the
cached per-tenant config/system-prompt (computed once per thread, not
re-derived every turn), and the chat history itself.

IMPORTANT: `messages` uses LangGraph's `add_messages` reducer. Each graph
invocation only needs to supply the NEW message(s) (e.g. one HumanMessage
per turn) - the checkpointer (MemorySaver, unchanged) automatically
appends to and persists the full history per thread_id. This replaces
ALL of the old manual retry-counter/interrupt-payload bookkeeping: the
LLM's own latest message simply either contains tool_calls (loop
continues) or doesn't (turn ends, that message IS the reply to the user).
"""

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):

    # ==========================================================
    # Identity / tenancy
    # ==========================================================

    client_id: str
    session_id: str
    channel_phone: Optional[str]  # verified channel identity (e.g. WhatsApp sender), used by compare_phone

    # ==========================================================
    # Cached per-tenant config (loaded once per thread by graph.py's
    # load_config node, not re-derived every turn)
    # ==========================================================

    # Merged client_config.csv + dialect_templates.csv row (renamed from
    # the old project's "messages" field, which clashed with the
    # conventional chat-history key name this architecture needs).
    templates: dict

    # The system prompt built from `templates` (prompts.build_system_prompt) -
    # cached so config.get_messages()/CSV lookups only happen once per
    # thread rather than on every single turn.
    system_prompt: Optional[str]

    # ==========================================================
    # Chat history (the entire conversation - replaces every one of the
    # old state machine's step-specific fields)
    # ==========================================================

    messages: Annotated[list, add_messages]

    # True once the exact opening greeting has been deterministically
    # prepended for this thread (see graph.py's agent() node). This
    # exists because relying on the LLM to reproduce the clinic's
    # greeting text verbatim, every single time, turned out to be
    # unreliable in practice (observed directly: the same clinic's
    # greeting came out differently worded/structured across separate
    # conversations despite explicit prompt instructions to reuse it
    # exactly). Guaranteeing it in code removes that source of
    # inconsistency entirely, without touching how the LLM handles
    # anything else in the conversation.
    greeted: bool
