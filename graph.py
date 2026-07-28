"""
LangGraph graph for the LLM-tool-calling Guest Booking Cancellation Agent.

REWRITTEN (see graph.py.pre_rewrite_backup for the old deterministic
20-node router). The graph is now a standard two-node ReAct-style loop:

    START -> load_config -> agent <-> tools -> END

  - load_config: loads/caches the tenant's client_config.csv +
    dialect_templates.csv (config.get_messages, UNCHANGED function) and
    builds the system prompt (prompts.build_system_prompt) - but only
    once per thread (checked via state.get("templates")), not every turn.
  - agent: calls the LLM, bound to every tool in tools.ALL_TOOLS. The LLM
    decides which tool(s) to call, if any.
  - tools: executes whatever tool call(s) the LLM just requested
    (LangGraph's prebuilt ToolNode - handles InjectedState wiring
    automatically) and appends the results as ToolMessages.
  - Routing: after "agent", if the LLM's latest message contains
    tool_calls, go to "tools"; otherwise the LLM's message IS the reply
    to the user this turn, so the graph ends (control returns to
    whichever caller - main.py's CLI or app.py's HTTP layer - the same
    way for both).

NOTE ON INTERRUPTS: the old graph used explicit interrupt() calls at
wait_for_otp/wait_for_selection/wait_for_confirmation/etc. Those are GONE
- there is no longer a fixed set of "steps that pause". Any time the
agent's response has no tool_calls (e.g. it's asking "which one?" or
"please confirm" or "what's the OTP?"), that's a natural, implicit pause:
the graph reaches END, and the next incoming HumanMessage (via app.py or
main.py) simply gets appended and the graph invoked again - the
checkpointer (MemorySaver, PRESERVED exactly per requirements) restores
the full chat history for that thread_id automatically. This IS how
interrupt/resume is achieved now - it's just no longer a graph-level
primitive, because the "steps" themselves no longer exist as distinct
nodes; the LLM decides turn-by-turn whether it needs another tool call
or needs to ask the user something.
"""

import logging
import re
from typing import Optional

from langchain_core.messages import AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

import config
import tools
from prompts import build_system_prompt
from state import AgentState

logger = logging.getLogger(__name__)


# ==========================================================
# LLM (bound to every tool) - api.py/config.py's OPENAI_* settings,
# UNCHANGED from the old project
# ==========================================================

_llm = ChatOpenAI(
    model=config.OPENAI_MODEL,
    api_key=config.OPENAI_API_KEY or "sk-not-configured",
    timeout=config.OPENAI_TIMEOUT_SECONDS,
)

_llm_with_tools = _llm.bind_tools(tools.ALL_TOOLS)


# ==========================================================
# Nodes
# ==========================================================

def load_config(state: AgentState) -> AgentState:
    """Loads client_config.csv/dialect_templates.csv (config.get_messages,
    UNCHANGED) and builds the system prompt EVERY turn.

    CHANGED: this used to skip rebuilding once state["templates"] was
    already set for a thread (a caching optimization). The real-world
    cost of that: any update to prompts.py or the CSVs would NEVER take
    effect for a conversation already in progress - only brand new
    threads picked it up - which is exactly what caused a live
    conversation to keep using stale, pre-fix wording long after
    multiple prompt improvements had already been deployed. Rebuilding
    every turn is cheap (in-memory CSV lookups, already cached at the
    row level by config.py's lru_cache, plus plain string formatting -
    no network calls), so there's no real performance reason to keep
    the old per-thread caching."""

    templates = config.get_messages(state["client_id"])

    state["templates"] = templates
    state["system_prompt"] = build_system_prompt(templates)

    return state


_MORNING_CUES = ("صباح", "good morning", "morning")
_EVENING_CUES = ("مساء", "good evening", "evening")


_ENGLISH_GREETING_TEMPLATE = (
    "{salutation}\n"
    "I'm {agent_name}, the virtual assistant at {clinic_name}, and I'm happy to help you today.\n"
    "I can help you with:\n"
    "\U0001F5D3\uFE0F Booking a new appointment\n"
    "\u270F\uFE0F Modifying or cancelling an existing appointment\n"
    "\U0001FA7A Medical guidance to choose the right specialty or doctor\n"
    "\u2139\uFE0F Questions about the hospital's services and doctors\n"
    "\U0001F464 Speaking with a customer service representative\n\n"
    "How can I help you today? \U0001F60A"
)


def _build_greeting(templates: dict, user_message: str, target_language: str) -> str:
    """
    Build the deterministic opening greeting for this conversation.

    For Arabic (or undetermined) conversations, this is the clinic's own
    CSV-authored `msg_unknown_fallback` text, verbatim, with only its
    fixed opening line optionally swapped for a time-of-day salutation
    (see _personalized_greeting).

    For English conversations, there is no English column in the CSV to
    reuse, so a fixed English template - built here, following the exact
    same structure (persona intro, service list, closing question) as
    the Arabic one - is used instead, so English speakers ALSO get a
    consistent, deterministic greeting rather than one freely composed
    by the LLM each time.
    """

    if target_language == "en":
        lowered = (user_message or "").lower()
        if any(cue in lowered for cue in _MORNING_CUES):
            salutation = "Good morning! \U0001F60A"
        elif any(cue in lowered for cue in _EVENING_CUES):
            salutation = "Good evening! \U0001F60A"
        else:
            salutation = "Hi there! \U0001F44B"

        return _ENGLISH_GREETING_TEMPLATE.format(
            salutation=salutation,
            agent_name=templates.get("_agent_name") or "the assistant",
            clinic_name=templates.get("_clinic_name") or "the clinic",
        )

    greeting = templates.get("msg_unknown_fallback")
    if not greeting:
        return ""

    return _personalized_greeting(greeting, user_message, target_language)


def _personalized_greeting(greeting: str, user_message: str, language_hint: str) -> str:
    """
    Swap ONLY the official greeting's fixed opening line ("أهلاً بيك 👋" /
    "أهلاً وسهلاً بك 👋") for a time-of-day salutation ("صباح النور!"/
    "مساء النور!"/"Good morning!"/"Good evening!") when the user's own
    first message clearly signals one - keeping the entire rest of the
    template (persona intro, service list, closing question) exactly as
    authored. Falls back to the original fixed opening line unchanged
    when the user's message doesn't give a clear time-of-day cue (e.g. a
    booking reference, a plain "hi", or anything else neutral).
    """

    lowered = (user_message or "").lower()

    if any(cue in lowered for cue in _MORNING_CUES):
        salutation = "صباح النور! 😊" if _looks_arabic(user_message) else "Good morning! 😊"
    elif any(cue in lowered for cue in _EVENING_CUES):
        salutation = "مساء النور! 😊" if _looks_arabic(user_message) else "Good evening! 😊"
    else:
        return greeting  # no clear time-of-day cue - use the template's own opening line as-is

    lines = greeting.split("\n", 1)
    if len(lines) == 2:
        return f"{salutation}\n{lines[1]}"
    return salutation


def _looks_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def _has_latin_letters(text: str) -> bool:
    return bool(re.search(r"[A-Za-z]{2,}", text or ""))


def _detect_target_language(messages: list) -> Optional[str]:
    """
    Determine which language THIS reply must be in, deterministically -
    by code, not left to the LLM to infer from a long system prompt.

    Scans the conversation's HumanMessages, most recent first, and
    returns "ar"/"en" based on the first one that gives a clear signal
    (Arabic script, or Latin letters). A message with neither (e.g. just
    digits like an OTP code, or "yes"/"نعم") is skipped in favor of an
    earlier message that does give a signal - this is what keeps the
    established language consistent through dialect-neutral replies
    without resetting.

    Returns None only if NO message in the whole conversation gives any
    signal at all (extremely unlikely in practice) - in that case the
    system prompt's own default dialect applies unmodified.

    WHY THIS EXISTS: relying solely on the LANGUAGE & DIALECT prose rule
    inside the (long, Arabic-reference-heavy) system prompt measurably
    did not reliably keep the reply in the user's actual language,
    including on a conversation that was purely English from its very
    first message - a plain prose instruction competing with thousands
    of characters of Arabic reference material was not a strong enough
    signal on its own.
    """

    for msg in reversed(messages):
        if getattr(msg, "type", None) != "human":
            continue
        content = msg.content or ""
        if _looks_arabic(content):
            return "ar"
        if _has_latin_letters(content):
            return "en"

    return None


_LANGUAGE_DIRECTIVE = {
    "en": (
        "============================================================\n"
        "MANDATORY LANGUAGE FOR THIS REPLY: ENGLISH\n"
        "============================================================\n"
        "This entire reply must be written in English only. Do not use "
        "any Arabic words, letters, or Arabic-script emoji captions "
        "anywhere in it. Ignore the Arabic dialect/reference-phrase "
        "sections further below for this reply - they do not apply.\n\n"
    ),
    "ar": (
        "============================================================\n"
        "MANDATORY LANGUAGE FOR THIS REPLY: ARABIC\n"
        "============================================================\n"
        "This entire reply must be written in Arabic, following the "
        "dialect/tone and reference phrases further below. Do not use "
        "any English words anywhere in it.\n\n"
    ),
}


def agent(state: AgentState) -> dict:
    """Calls the LLM with the cached system prompt + full chat history.
    The LLM decides whether to call a tool or reply directly.

    GREETING GUARANTEE: if this call produces a final reply (no
    tool_calls, i.e. this turn is about to end) and the conversation
    hasn't been greeted yet, the clinic's exact opening greeting text is
    deterministically prepended in code - not left to the LLM to
    reproduce from the system prompt's reference phrases. This was
    added because relying on the LLM alone measurably did not keep the
    greeting's exact wording/structure consistent across separate
    conversations, despite explicit instructions to reuse it verbatim.
    The opening line specifically is swapped for a time-of-day salutation
    when the user's own first message signals one (see
    _personalized_greeting) - the rest of the template is untouched.

    DOUBLE-GREETING FIX: on the first turn, the LLM is ALSO told, via an
    extra instruction appended to the system message, not to write any
    greeting/opener of its own, AND not to jump ahead into asking about a
    reference/phone number before the user has actually said they want
    to cancel something - a bare greeting like "صباح الخير" states no
    intent yet, so the reply should be the greeting's own closing
    question only, waiting for the user's actual next message.

    DETERMINISTIC LANGUAGE DIRECTIVE: which language this reply must be
    in is computed by code (_detect_target_language) from the
    conversation's actual messages and placed at the very TOP of the
    system message - not left to the prose LANGUAGE & DIALECT rule
    buried inside the (long) system prompt, which measurably was not a
    strong enough signal on its own to keep replies in the user's actual
    language, even for a conversation that had been purely English from
    its first message."""

    target_language = _detect_target_language(state["messages"])
    language_directive = _LANGUAGE_DIRECTIVE.get(target_language, "")

    system_content = language_directive + state["system_prompt"]

    if not state.get("greeted"):
        system_content += (
            "\n\n============================================================\n"
            "FIRST-TURN OVERRIDE\n"
            "============================================================\n"
            "This is the first message of a new conversation. The opening "
            "greeting/persona introduction has ALREADY been (or will be) sent "
            "separately, outside of what you write here. Do NOT write any "
            "greeting, self-introduction, or generic opener of your own, in "
            "any language (no 'صباح النور'/'مساء النور'/'Hi there! How can I "
            "help?' or similar).\n\n"
            "IMPORTANT - do not jump ahead: if the user's message is just a "
            "greeting or small talk with no stated intent yet (e.g. "
            "'صباح الخير', 'hi', 'مرحبا', 'good morning', with nothing else), "
            "do NOT ask about a booking reference or phone number yet - you "
            "don't know they want to cancel anything yet. In that case, "
            "simply write NOTHING here (an empty reply is fine) and let the "
            "greeting's own closing question stand on its own, waiting for "
            "them to say what they need. Only start STEP 1 (asking to "
            "identify the booking) once the user's message actually "
            "indicates they want to cancel an appointment."
        )

    system_message = SystemMessage(content=system_content)
    response = _llm_with_tools.invoke([system_message] + state["messages"])

    updates: dict = {}

    has_tool_calls = bool(getattr(response, "tool_calls", None))

    if not has_tool_calls and not state.get("greeted"):
        first_user_message = state["messages"][0].content if state["messages"] else ""
        greeting = _build_greeting(state.get("templates") or {}, first_user_message, target_language or "ar")

        if greeting and greeting.strip() not in (response.content or ""):
            combined = f"{greeting.strip()}\n\n{response.content}".strip() if response.content else greeting.strip()
            response = AIMessage(content=combined)

        updates["greeted"] = True

    updates["messages"] = [response]

    return updates


def route_after_agent(state: AgentState) -> str:
    """If the LLM's latest message requested tool call(s), run them.
    Otherwise its message is the reply for this turn - end the graph."""

    last_message = state["messages"][-1]

    if getattr(last_message, "tool_calls", None):
        return "tools"

    return END


# ToolNode automatically injects graph state into any tool parameter
# annotated with InjectedState (see tools.py's `state` params) without
# exposing it to the LLM's function-calling schema.
_tool_node = ToolNode(tools.ALL_TOOLS)


# ==========================================================
# Build graph
# ==========================================================

builder = StateGraph(AgentState)

builder.add_node("load_config", load_config)
builder.add_node("agent", agent)
builder.add_node("tools", _tool_node)

builder.set_entry_point("load_config")
builder.add_edge("load_config", "agent")
builder.add_conditional_edges("agent", route_after_agent, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

checkpointer = MemorySaver()

graph = builder.compile(checkpointer=checkpointer)
