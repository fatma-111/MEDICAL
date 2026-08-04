"""
LangChain tools for the LLM-tool-calling Guest Booking Cancellation Agent.

REWRITTEN (see tools.py.pre_rewrite_backup for the old version). Every
tool now returns STRUCTURED DATA ONLY - no formatted sentences, no
message-template lookups, no natural language of any kind. The LLM
(driven by prompts.AGENT_SYSTEM_PROMPT_TEMPLATE) is solely responsible
for turning these status codes/data into user-facing replies. This is
the literal architecture change requested: tools never speak to the
user.

What did NOT change: api.py (all raw HTTP calls), config.py (client
config / base_url resolution), the timezone conversion math, the
active-booking filter, and the OTP dummy-provider mechanics. Those are
"Company APIs" / "booking logic" / "OTP logic" and were explicitly
required to stay untouched - only the OUTPUT SHAPE of the functions that
wrap them changed, from "already-formatted text" to "plain status/data".

Removed entirely (superseded by the LLM's own reasoning, since "the LLM
should decide" replaces every heuristic classifier):
  detect_message, extract_input_details, resolve_selection,
  parse_confirmation, detect_step_back, format_message,
  format_booking_card, format_booking_list, format_time_12h, format_date,
  find_matching_appointment (replaced by check_booking_status's ref-based
  re-lookup, simpler and equally safe since ref numbers are unique).
"""

import logging
import re
import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Annotated, Dict, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

import api
import rag
from config import (
    DEFAULT_TIMEZONE,
    CANCELLABLE_STATUS_CODES,
    CANCELLED_STATUS_NAME,
    DEFAULT_COUNTRY_CODE,
    DOCTOR_AVAILABILITY_WINDOW_DAYS,
    OTP_PROVIDER,
    OTP_TTL_SECONDS,
    TEST_OTP,
)
from state import AgentState

logger = logging.getLogger(__name__)


# ==========================================================
# Pure data helpers (unchanged in spirit from the old tools.py - these
# are data transforms, not user-facing text, so they stay)
# ==========================================================

def normalize_phone_number(phone: Optional[str]) -> Optional[str]:
    """Normalize a phone number to E.164 (e.g. "+201001255864")."""

    if not phone:
        return phone

    cleaned = re.sub(r"[\s\-().]", "", phone.strip())

    if cleaned.startswith("+"):
        return cleaned
    if cleaned.startswith("00"):
        return "+" + cleaned[2:]
    if cleaned.startswith(DEFAULT_COUNTRY_CODE):
        return "+" + cleaned
    if cleaned.startswith("0"):
        return "+" + DEFAULT_COUNTRY_CODE + cleaned[1:]

    return "+" + DEFAULT_COUNTRY_CODE + cleaned


def _is_valid_phone_format(phone: Optional[str]) -> bool:
    if not phone:
        return False
    return bool(re.match(r"^\+\d{7,15}$", phone.strip()))


def to_riyadh(utc_string: Optional[str], timezone_name: str = DEFAULT_TIMEZONE) -> Optional[str]:
    """ISO string -> the CLIENT'S OWN local time zone, as an ISO string.

    Despite the historical name (kept to minimize churn - this function
    used to be Riyadh-only), `timezone_name` is now a real per-client
    IANA zone name (e.g. "Africa/Cairo", "Asia/Riyadh" - both are real
    values already present in client_config.csv's own "timezone" column,
    exposed as state["templates"]["_timezone"]). This replaces a single
    hardcoded "+3 hours" that used to be applied to every clinic
    regardless of its actual location, which would have silently
    produced wrong times for any clinic outside Saudi Arabia, and
    doesn't account for DST where applicable.

    CRITICAL FIX (kept from the previous version): this used to blindly
    append a literal offset string to whatever `.isoformat()` produced,
    regardless of whether the parsed datetime was already timezone-aware.
    If the input was ALREADY timezone-aware (e.g.
    "2026-08-06T16:00:00+00:00" - confirmed directly from the real
    Doctors/GetDoctorScheduleSlots API response), that produced a
    doubled, invalid offset like "2026-08-06T19:00:00+00:00+03:00" -
    which caused a real production 400 error from GuestBookings/Update
    (it received an unparseable timestamp). Now: if the input is
    timezone-aware, convert it via astimezone(); if naive, assume UTC
    and attach the target zone directly on the datetime object - never
    by string concatenation."""

    if not utc_string:
        return None

    try:
        target_tz = ZoneInfo(timezone_name)
    except Exception:
        logger.warning("to_riyadh: unknown timezone %r, falling back to %s", timezone_name, DEFAULT_TIMEZONE)
        target_tz = ZoneInfo(DEFAULT_TIMEZONE)

    cleaned = utc_string.replace("Z", "+00:00")

    dt = None

    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue

    if dt is None:
        return utc_string

    if dt.tzinfo is not None:
        # Already timezone-aware - convert the actual instant to the
        # target zone (adjusts the wall-clock time correctly), don't
        # just relabel or append to it.
        local_dt = dt.astimezone(target_tz)
    else:
        # Naive - assume UTC, attach UTC first then convert, so DST
        # rules (where applicable) are resolved correctly rather than
        # applying a flat manual offset.
        local_dt = dt.replace(tzinfo=timezone.utc).astimezone(target_tz)

    return local_dt.isoformat()


def _display_time_12h(iso_string: Optional[str]) -> str:
    """12-hour AM/PM display string - DATA, not a sentence, so tools may
    still compute it (an LLM doing manual date arithmetic is unreliable;
    this is exactly why the hard rule in prompts.py tells it to use this
    field instead of formatting timestamps itself)."""

    if not iso_string:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "").split("+")[0])
    except ValueError:
        return iso_string
    return dt.strftime("%I:%M %p").lstrip("0") or dt.strftime("%I:%M %p")


def _display_date(iso_string: Optional[str]) -> str:
    if not iso_string:
        return "-"
    try:
        dt = datetime.fromisoformat(iso_string.replace("Z", "").split("+")[0])
    except ValueError:
        return iso_string
    return dt.strftime("%d/%m/%Y")


_FIELD_MAP = (
    ("ref", ("bookingRefNum",)),
    ("servicePrice", ("servicePrice",)),
    ("patientFullName", ("patientFullName",)),
    ("mobileNumber", ("mobileNumber",)),
    ("email", ("email",)),
    ("statusName", ("statusName",)),
    ("branchName", ("branchName",)),
    ("doctorName", ("doctorName",)),
    ("doctorId", ("doctorId",)),
    ("serviceName", ("serviceName",)),
    ("specialtyName", ("specialtyName",)),
)


def _shape_appointment(item: dict, timezone_name: str = DEFAULT_TIMEZONE) -> dict:
    """Flatten one raw API booking item into plain data fields - no
    sentences, just values, for the LLM to reference directly.
    `timezone_name` should be this client's own IANA zone (from
    state["templates"]["_timezone"]) - see to_riyadh()."""

    shaped = {}
    for name, keys in _FIELD_MAP:
        for key in keys:
            if key in item:
                shaped[name] = item[key]
                break

    local_from = to_riyadh(item.get("bookingTimeFrom"), timezone_name)
    local_to = to_riyadh(item.get("bookingTimeTo"), timezone_name)

    shaped["bookingTimeFrom"] = local_from
    shaped["bookingTimeTo"] = local_to
    shaped["date_display"] = _display_date(local_from)
    shaped["time_display"] = _display_time_12h(local_from)
    shaped["id"] = item.get("id")
    shaped["status"] = item.get("status")

    return shaped


def _filter_active(items: list) -> list:
    """Excludes bookings that can no longer be cancelled or rescheduled:
    already cancelled/completed/arrived/no-show, or past their own
    scheduled date. Applied to BOTH the reference-number and phone-number
    lookup paths (see lookup_appointment) - an earlier version only
    applied this to the phone path, preserving an asymmetry from the
    original n8n business logic; that asymmetry was explicitly removed
    per a later request: a past/inactive booking must never be offered
    for cancellation or rescheduling regardless of how it was found.

    CHANGED (explicit user request, based on a real dashboard screenshot):
    "active"/cancellable no longer requires a scheduled future visit date.
    It now means the booking's statusName indicates it HASN'T happened
    yet - i.e. anything other than Cancelled/Completed/Arrived. This
    specifically includes "New" bookings that don't have a visit date
    set yet at all (shown as "-" in the dashboard) - those are still
    perfectly cancellable and must appear.

    Previously this required `bookingTimeFrom` to be set AND in the
    future, which silently excluded every "New" booking without a
    visit date yet - that was the actual root cause of "no booking
    found" despite a visible, cancellable "New" row in the dashboard.

    ADDED BACK (explicit follow-up request): a booking with a scheduled
    visit date that has already passed must be excluded too, even if its
    status is still "New" (e.g. a no-show never updated in the source
    system) - it can't practically be cancelled anymore. A "New" booking
    with NO visit date set at all is still included (nothing to compare
    against - it hasn't happened by definition).

    STATUS CODES (confirmed directly from the Booking API's own
    documentation): New=1, Confirmed=2, Arrived=3, NoShow=4, Completed=5,
    Cancelled=6. Only New/Confirmed are cancellable. This now checks the
    NUMERIC `status` field as the primary, reliable mechanism (language-
    independent - no more guessing at Arabic vs English spelling), with
    the earlier string-based `statusName` matching kept only as a
    fallback for the rare item that might be missing a numeric status
    for some reason."""

    _excluded_keywords = (
        "cancelled", "canceled", "completed", "arrived", "no show", "no-show",
        "ملغ", "ألغي", "مكتمل", "منتهي", "وصل", "لم يحضر",
    )

    now = datetime.now(timezone.utc)

    active = []
    for item in items:
        status_code = item.get("status")

        if status_code is not None:
            if status_code not in CANCELLABLE_STATUS_CODES:
                continue
        else:
            # No numeric status on this item at all - fall back to the
            # string-based check as a defense-in-depth safety net.
            status_name = (item.get("statusName") or "").strip().lower()
            if any(keyword in status_name for keyword in _excluded_keywords):
                continue

        raw_from = item.get("bookingTimeFrom")
        if raw_from:
            try:
                dt = datetime.fromisoformat(raw_from.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    # Naive - assume UTC, same as this function always did.
                    dt = dt.replace(tzinfo=timezone.utc)
                if dt <= now:
                    continue  # has a scheduled date, and it's already passed
            except ValueError:
                pass  # unparsable date - don't let a bad format hide an otherwise-active booking

        active.append(item)

    return active


def _base_url(state: AgentState) -> str:
    return state.get("templates", {}).get("_base_url") or "https://demo.catalystsystems.io:1102"


# ==========================================================
# Tools - each returns STATUS + DATA ONLY, never a sentence
# ==========================================================

@tool
def validate_phone_format(phone: str) -> dict:
    """Validate that a phone number is in international format
    (starts with + and a country code). Returns {"status": "valid",
    "normalized": "+201234567890"} or {"status": "invalid"}."""

    if not _is_valid_phone_format(phone):
        return {"status": "invalid"}

    return {"status": "valid", "normalized": normalize_phone_number(phone)}


@tool
def compare_phone(provided_phone: str, channel_phone: str = "") -> dict:
    """Compare a user-provided phone number against the verified channel
    identity phone number (if any). Returns {"status": "match"} or
    {"status": "no_match"}. Never decide this yourself - always call
    this tool."""

    a = normalize_phone_number(provided_phone)
    b = normalize_phone_number(channel_phone) if channel_phone else None

    match = bool(a and b and a == b)

    logger.info(
        "compare_phone: provided=%r -> normalized=%r | channel=%r -> normalized=%r | match=%s",
        provided_phone, a, channel_phone, b, match,
    )

    if match:
        return {"status": "match"}

    return {"status": "no_match"}


@tool
def lookup_appointment(
    state: Annotated[AgentState, InjectedState],
    ref_number: str = "",
    phone: str = "",
    use_channel_identity: bool = False,
    language: str = "en",
) -> dict:
    """Look up bookings by reference number OR phone number.

    If the user chose to cancel by phone number and a verified channel
    identity (e.g. their WhatsApp number) is already known, call this
    with `use_channel_identity=True` and leave `phone` empty - this
    automatically searches using that verified number WITHOUT you ever
    needing to ask the user to type it, and WITHOUT you ever seeing the
    actual digits yourself. Any booking found this way is by definition
    already verified (it was found using their own verified channel
    number), so NO OTP is ever needed in this case - skip straight to
    STEP 3/4 of the flow.

    Only ask the user to type a phone number, and only then go through
    compare_phone/OTP, if `use_channel_identity` returns "no_channel_identity"
    (there is none available) or if the user explicitly says the booking
    is under a DIFFERENT number than the one they're messaging from.

    ALWAYS pass `language` as "ar" if you are about to reply to the user
    in Arabic (any dialect), or "en" if replying in English - this makes
    the booking system return doctor/branch/service names already
    spelled correctly in that language, so you never have to translate
    or transliterate a name yourself (which risks misspelling it).

    Returns one of:
    {"status": "not_found"}
    {"status": "found_one", "appointment": {...}}
    {"status": "found_many", "appointments": [...]}
    {"status": "found_but_inactive"}  # a booking exists under this ref/phone,
                          # but it's already cancelled, completed, or its
                          # own date/time has already passed - it can no
                          # longer be cancelled or rescheduled. Tell the
                          # user plainly why, don't just say "not found"
                          # (which would wrongly imply they mistyped
                          # something).
    {"status": "error"}  # the booking API call itself failed - a technical
                          # problem, NOT the same as "no booking exists"
    {"status": "no_channel_identity"}  # use_channel_identity was True but
                          # no verified channel number is available - ask
                          # the user to type their phone number instead
    Appointment fields: ref, doctorName, branchName, serviceName,
    specialtyName, statusName, date_display, time_display, patientFullName,
    mobileNumber, email, id."""

    if use_channel_identity:
        channel_phone = state.get("channel_phone")
        logger.info("lookup_appointment: use_channel_identity=True, channel_phone=%r", channel_phone)
        if not channel_phone:
            return {"status": "no_channel_identity"}
        phone = channel_phone

    base_url = _base_url(state)

    if ref_number:
        result = api.get_bookings_by_ref(base_url, ref_number, language=language)
    elif phone:
        result = api.get_bookings_by_phone(
            base_url, normalize_phone_number(phone), language=language,
            status_list=list(CANCELLABLE_STATUS_CODES),
        )
    else:
        return {"status": "not_found"}

    if not result["success"]:
        # IMPORTANT: this used to silently return "not_found" for ANY
        # failure - timeouts, wrong base_url, 4xx/5xx, bad JSON - making
        # a real connectivity/config problem indistinguishable from a
        # genuinely empty result, both to logs and to the user. Now it's
        # logged with the real reason and reported as a distinct
        # "error" status so the LLM (per prompts.py) tells the user
        # there was a technical problem instead of "no booking found".
        logger.error(
            "lookup_appointment API call failed: base_url=%s ref=%r phone=%r status_code=%s error=%s",
            base_url, ref_number, phone, result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])

    if not items:
        return {"status": "not_found"}

    # CHANGED (explicit request): both paths now apply the same
    # active-only filter (excludes cancelled/completed/already-passed
    # bookings). This used to only apply to the phone path, matching the
    # original n8n business logic's asymmetry - that asymmetry is no
    # longer wanted: a past/inactive booking must never be offered for
    # cancellation or rescheduling, regardless of how it was looked up.
    active_items = _filter_active(items)
    if not active_items:
        # A booking WAS found, but every match is already past/cancelled/
        # completed - distinct from "no booking with that ref/phone at
        # all", so the LLM can say why plainly instead of implying they
        # may have mistyped something.
        return {"status": "found_but_inactive"}
    items = active_items

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    shaped = [_shape_appointment(i, timezone_name) for i in items]

    if len(shaped) == 1:
        return {"status": "found_one", "appointment": shaped[0]}

    return {"status": "found_many", "appointments": shaped}


@tool
def check_booking_status(
    state: Annotated[AgentState, InjectedState],
    ref_number: str,
    language: str = "en",
) -> dict:
    """Re-fetch a booking by its reference number IMMEDIATELY before
    cancelling it - never trust anything earlier in the conversation as
    still current. ALWAYS pass `language` as "ar" or "en" matching what
    you're about to reply in (see lookup_appointment). Returns:
    {"status": "active", "appointment": {...}}
    {"status": "already_cancelled", "appointment": {...}}
    {"status": "not_found"}
    {"status": "error"}  # the booking API call itself failed - a technical
                          # problem, NOT the same as "booking not found"
    """

    base_url = _base_url(state)
    result = api.get_bookings_by_ref(base_url, ref_number, language=language)

    if not result["success"]:
        logger.error(
            "check_booking_status API call failed: base_url=%s ref=%r status_code=%s error=%s",
            base_url, ref_number, result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    appt = _shape_appointment(items[0], timezone_name)

    if appt.get("statusName") == CANCELLED_STATUS_NAME:
        return {"status": "already_cancelled", "appointment": appt}

    return {"status": "active", "appointment": appt}


@tool
def cancel_appointment(
    state: Annotated[AgentState, InjectedState],
    booking_id: str,
) -> dict:
    """Cancel a booking by its internal id (from a previous tool's
    "appointment"/"id" field - NEVER the human-readable reference
    number). Always call check_booking_status on the same booking
    immediately before this. Returns {"status": "success"} or
    {"status": "error"}."""

    base_url = _base_url(state)
    result = api.cancel_booking_by_guid(base_url, booking_id)

    if result["success"]:
        return {"status": "success"}

    return {"status": "error"}


# ==========================================================
# OTP (dummy provider by default, Authentica when configured) - internal
# mechanics unchanged from the old tools.py, only the return shape changed
# ==========================================================

_otp_storage: Dict[str, dict] = {}


@tool
def send_otp(state: Annotated[AgentState, InjectedState], phone: str) -> dict:
    """Send an OTP code to the given phone number (the number ON FILE
    for the booking, not necessarily what the user typed). Returns
    {"status": "otp_sent"}, or {"status": "otp_not_needed_matches_channel"}
    if this number turns out to match the user's own verified channel
    identity (see note below) - in that case, treat it exactly like a
    successful compare_phone match: skip OTP entirely and continue
    straight to looking up the appointment.

    SAFETY NET: this checks the phone number against the channel
    identity itself before sending anything, even though you should
    already have called `compare_phone` before ever calling this tool -
    this is a defensive backstop in case that step was skipped, not a
    replacement for calling `compare_phone` first."""

    normalized = normalize_phone_number(phone)

    channel_phone = state.get("channel_phone")
    normalized_channel = normalize_phone_number(channel_phone) if channel_phone else None

    if normalized_channel and normalized and normalized_channel == normalized:
        logger.warning(
            "send_otp called for phone=%r which matches channel_phone=%r - "
            "skipping OTP entirely (compare_phone should have caught this "
            "before send_otp was ever called)",
            normalized, normalized_channel,
        )
        return {"status": "otp_not_needed_matches_channel"}

    if OTP_PROVIDER == "authentica":
        api.authentica_send_otp(normalized)
        return {"status": "otp_sent"}

    _otp_storage[normalized] = {"otp": TEST_OTP, "created_at": time.time()}
    logger.info("OTP sent for %s (test otp=%s)", normalized, TEST_OTP)
    return {"status": "otp_sent"}


@tool
def verify_otp(phone: str, otp: str) -> dict:
    """Verify a user-entered OTP code against the one sent to `phone`.
    Returns {"status": "otp_valid"} or {"status": "otp_invalid"}."""

    normalized = normalize_phone_number(phone)

    if OTP_PROVIDER == "authentica":
        result = api.authentica_verify_otp(normalized, otp)
        return {"status": "otp_valid" if result["success"] else "otp_invalid"}

    record = _otp_storage.get(normalized)

    if not record:
        return {"status": "otp_invalid"}

    if time.time() - record["created_at"] > OTP_TTL_SECONDS:
        return {"status": "otp_invalid"}

    if str(otp).strip() == str(record["otp"]):
        return {"status": "otp_valid"}

    return {"status": "otp_invalid"}


# ==========================================================
# Medical Concierge (symptom -> specialty -> available doctor guidance)
# ==========================================================
#
# Confirmed directly by the user: the Doctors/Specialties API is scoped
# to the correct clinic by its own base_url alone (like GuestBookings),
# on a separate port (1102) from GuestBookings (1101). Response shapes
# confirmed directly from the API's own Swagger "Execute" output.

def _doctors_base_url(state: AgentState) -> Optional[str]:
    """Returns this client's configured Doctors/Specialties API base_url,
    or None if it isn't configured for this client at all. Deliberately
    NEVER falls back to some other client's URL - see config.py's
    extensive comment on why (a real cross-tenant data leak risk)."""

    return (state.get("templates") or {}).get("_doctors_base_url")


@tool
def list_specialties(state: Annotated[AgentState, InjectedState]) -> dict:
    """List every medical specialty this clinic actually offers. ALWAYS
    call this before suggesting a specialty to a user describing a
    symptom/concern - never guess whether this clinic has a given
    specialty. Returns:
    {"status": "found", "specialties": [{"id": ..., "name": ...}, ...]}
    {"status": "not_found"}  # this clinic has no specialties registered
    {"status": "not_configured"}  # this clinic doesn't have this feature set up yet
    {"status": "error"}  # the API call itself failed

    IMPLEMENTATION NOTE: this uses the Specialties/GetList endpoint
    directly. An earlier version derived specialties from the doctors
    endpoint instead, after an initial test call to Specialties/GetList
    returned mismatched placeholder data ("New NEw", unrelated ids).
    That turned out to be stale/unrelated test data, not a real problem
    with the endpoint - a follow-up call (after fixing pageNumber=1 and
    the /GetList path) returned the correct, complete specialty list,
    confirmed to share the exact same ids as the doctors' own
    specialtyId field. Using this endpoint (rather than deriving from
    doctors) is more correct: it includes every specialty this clinic
    has registered, even ones with zero doctors currently assigned -
    letting the agent correctly say "we don't offer that" only when
    truly true, rather than only when nobody happens to be staffed."""

    base_url = _doctors_base_url(state)

    if not base_url:
        logger.warning("list_specialties called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    result = api.get_specialties(base_url)

    if not result["success"]:
        logger.error(
            "list_specialties API call failed: base_url=%s status_code=%s error=%s",
            base_url, result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])

    # The API's name fields are usually populated (confirmed against
    # real data), but fall back through the alternatives defensively -
    # dropping anything with no usable name at all rather than
    # surfacing a blank to the user.
    specialties = []
    for item in items:
        if not item.get("id"):
            continue

        name = (
            item.get("name")
            or item.get("formatedName")
            or item.get("altName")
            or item.get("code")
        )

        if not name or not str(name).strip():
            logger.warning("Skipping specialty with no usable name: id=%s", item.get("id"))
            continue

        specialties.append({"id": item["id"], "name": str(name).strip()})

    logger.info("list_specialties: %d specialties returned, %d usable", len(items), len(specialties))

    if not specialties:
        return {"status": "not_found"}

    return {"status": "found", "specialties": specialties}


@tool
def find_available_doctors(
    state: Annotated[AgentState, InjectedState],
    specialty_ids: list,
    days_ahead: int = DOCTOR_AVAILABILITY_WINDOW_DAYS,
) -> dict:
    """Find doctors who currently have a bookable service AND an available
    schedule slot within the next `days_ahead` days, across one or more
    specialties. ALWAYS call `list_specialties` first to get correct ids
    - never guess or invent one.

    IMPORTANT: pass ALL plausibly-matching specialty ids in ONE call as a
    list, not just the single most obvious one. Clinics often have both
    a general specialty and a more specific sub-specialty that could
    both reasonably cover the same complaint (e.g. "Ophthalmology" AND
    "Vitreoretinal Surgery" both relate to eye problems). If more than
    one specialty from `list_specialties` could plausibly match what the
    user described, include all of their ids here together - e.g.
    specialty_ids=["<ophthalmology-id>", "<vitreoretinal-surgery-id>"] -
    so a doctor registered under any of them is found. Do not conclude
    "no doctors available" after checking only one plausible specialty.

    Returns:
    {"status": "found", "doctors": [{"id", "name", "specialtyName", "degreeName"}, ...]}
    {"status": "not_found"}  # these specialties exist, but no doctor has availability right now
    {"status": "not_configured"}  # this clinic doesn't have this feature set up yet
    {"status": "error"}  # the API call itself failed"""

    base_url = _doctors_base_url(state)

    if not base_url:
        logger.warning("find_available_doctors called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    now = datetime.utcnow()
    intersection_start = now.isoformat() + "Z"
    intersection_end = (now + timedelta(days=days_ahead)).isoformat() + "Z"

    result = api.get_doctors(
        base_url,
        specialty_ids=specialty_ids,
        has_published_service=True,
        has_service_schedule=True,
        intersection_start=intersection_start,
        intersection_end=intersection_end,
    )

    if not result["success"]:
        logger.error(
            "find_available_doctors API call failed: base_url=%s specialty_ids=%s status_code=%s error=%s",
            base_url, specialty_ids, result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])

    # The server already filtered by hasPublishedService +
    # hasServiceSchedule + the intersection window, so treat hasSlots as
    # a refinement only: exclude a doctor ONLY when the API explicitly
    # says hasSlots is False. Previously this required hasSlots to be
    # truthy, which silently discarded EVERY doctor whenever the field
    # was absent or null in the response - indistinguishable from
    # "nobody is available".
    available = [i for i in items if i.get("hasSlots") is not False]

    logger.info(
        "find_available_doctors: specialty_ids=%s api_returned=%d after_hasSlots_filter=%d",
        specialty_ids, len(items), len(available),
    )

    if not available:
        return {"status": "not_found"}

    doctors = []
    for i in available:
        name = i.get("name") or i.get("formatedName") or i.get("altName")
        if not name or not str(name).strip():
            logger.warning("Skipping doctor with no usable name: id=%s", i.get("id"))
            continue

        doctors.append({
            "id": i.get("id"),
            "name": str(name).strip(),
            "specialtyName": i.get("specialtyName"),
            "degreeName": i.get("degreeName"),
        })

    if not doctors:
        return {"status": "not_found"}

    return {"status": "found", "doctors": doctors}


# ==========================================================
# Reschedule Appointment (change an existing booking's time)
# ==========================================================
#
# Reuses lookup_appointment/compare_phone/send_otp/verify_otp exactly
# as-is for identifying the booking and verifying identity - see
# prompts.py's RESCHEDULE FLOW, which mirrors the same STEP 1-3 logic
# already used for cancellation. These three tools cover what's new:
# checking the doctor's schedule/availability and performing the update.

def _resolve_doctor_id(state: AgentState, ref_number: str, language: Optional[str]) -> dict:
    """Internal helper: look up a booking by its reference number and
    return its doctorId, so schedule/slot tools know which doctor to
    query without the LLM ever having to know or pass a doctor's GUID
    directly. Returns {"status": "found", "doctor_id": ...} or an error
    status matching lookup_appointment's own conventions."""

    base_url = _base_url(state)
    result = api.get_bookings_by_ref(base_url, ref_number, language=language)

    if not result["success"]:
        logger.error("_resolve_doctor_id: API call failed for ref_number=%s error=%s", ref_number, result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    doctor_id = items[0].get("doctorId")
    if not doctor_id:
        logger.warning("_resolve_doctor_id: booking found but has no doctorId - ref_number=%s", ref_number)
        return {"status": "error"}

    return {"status": "found", "doctor_id": doctor_id}


_WEEKDAY_NAMES = {
    # English (case-insensitive)
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
    # Arabic
    "الاثنين": 0, "الإثنين": 0, "اثنين": 0,
    "الثلاثاء": 1, "ثلاثاء": 1,
    "الأربعاء": 2, "الاربعاء": 2, "أربعاء": 2, "اربعاء": 2,
    "الخميس": 3, "خميس": 3,
    "الجمعة": 4, "جمعة": 4,
    "السبت": 5, "سبت": 5,
    "الأحد": 6, "الاحد": 6, "أحد": 6, "احد": 6,
}


@tool
def get_next_weekday_date(
    weekday_name: str,
    after_date: str = "",
    timezone_name: str = DEFAULT_TIMEZONE,
) -> dict:
    """Resolve a weekday NAME (e.g. "Thursday"/"الخميس") to an actual
    calendar date, computed exactly - NEVER work out which calendar date
    a weekday name falls on yourself, your own mental date arithmetic is
    not reliable enough for this and has caused real incorrect answers
    before (e.g. calling a date "Thursday" that was not actually a
    Thursday). ALWAYS call this tool instead, every time a user names a
    day of the week rather than a specific date.

    Two modes:
    - `after_date` OMITTED (empty): returns the NEXT upcoming date for
      that weekday counting from TODAY. If today itself already IS that
      weekday, returns TODAY's date.
    - `after_date` GIVEN (format "YYYY-MM-DD", e.g. from an earlier call
      to this same tool, or from get_available_reschedule_slots): returns
      the next occurrence of that weekday STRICTLY AFTER that date - use
      this whenever the user refers to a day relative to one you already
      discussed (e.g. "the following Monday" / "الاثنين اللي بعده" after
      you'd already established a specific Monday's date) - do NOT ask
      them to clarify what date they mean, just call this directly.
    Returns:
    {"status": "found", "date": "YYYY-MM-DD", "weekday_name": "Thursday"}
    {"status": "error"}  # unrecognized weekday name or bad after_date"""

    key = (weekday_name or "").strip().lower()
    target_weekday = _WEEKDAY_NAMES.get(key)

    if target_weekday is None:
        logger.warning("get_next_weekday_date: unrecognized weekday_name=%r", weekday_name)
        return {"status": "error"}

    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    if after_date:
        try:
            reference = date.fromisoformat(after_date.strip())
        except ValueError:
            logger.warning("get_next_weekday_date: invalid after_date=%r", after_date)
            return {"status": "error"}
        days_ahead = (target_weekday - reference.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7  # strictly AFTER the reference date, never the same day
    else:
        reference = datetime.now(tz).date()
        days_ahead = (target_weekday - reference.weekday()) % 7

    target_date = reference + timedelta(days=days_ahead)
    english_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][target_weekday]

    return {"status": "found", "date": target_date.isoformat(), "weekday_name": english_name}


@tool
def get_doctor_schedule(
    state: Annotated[AgentState, InjectedState],
    ref_number: str,
    target_date: str = "",
    language: str = "en",
) -> dict:
    """Get the GENERAL RECURRING weekly schedule of the doctor on a given
    booking - which weekdays they work, their daily start/end times, and
    the date range this schedule is valid for. Call this BEFORE offering
    to reschedule, to know which days of the week are even worth
    checking - this does NOT return specific open time slots (use
    `get_available_reschedule_slots` for that once you've picked a
    target date).

    `target_date` (format "YYYY-MM-DD"), if you already have one in mind
    (e.g. from `get_next_weekday_date`), filters to only the schedule
    row(s) actually valid/effective on that specific date - avoiding
    stale/expired or not-yet-started schedule rows for the same doctor.
    If omitted, defaults to today.
    Returns:
    {"status": "found", "schedules": [{"recurringDaysNames": [...], "fromDateTime": ..., "toDateTime": ...}, ...]}
    {"status": "not_found"}  # booking or schedule doesn't exist
    {"status": "not_configured"}  # this clinic doesn't have this feature set up yet
    {"status": "error"}"""

    resolved = _resolve_doctor_id(state, ref_number, language)
    if resolved["status"] != "found":
        return resolved

    base_url = _doctors_base_url(state)

    if not base_url:
        logger.warning("get_doctor_schedule called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    if target_date:
        effective_date = target_date
    else:
        try:
            effective_date = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
        except Exception:
            effective_date = None

    result = api.get_doctor_schedule(base_url, doctor_ids=[resolved["doctor_id"]], effective_date=effective_date)

    if not result["success"]:
        logger.error("get_doctor_schedule API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    schedules = [
        {
            "recurringDaysNames": item.get("recurringDaysNames"),
            "fromDateTime": to_riyadh(item.get("fromDateTime"), timezone_name),
            "toDateTime": to_riyadh(item.get("toDateTime"), timezone_name),
            "branchName": item.get("branchName"),
            "doctorName": item.get("doctorName"),
        }
        for item in items
    ]

    return {"status": "found", "schedules": schedules}


@tool
def get_available_reschedule_slots(
    state: Annotated[AgentState, InjectedState],
    ref_number: str,
    from_date: str,
    to_date: str,
    language: str = "en",
) -> dict:
    """Get the doctor's ACTUAL open time slots (not just working days)
    for the booking's doctor, within [from_date, to_date] - both in ISO
    format, e.g. "2026-05-01T09:00:00+03:00". Only genuinely available
    (not already booked) slots are returned. Call `get_doctor_schedule`
    first to know which weekdays/hours are worth checking, then call
    this with a specific day's full working-hours range to see the
    exact bookable times. Returns:
    {"status": "found", "slots": [{"slotStart": ..., "slotEnd": ..., "date_display": ..., "time_display": ..., "doctorName": ..., "serviceName": ..., "servicePrice": ...}, ...]}
    {"status": "not_found"}  # no open slots in this range
    {"status": "not_configured"}  # this clinic doesn't have this feature set up yet
    {"status": "error"}"""

    resolved = _resolve_doctor_id(state, ref_number, language)
    if resolved["status"] != "found":
        return resolved

    # Safety net: if the range came in backwards (from_date after
    # to_date), swap them. Confirmed directly in production: the LLM
    # passed from_date=09:00 and to_date=07:00 (inverted) - the real API
    # appears to silently ignore date filtering entirely when given a
    # nonsensical inverted range, returning generic/unfiltered slots
    # instead (which is what caused already-passed times to still
    # appear). Guaranteeing a valid, forward-ordered range here removes
    # dependence on the LLM getting the order right.
    try:
        if from_date and to_date and datetime.fromisoformat(from_date) > datetime.fromisoformat(to_date):
            logger.warning(
                "get_available_reschedule_slots: from_date=%r was AFTER to_date=%r - swapping them",
                from_date, to_date,
            )
            from_date, to_date = to_date, from_date
    except ValueError:
        pass  # let the API itself reject a genuinely malformed date string

    base_url = _doctors_base_url(state)

    if not base_url:
        logger.warning("get_available_reschedule_slots called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    result = api.get_doctor_schedule_slots(
        base_url, doctor_ids=[resolved["doctor_id"]],
        from_date=from_date, to_date=to_date, is_booked=False,
    )

    if not result["success"]:
        logger.error("get_available_reschedule_slots API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    # Defense in depth: exclude any item explicitly marked isBooked=True,
    # even though is_booked=False was already sent as a request filter -
    # other endpoints in this same system have been observed to not
    # always respect their own request filters (e.g. the inverted
    # from_date/to_date range issue), so don't rely on the request filter
    # alone for something this important (double-booking a doctor).
    items = [i for i in items if i.get("isBooked") is not True]
    if not items:
        return {"status": "not_found"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    slots = []
    for item in items:
        slot_start = to_riyadh(item.get("slotStart"), timezone_name)
        slot_end = to_riyadh(item.get("slotEnd"), timezone_name)
        slots.append({
            "slotStart": slot_start,
            "slotEnd": slot_end,
            "date_display": _display_date(slot_start),
            "time_display": _display_time_12h(slot_start),
            "doctorName": item.get("doctorName"),
            "serviceName": item.get("serviceName"),
            "servicePrice": item.get("servicePrice"),
        })

    # Exclude slots that have already passed - a slot for TODAY earlier
    # than right now must never still be offered (observed directly:
    # 9:00 AM was still shown while the conversation was happening at
    # ~5pm the same day). Compared in the client's own local timezone,
    # matching how slotStart itself was already converted.
    try:
        now_local = datetime.now(ZoneInfo(timezone_name))
        slots = [
            s for s in slots
            if s["slotStart"] and datetime.fromisoformat(s["slotStart"]) > now_local
        ]
    except Exception:
        logger.exception("get_available_reschedule_slots: failed to filter past slots, showing all")

    if not slots:
        return {"status": "not_found"}

    # Always chronological - the API's own return order was observed to
    # be scrambled in production (slots came back neither ascending nor
    # descending), and relying on the LLM to re-sort dozens of items
    # correctly by eye is not realistic. Sort here, once, in code.
    slots.sort(key=lambda s: s["slotStart"] or "")

    # Deduplicate by exact start time - confirmed directly in production:
    # the API returned every distinct time TWICE in a row (e.g. "11:00
    # AM, 11:00 AM, 12:00 PM, 12:00 PM, ..."), likely once per underlying
    # resource/service sharing the same schedule slot. The user must
    # never see the same bookable time offered more than once.
    seen_starts = set()
    deduped = []
    for s in slots:
        key = s["slotStart"]
        if key in seen_starts:
            continue
        seen_starts.add(key)
        deduped.append(s)
    if len(deduped) != len(slots):
        logger.warning("get_available_reschedule_slots: removed %d duplicate slot(s) with the same start time", len(slots) - len(deduped))
    slots = deduped

    # Cap to a reasonable, actually-usable count for a chat interface.
    # Observed in production: a too-wide [from_date, to_date] query
    # returned 44 slots spanning nearly 24 hours - regardless of why
    # that range was too wide, showing dozens of options in a chat
    # message is not usable. This guarantees a sane result independent
    # of whether the date-range scoping prompt guidance is followed.
    MAX_SLOTS_TO_SHOW = 20
    if len(slots) > MAX_SLOTS_TO_SHOW:
        logger.warning(
            "get_available_reschedule_slots: %d slots returned for range [%s, %s] - "
            "capping to the first %d chronologically (this usually means the "
            "queried date range was wider than a single day's actual working hours)",
            len(slots), from_date, to_date, MAX_SLOTS_TO_SHOW,
        )
        slots = slots[:MAX_SLOTS_TO_SHOW]

    return {"status": "found", "slots": slots}


@tool
def reschedule_appointment(
    state: Annotated[AgentState, InjectedState],
    booking_id: str,
    new_time_from: str,
    new_time_to: str,
) -> dict:
    """Change an existing booking to a new time. `booking_id` MUST be the
    booking's own "id" field (a GUID) from a FRESH `lookup_appointment`
    or `check_booking_status` call in THIS conversation - never invent
    or reuse an old value from memory. `new_time_from`/`new_time_to` must
    be the EXACT slotStart/slotEnd values from `get_available_reschedule_slots`
    - never modify or recompute them yourself. Returns:
    {"status": "success"} or {"status": "error"}"""

    # NOTE: confirmed directly from the user's own curl - GuestBookings/Update
    # lives on the SAME port as Doctors/Specialties (1302), NOT the regular
    # GuestBookings port used for cancellation (1101), despite the "GuestBookings"
    # name. Trusting the confirmed URL over the path-name convention.
    base_url = _doctors_base_url(state)

    if not base_url:
        logger.warning("reschedule_appointment called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "error"}

    result = api.reschedule_booking(base_url, booking_id, new_time_from, new_time_to)

    if not result["success"]:
        logger.error(
            "reschedule_appointment API call failed: booking_id=%s status_code=%s error=%s",
            booking_id, result.get("status_code"), result.get("error"),
        )
        return {"status": "error"}

    return {"status": "success"}


# ==========================================================
# General Hospital FAQ (RAG over a per-client knowledge base document)
# ==========================================================
#
# Fully generic - reads whichever file client_config.csv's
# knowledge_base_file column points to for THIS client_id (see rag.py).
# Adding a new clinic's FAQ knowledge base is just adding a text file and
# setting that column - no code changes needed.

@tool
def answer_hospital_faq(
    state: Annotated[AgentState, InjectedState],
    question: str,
) -> dict:
    """Look up this clinic's own general information (vision, mission,
    values, goals, services offered, branch addresses/contact details,
    policies, partners, etc.) to answer an FAQ-style question - NOT for
    schedules, availability, or booking (those have their own tools).
    Returns the most relevant passages found; summarize them naturally
    in 2-3 sentences rather than reproducing them verbatim. Returns:
    {"status": "found", "passages": ["...", ...]}
    {"status": "not_found"}  # nothing relevant enough was found
    {"status": "not_configured"}  # this clinic has no FAQ knowledge base set up yet"""

    kb_file = (state.get("templates") or {}).get("_knowledge_base_file", "")

    if not kb_file:
        logger.warning("answer_hospital_faq called but no knowledge_base_file is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    passages = rag.search(kb_file, question)

    if not passages:
        return {"status": "not_found"}

    return {"status": "found", "passages": passages}


# ==========================================================
# Doctor/Branch info lookup (fuzzy name matching + listing)
# ==========================================================
#
# READ-ONLY FAQ/info lookup - never touches booking/availability. Fully
# generic: works off whatever Doctors/GetList and Branches/GetList
# return for THIS client_id, no per-clinic hardcoding.

def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text for fuzzy comparison: strip diacritics and
    collapse common letter variants (alef forms, ta marbuta/ha, alef
    maksura/ya) so typo/spelling variations still match."""

    if not text:
        return ""

    text = str(text).strip().lower()
    # Strip Arabic diacritics (tashkeel)
    text = re.sub(r"[\u064B-\u0652\u0670]", "", text)
    # Normalize alef variants -> ا
    text = re.sub(r"[إأآٱ]", "ا", text)
    # Normalize ta marbuta -> ه, alef maksura -> ي
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _fuzzy_match(user_input: str, candidates: list, name_keys: list) -> dict:
    """Match `user_input` against `candidates` (list of raw API items),
    checking each of `name_keys` per candidate. Returns:
    {"result": "matched", "item": {...}, "score": 0.0-1.0}
    {"result": "ambiguous", "items": [...]}   # 2+ close, similarly-scored matches
    {"result": "not_matched"}

    `score` lets callers distinguish a high-confidence match (exact or
    unique) from a lower-confidence one that's still worth confirming
    with the user (likely typo) - see match_entity_for_booking."""

    import difflib

    normalized_input = _normalize_arabic(user_input)
    if not normalized_input:
        return {"result": "not_matched"}

    scored = []
    for item in candidates:
        best_score = 0.0
        for key in name_keys:
            value = item.get(key)
            if not value:
                continue
            normalized_value = _normalize_arabic(value)
            if normalized_input == normalized_value:
                best_score = max(best_score, 1.0)
            elif normalized_input in normalized_value or normalized_value in normalized_input:
                best_score = max(best_score, 0.96)
            else:
                ratio = difflib.SequenceMatcher(None, normalized_input, normalized_value).ratio()
                best_score = max(best_score, ratio)
        if best_score >= 0.6:
            scored.append((item, best_score))

    if not scored:
        return {"result": "not_matched"}

    scored.sort(key=lambda pair: pair[1], reverse=True)

    top_score = scored[0][1]
    close_matches = [item for item, score in scored if score >= top_score - 0.08]

    if len(close_matches) == 1 or top_score >= 0.98:
        return {"result": "matched", "item": close_matches[0], "score": top_score}

    return {"result": "ambiguous", "items": close_matches[:5]}


@tool
def match_entity_info(
    state: Annotated[AgentState, InjectedState],
    user_input: str,
    entity_type: str,
) -> dict:
    """FAQ/info lookup for doctors and branches - fuzzy name matching +
    listing. READ-ONLY: never touches booking, schedules, or
    availability - use the other tools for those.

    DUAL MODE:
      LIST MODE (user_input=""): returns ALL doctors or ALL branches as
        a list for display.
      RESOLVE MODE (user_input="user's raw text"): fuzzy-matches to ONE
        entity and returns its details. Tolerates Arabic typos, letter
        substitutions, and partial names - always pass the user's raw
        text, don't pre-process it yourself.

    `entity_type`: "doctor" or "branch".

    Returns one of:
    {"status": "list", "items": [...]}
    {"status": "matched", "item": {...}}
    {"status": "ambiguous", "candidates": [...]}  # show each candidate's
        name and ask the user which one they meant
    {"status": "not_matched"}
    {"status": "not_configured"}  # no doctors_base_url set up for this client
    {"status": "error"}

    Doctor fields: formatedName, altName, degreeName, specialtyName,
    defaultServiceName (serviceName), defaultServiceFee (servicePrice).
    Branch fields: name, altName, address, cityName, countryName,
    stateName, email, mobile."""

    entity_type = (entity_type or "").strip().lower()
    if entity_type not in ("doctor", "branch"):
        return {"status": "error"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("match_entity_info called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    if entity_type == "doctor":
        result = api.get_doctors(base_url, page_size=200)
        name_keys = ["formatedName", "altName", "name"]
    else:
        result = api.get_branches(base_url, page_size=200)
        name_keys = ["name", "altName", "formatedName", "cityName"]

    if not result["success"]:
        logger.error("match_entity_info API call failed: entity_type=%s status_code=%s error=%s", entity_type, result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])

    if not user_input or not user_input.strip():
        if entity_type == "doctor":
            shaped = [
                {
                    "formatedName": i.get("formatedName") or i.get("name"),
                    "altName": i.get("altName"),
                    "specialtyName": i.get("specialtyName"),
                    "degreeName": i.get("degreeName"),
                }
                for i in items
            ]
        else:
            shaped = [
                {
                    "name": i.get("name") or i.get("formatedName"),
                    "altName": i.get("altName"),
                    "address": i.get("address"),
                    "cityName": i.get("cityName"),
                }
                for i in items
            ]
        if not shaped:
            return {"status": "not_matched"}
        return {"status": "list", "items": shaped}

    match_result = _fuzzy_match(user_input, items, name_keys)

    if match_result["result"] == "not_matched":
        return {"status": "not_matched"}

    def _shape_doctor(i):
        return {
            "formatedName": i.get("formatedName") or i.get("name"),
            "altName": i.get("altName"),
            "degreeName": i.get("degreeName"),
            "specialtyName": i.get("specialtyName"),
            "serviceName": i.get("defaultServiceName") or i.get("serviceName"),
            "servicePrice": i.get("defaultServiceFee") or i.get("servicePrice"),
        }

    def _shape_branch(i):
        return {
            "name": i.get("name") or i.get("formatedName"),
            "altName": i.get("altName"),
            "address": i.get("address"),
            "cityName": i.get("cityName"),
            "countryName": i.get("countryName"),
            "stateName": i.get("stateName"),
            "email": i.get("email"),
            "mobile": i.get("mobile"),
        }

    shape_fn = _shape_doctor if entity_type == "doctor" else _shape_branch

    if match_result["result"] == "matched":
        return {"status": "matched", "item": shape_fn(match_result["item"])}

    return {"status": "ambiguous", "candidates": [shape_fn(i) for i in match_result["items"]]}


# ==========================================================
# New Booking (create a brand new appointment)
# ==========================================================
#
# Uses an internal per-session "booking session" store (module-level
# dict keyed by session_id) so the LLM never has to handle or pass raw
# doctor/branch UUIDs itself. This mirrors a deliberate, battle-tested
# design confirmed directly from a real production n8n system: even
# with full conversation history available to the model, having it
# re-type or pass UUIDs reliably was NOT safe enough in practice there.
# Tools read/write these fields directly via session_id; the LLM only
# ever passes plain names/text, never IDs.

_BOOKING_SESSIONS: Dict[str, dict] = {}


def _get_booking_session(session_id: str) -> dict:
    return _BOOKING_SESSIONS.setdefault(session_id, {
        "doctor_id": None, "branch_id": None, "service_id": None,
        "last_list": None,  # {"entity_type": "doctor"/"branch", "items": [shaped items]}
    })


@tool
def reset_booking_session(state: Annotated[AgentState, InjectedState]) -> dict:
    """Clear any previously-confirmed doctor/branch/service for a NEW
    booking. Call this as the FIRST action whenever the user starts a
    brand new booking ("حجز جديد"/"new booking"/"ابي احجز"), or
    explicitly wants to change branch or start completely over - this
    prevents a stale doctor/branch from a PREVIOUS booking earlier in
    this same conversation from silently carrying over and filtering
    results. Do NOT call this mid-flow otherwise (e.g. not just because
    the user picked a different day or time - only for a genuine restart
    or explicit branch change). Returns {"status": "reset"}."""

    session_id = state.get("session_id")
    _BOOKING_SESSIONS[session_id] = {"doctor_id": None, "branch_id": None, "service_id": None, "last_list": None}
    return {"status": "reset"}


@tool
def match_entity_for_booking(
    state: Annotated[AgentState, InjectedState],
    user_input: str,
    entity_type: str,
) -> dict:
    """Resolve a doctor or branch by the user's raw text for a NEW
    BOOKING, AND automatically confirm+remember it in this booking's
    session - you NEVER need to track, save, or pass any ID yourself;
    this tool handles that entirely, including filtering doctors to an
    already-confirmed branch automatically.

    DUAL MODE:
      LIST MODE (user_input=""): lists all doctors/branches. If a
        branch is already confirmed in this booking session and
        entity_type="doctor", the list is automatically filtered to
        doctors at that branch only - you don't need to filter it
        yourself or pass the branch.
      RESOLVE MODE (user_input="user's raw text"): matches to ONE
        entity. This also accepts a bare number referring to a position
        in the list you most recently showed via this same tool (e.g.
        user replies "2" after you displayed a numbered list) - always
        pass the user's raw text/number as-is, the tool handles both
        cases.

    `entity_type`: "doctor" or "branch".

    Returns one of:
    {"matched": true, "needsConfirmation": false, "item": {...}}
        -> CONFIRMED AND SAVED to the booking session automatically -
           do NOT ask "are you sure" for this case, proceed directly.
    {"matched": true, "needsConfirmation": true, "item": {...}}
        -> a close-but-not-exact match (likely a typo) - nothing was
           saved yet. Ask the user "did you mean [item]?" and WAIT.
           Their "yes" is NOT a confirmation by itself - call this tool
           AGAIN with the corrected name on that turn (that call is what
           actually saves it) before proceeding.
    {"matched": false, "ambiguous": true, "candidates": [...]}
        -> multiple similarly-close matches - show each candidate's name
           and ask the user to pick one; nothing was saved.
    {"matched": false, "ambiguous": false}
        -> no match at all.
    {"status": "list", "items": [...]}
        -> list mode result (user_input was empty).
    {"status": "not_configured"} / {"status": "error"}"""

    entity_type = (entity_type or "").strip().lower()
    if entity_type not in ("doctor", "branch"):
        return {"matched": False, "ambiguous": False, "status": "error"}

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("match_entity_for_booking called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"matched": False, "ambiguous": False, "status": "not_configured"}

    if entity_type == "doctor":
        branch_filter = [session["branch_id"]] if session.get("branch_id") else None
        result = api.get_doctors(base_url, branch_ids=branch_filter, page_size=200)
        name_keys = ["formatedName", "altName", "name"]
    else:
        result = api.get_branches(base_url, page_size=200)
        name_keys = ["name", "altName", "formatedName", "cityName"]

    if not result["success"]:
        logger.error("match_entity_for_booking API call failed: entity_type=%s status_code=%s error=%s", entity_type, result.get("status_code"), result.get("error"))
        return {"matched": False, "ambiguous": False, "status": "error"}

    items = (result["data"] or {}).get("items", [])

    def _shape(i):
        if entity_type == "doctor":
            return {
                "id": i.get("id"),
                "formatedName": i.get("formatedName") or i.get("name"),
                "altName": i.get("altName"),
                "degreeName": i.get("degreeName"),
                "specialtyName": i.get("specialtyName"),
                "branchId": i.get("branchId"),
                "branchName": i.get("branchName"),
            }
        return {
            "id": i.get("id"),
            "name": i.get("name") or i.get("formatedName"),
            "altName": i.get("altName"),
            "address": i.get("address"),
            "cityName": i.get("cityName"),
        }

    shaped_items = [_shape(i) for i in items]

    if not user_input or not user_input.strip():
        session["last_list"] = {"entity_type": entity_type, "items": shaped_items}
        if not shaped_items:
            return {"matched": False, "ambiguous": False, "status": "not_matched"}
        return {"status": "list", "items": shaped_items}

    # Bare number -> position in the list most recently shown for THIS entity_type
    stripped_input = user_input.strip()
    if stripped_input.isdigit() and session.get("last_list") and session["last_list"]["entity_type"] == entity_type:
        position = int(stripped_input)
        list_items = session["last_list"]["items"]
        if 1 <= position <= len(list_items):
            chosen_id = list_items[position - 1]["id"]
            chosen_raw = next((i for i in items if i.get("id") == chosen_id), None)
            if chosen_raw:
                shaped = _shape(chosen_raw)
                session[f"{entity_type}_id"] = shaped["id"]
                return {"matched": True, "needsConfirmation": False, "item": shaped}

    match_result = _fuzzy_match(user_input, items, name_keys)

    if match_result["result"] == "not_matched":
        return {"matched": False, "ambiguous": False}

    if match_result["result"] == "ambiguous":
        return {"matched": False, "ambiguous": True, "candidates": [_shape(i) for i in match_result["items"]]}

    # matched - decide confidence: high score (exact/unique) auto-confirms
    # and saves to session; lower score is a likely typo needing "did you
    # mean X?" confirmation before anything is saved.
    shaped = _shape(match_result["item"])
    needs_confirmation = match_result["score"] < 0.95

    if not needs_confirmation:
        session[f"{entity_type}_id"] = shaped["id"]

    return {"matched": True, "needsConfirmation": needs_confirmation, "item": shaped}


@tool
def get_doctor_fees(state: Annotated[AgentState, InjectedState]) -> dict:
    """Get the currently-confirmed doctor's published services and
    prices for a NEW BOOKING. Reads the doctor from the booking session
    automatically - you never pass an ID. A doctor MUST already be
    confirmed (via `match_entity_for_booking`, needsConfirmation=false)
    before calling this - if none is confirmed yet, this returns
    {"status": "no_doctor_confirmed"} and you should ask which doctor
    they're asking about first.

    IMPORTANT: fees are PRIVATE BY DEFAULT - only call this when the
    user EXPLICITLY asks about price/cost/fee. Never mention a fee
    proactively, and never quote one from schedule/slot data instead of
    this tool. Returns:
    {"status": "found", "fees": [{"service": ..., "price": ...}, ...]}
    {"status": "no_doctor_confirmed"}
    {"status": "not_found"}  # doctor has no published services
    {"status": "not_configured"} / {"status": "error"}"""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")

    if not doctor_id:
        return {"status": "no_doctor_confirmed"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("get_doctor_fees called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    result = api.get_doctor_fees(base_url, doctor_ids=[doctor_id])

    if not result["success"]:
        logger.error("get_doctor_fees API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    fees = [{"service": i.get("serviceName"), "price": i.get("price")} for i in items]
    return {"status": "found", "fees": fees}


@tool
def get_patient_info(state: Annotated[AgentState, InjectedState], mobile_number: str) -> dict:
    """Look up whether a patient is already registered by phone number,
    for a NEW BOOKING - to avoid re-asking for their name/email if
    they've booked before. Returns:
    {"status": "found", "patientFullName": ..., "mobileNumber": ..., "email": ...}
    {"status": "not_found"}  # not registered - collect name/email fresh
    {"status": "not_configured"} / {"status": "error"}"""

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("get_patient_info called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    result = api.get_patient_info(base_url, mobile_number)

    if not result["success"]:
        logger.error("get_patient_info API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    data = result["data"] or {}
    items = data.get("items", [])
    if not items or not data.get("totalCount"):
        return {"status": "not_found"}

    item = items[0]
    return {
        "status": "found",
        "patientFullName": item.get("patientFullName"),
        "mobileNumber": item.get("mobileNumber"),
        "email": item.get("email"),
    }


@tool
def resolve_available_day(
    state: Annotated[AgentState, InjectedState],
    weekday_name: str,
    after_date: str = "",
) -> dict:
    """For a NEW BOOKING: find the NEAREST date of a given weekday that
    the currently-confirmed doctor (and branch, if also confirmed)
    ACTUALLY has a real, non-booked slot available - not just any
    calendar date matching that weekday. Reads doctor_id/branch_id from
    the booking session automatically - both must already be confirmed
    via `match_entity_for_booking` first, or this returns an error
    telling you which is missing.

    NEVER compute or guess a date yourself for a new booking - always
    call this. `after_date` (format "YYYY-MM-DD"), if given, finds the
    next occurrence STRICTLY AFTER that date - use this for "next
    Thursday"/"الخميس اللي بعده" relative to one already discussed, or
    to retry after a day turned out fully booked.
    Returns:
    {"status": "found", "date": "YYYY-MM-DD", "weekday_name": "Thursday", "from_date": ..., "to_date": ...}
    {"status": "not_found"}  # no available slot for that weekday within the booking window
    {"status": "missing_doctor"} / {"status": "missing_branch"}
    {"status": "not_configured"} / {"status": "error"}"""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")
    branch_id = session.get("branch_id")

    if not doctor_id:
        return {"status": "missing_doctor"}
    if not branch_id:
        return {"status": "missing_branch"}

    key = (weekday_name or "").strip().lower()
    target_weekday = _WEEKDAY_NAMES.get(key)
    if target_weekday is None:
        logger.warning("resolve_available_day: unrecognized weekday_name=%r", weekday_name)
        return {"status": "error"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("resolve_available_day called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    now = datetime.now(tz)
    horizon_days = 42  # matches the confirmed production booking window
    from_date = now.isoformat()
    to_date = (now + timedelta(days=horizon_days)).isoformat()

    result = api.get_doctor_schedule_slots(
        base_url, doctor_ids=[doctor_id], branch_ids=[branch_id],
        from_date=from_date, to_date=to_date, is_booked=False, page_size=1000,
    )

    if not result["success"]:
        logger.error("resolve_available_day API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    logger.info(
        "resolve_available_day: doctor_id=%s branch_id=%s weekday=%s after_date=%r from_date=%s to_date=%s api_returned=%d",
        doctor_id, branch_id, weekday_name, after_date, from_date, to_date, len(items),
    )

    lead_time = now + timedelta(hours=12)  # 12h minimum advance booking lead, matches production
    after_dt = None
    if after_date:
        try:
            after_dt = date.fromisoformat(after_date.strip())
        except ValueError:
            after_dt = None

    candidates = []
    for item in items:
        if item.get("isBooked"):
            continue
        slot_start_local = to_riyadh(item.get("slotStart"), timezone_name)
        if not slot_start_local:
            continue
        try:
            dt = datetime.fromisoformat(slot_start_local)
        except ValueError:
            continue
        if dt <= lead_time:
            continue
        if dt.weekday() != target_weekday:
            continue
        if after_dt and dt.date() <= after_dt:
            continue
        candidates.append(dt)

    if not candidates:
        logger.info(
            "resolve_available_day: not_found - %d raw items, none matched (weekday=%s, lead_time=%s, after_date=%s). Sample raw items: %s",
            len(items), weekday_name, lead_time.isoformat(), after_dt, items[:3],
        )
        return {"status": "not_found"}

    candidates.sort()
    chosen_dt = candidates[0]
    chosen_date = chosen_dt.date()
    english_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][target_weekday]
    logger.info("resolve_available_day: found date=%s (weekday=%s) from %d candidate(s)", chosen_date.isoformat(), english_name, len(candidates))

    day_start = datetime.combine(chosen_date, datetime.min.time(), tzinfo=chosen_dt.tzinfo)
    day_end = datetime.combine(chosen_date, datetime.max.time().replace(microsecond=0), tzinfo=chosen_dt.tzinfo)

    return {
        "status": "found",
        "date": chosen_date.isoformat(),
        "weekday_name": english_name,
        "from_date": day_start.isoformat(),
        "to_date": day_end.isoformat(),
    }


@tool
def create_new_booking(
    state: Annotated[AgentState, InjectedState],
    slot_start: str,
    slot_end: str,
    patient_full_name: str,
    mobile_number: str,
    email: str = "",
) -> dict:
    """Create a brand new appointment booking. Reads the confirmed
    doctor_id/branch_id from the booking session automatically - you
    never pass an ID. `slot_start`/`slot_end` MUST be the EXACT values
    from a `resolve_available_day` + slot-lookup step in THIS
    conversation - never modified, recomputed, or invented.

    CRITICAL SAFETY CHECK (always performed automatically, you don't
    need to do anything extra): before creating the booking, this tool
    RE-VERIFIES the exact requested slot is still genuinely available
    right now (someone else may have booked it in the meantime) - this
    is not optional and cannot be skipped.

    A doctor AND branch must both already be confirmed (via
    `match_entity_for_booking`) before calling this. Returns:
    {"status": "success", "booking_ref": "GBN-..."}
    {"status": "slot_unavailable"}  # the requested slot is no longer free - tell the user and offer to pick again
    {"status": "missing_doctor"} / {"status": "missing_branch"}
    {"status": "not_configured"} / {"status": "error"}"""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")
    branch_id = session.get("branch_id")

    if not doctor_id:
        return {"status": "missing_doctor"}
    if not branch_id:
        return {"status": "missing_branch"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("create_new_booking called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    # Re-verify: query the EXACT requested slot's own narrow time window
    # and confirm it's still isBooked=false right now, immediately before
    # creating the booking - someone else may have taken it since it was
    # first shown to the user. Never skip this and never trust an older
    # lookup from earlier in the conversation.
    #
    # IMPORTANT: query the FULL DAY containing the slot, not just the
    # slot's own narrow start/end window - a too-narrow range risks the
    # API's own date-range boundary filtering excluding the exact slot
    # (e.g. an inclusive/exclusive edge mismatch), even though it's a
    # real, bookable slot (confirmed suspicious in production: a slot
    # that was successfully booked via the website was reported
    # "unavailable" by this exact check). The precise timestamp match
    # below already correctly isolates the one exact slot regardless of
    # how many others come back in a wider window.
    try:
        requested_start_dt = datetime.fromisoformat(slot_start)
        day_start = requested_start_dt.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        day_end = requested_start_dt.replace(hour=23, minute=59, second=59, microsecond=0).isoformat()
    except ValueError:
        logger.warning("create_new_booking: unparsable slot_start=%r", slot_start)
        return {"status": "error"}

    slots_result = api.get_doctor_schedule_slots(
        base_url, doctor_ids=[doctor_id], branch_ids=[branch_id],
        from_date=day_start, to_date=day_end, is_booked=False, page_size=200,
    )

    if not slots_result["success"]:
        logger.error("create_new_booking: re-verification API call failed: status_code=%s error=%s", slots_result.get("status_code"), slots_result.get("error"))
        return {"status": "error"}

    try:
        requested_ms = requested_start_dt.timestamp()
    except ValueError:
        logger.warning("create_new_booking: unparsable slot_start=%r", slot_start)
        return {"status": "error"}

    raw_items = (slots_result["data"] or {}).get("items", [])
    logger.info(
        "create_new_booking: re-verification doctor_id=%s branch_id=%s day_range=[%s, %s] requested_slot_start=%s api_returned=%d",
        doctor_id, branch_id, day_start, day_end, slot_start, len(raw_items),
    )

    matched_slot = None
    for item in raw_items:
        if item.get("isBooked"):
            continue
        try:
            item_ms = datetime.fromisoformat(item["slotStart"].replace("Z", "+00:00")).timestamp()
        except (ValueError, KeyError, AttributeError):
            continue
        if abs(item_ms - requested_ms) < 1:  # same instant
            matched_slot = item
            break

    if not matched_slot:
        logger.warning(
            "create_new_booking: requested slot %s not found or already booked (doctor_id=%s branch_id=%s). Raw slotStarts returned: %s",
            slot_start, doctor_id, branch_id, [i.get("slotStart") for i in raw_items][:20],
        )
        return {"status": "slot_unavailable"}

    result = api.create_booking(
        base_url,
        patient_full_name=patient_full_name,
        mobile_number=mobile_number,
        branch_id=matched_slot.get("branchId") or branch_id,
        doctor_id=matched_slot.get("doctorId") or doctor_id,
        service_id=matched_slot.get("serviceId"),
        service_price=matched_slot.get("servicePrice"),
        booking_time_from=slot_start,
        booking_time_to=slot_end,
        specialty_id=matched_slot.get("specialtyId"),
        doctor_schedule_id=matched_slot.get("scheduleId"),
        space_id=matched_slot.get("spaceId"),
        email=email,
    )

    if not result["success"]:
        logger.error("create_new_booking API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    new_booking_id = result["data"]
    booking_ref = None

    if new_booking_id:
        lookup_result = api.get_booking_by_id(base_url, new_booking_id)
        if lookup_result["success"]:
            booking_ref = (lookup_result["data"] or {}).get("bookingRefNum")

    # Booking complete - clear the session so a subsequent NEW booking
    # in the same conversation starts clean, matching the confirmed
    # production behavior (session auto-cleans on success).
    _BOOKING_SESSIONS.pop(session_id, None)

    return {"status": "success", "booking_ref": booking_ref}


@tool
def get_doctor_schedule_for_booking(
    state: Annotated[AgentState, InjectedState],
    target_date: str = "",
) -> dict:
    """For a NEW BOOKING: get the confirmed doctor's general recurring
    schedule (which weekdays they work, daily hours, and which branch
    each applies to). Reads doctor_id from the booking session
    automatically - a doctor must already be confirmed via
    `match_entity_for_booking` first.

    If a branch is ALSO already confirmed in the session, the schedule
    is automatically narrowed to that branch only. If no branch is
    confirmed yet, the schedule spans EVERY branch the doctor works at -
    group your reply by branch in that case (see the booking flow's own
    display instructions).

    `target_date` (format "YYYY-MM-DD"), if given, filters to only
    currently-effective schedule rows on that date; defaults to today.
    Returns:
    {"status": "found", "schedules": [{"recurringDaysNames": [...], "fromDateTime": ..., "toDateTime": ..., "branchName": ..., "doctorName": ...}, ...]}
    {"status": "not_found"} / {"status": "missing_doctor"}
    {"status": "not_configured"} / {"status": "error"}"""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")

    if not doctor_id:
        return {"status": "missing_doctor"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("get_doctor_schedule_for_booking called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    if target_date:
        effective_date = target_date
    else:
        try:
            effective_date = datetime.now(ZoneInfo(timezone_name)).date().isoformat()
        except Exception:
            effective_date = None

    branch_id = session.get("branch_id")
    result = api.get_doctor_schedule(
        base_url, doctor_ids=[doctor_id],
        branch_ids=[branch_id] if branch_id else None,
        effective_date=effective_date,
    )

    if not result["success"]:
        logger.error("get_doctor_schedule_for_booking API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    if not items:
        return {"status": "not_found"}

    schedules = [
        {
            "recurringDaysNames": item.get("recurringDaysNames"),
            "fromDateTime": to_riyadh(item.get("fromDateTime"), timezone_name),
            "toDateTime": to_riyadh(item.get("toDateTime"), timezone_name),
            "branchName": item.get("branchName"),
            "branchId": item.get("branchId"),
            "doctorName": item.get("doctorName"),
        }
        for item in items
    ]

    return {"status": "found", "schedules": schedules}


@tool
def get_available_slots_for_booking(
    state: Annotated[AgentState, InjectedState],
    from_date: str,
    to_date: str,
) -> dict:
    """For a NEW BOOKING: get the confirmed doctor's ACTUAL open time
    slots (not just working hours) within [from_date, to_date] - both
    ISO format, e.g. "2026-05-01T09:00:00+03:00". Typically called with
    the exact from_date/to_date returned by `resolve_available_day`.
    Reads doctor_id AND branch_id from the booking session automatically
    - both must already be confirmed. Only genuinely available (not
    already booked) slots are returned. Returns:
    {"status": "found", "slots": [{"slotStart": ..., "slotEnd": ..., "date_display": ..., "time_display": ..., "serviceName": ..., "servicePrice": ...}, ...]}
    {"status": "not_found"}  # no open slots in this range
    {"status": "missing_doctor"} / {"status": "missing_branch"}
    {"status": "not_configured"} / {"status": "error"}"""

    session_id = state.get("session_id")
    session = _get_booking_session(session_id)
    doctor_id = session.get("doctor_id")
    branch_id = session.get("branch_id")

    if not doctor_id:
        return {"status": "missing_doctor"}
    if not branch_id:
        return {"status": "missing_branch"}

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("get_available_slots_for_booking called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    # Safety net: swap an inverted range - confirmed real production bug
    # for the same underlying endpoint (see get_available_reschedule_slots).
    try:
        if from_date and to_date and datetime.fromisoformat(from_date) > datetime.fromisoformat(to_date):
            logger.warning("get_available_slots_for_booking: from_date=%r was AFTER to_date=%r - swapping them", from_date, to_date)
            from_date, to_date = to_date, from_date
    except ValueError:
        pass

    result = api.get_doctor_schedule_slots(
        base_url, doctor_ids=[doctor_id], branch_ids=[branch_id],
        from_date=from_date, to_date=to_date, is_booked=False, page_size=200,
    )

    if not result["success"]:
        logger.error("get_available_slots_for_booking API call failed: status_code=%s error=%s", result.get("status_code"), result.get("error"))
        return {"status": "error"}

    items = (result["data"] or {}).get("items", [])
    logger.info(
        "get_available_slots_for_booking: doctor_id=%s branch_id=%s from_date=%s to_date=%s api_returned=%d",
        doctor_id, branch_id, from_date, to_date, len(items),
    )
    if not items:
        logger.info("get_available_slots_for_booking: not_found - API returned zero items for this range")
        return {"status": "not_found"}

    items_before_filter = len(items)
    items = [i for i in items if i.get("isBooked") is not True]
    if not items:
        logger.info("get_available_slots_for_booking: not_found - all %d item(s) were isBooked=True", items_before_filter)
        return {"status": "not_found"}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    slots = []
    for item in items:
        slot_start = to_riyadh(item.get("slotStart"), timezone_name)
        slot_end = to_riyadh(item.get("slotEnd"), timezone_name)
        slots.append({
            "slotStart": slot_start,
            "slotEnd": slot_end,
            "date_display": _display_date(slot_start),
            "time_display": _display_time_12h(slot_start),
            "serviceId": item.get("serviceId"),
            "serviceName": item.get("serviceName"),
            "servicePrice": item.get("servicePrice"),
        })

    # Exclude past slots, dedupe, sort, cap - same safeguards as the
    # reschedule flow's equivalent (all confirmed real production issues).
    try:
        now_local = datetime.now(ZoneInfo(timezone_name))
        slots = [s for s in slots if s["slotStart"] and datetime.fromisoformat(s["slotStart"]) > now_local]
    except Exception:
        logger.exception("get_available_slots_for_booking: failed to filter past slots, showing all")

    if not slots:
        logger.info("get_available_slots_for_booking: not_found - all slots were in the past relative to now")
        return {"status": "not_found"}

    slots.sort(key=lambda s: s["slotStart"] or "")

    seen_starts = set()
    deduped = []
    for s in slots:
        if s["slotStart"] in seen_starts:
            continue
        seen_starts.add(s["slotStart"])
        deduped.append(s)
    slots = deduped

    MAX_SLOTS_TO_SHOW = 20
    if len(slots) > MAX_SLOTS_TO_SHOW:
        slots = slots[:MAX_SLOTS_TO_SHOW]

    return {"status": "found", "slots": slots}


@tool
def find_best_doctor_in_specialty(
    state: Annotated[AgentState, InjectedState],
    specialty_id: str,
    criteria: str = "soonest",
) -> dict:
    """Among ALL doctors in a given specialty, find either the one with
    the SOONEST available appointment, or the one with the CHEAPEST
    fee - use this when the user says they don't care which specific
    doctor they see and just want the earliest opening, or explicitly
    ask for the cheapest option (e.g. after seeing a list of doctors
    for a specialty and asking "who's soonest?" or "who's cheapest?").
    `specialty_id` must come from `list_specialties`'s own response -
    never invented. `criteria`: "soonest" (default) or "cheapest".
    Returns:
    {"status": "found", "doctor": {...}, "slot": {...}}  # for "soonest" - present the doctor and when
    {"status": "found", "doctor": {...}, "price": ..., "service": ...}  # for "cheapest"
    {"status": "not_found"}  # no doctors in this specialty currently qualify
    {"status": "not_configured"} / {"status": "error"}"""

    criteria = (criteria or "soonest").strip().lower()
    if criteria not in ("soonest", "cheapest"):
        criteria = "soonest"

    base_url = _doctors_base_url(state)
    if not base_url:
        logger.warning("find_best_doctor_in_specialty called but no doctors_base_url is configured for client_id=%s", state.get("client_id"))
        return {"status": "not_configured"}

    doctors_result = api.get_doctors(base_url, specialty_ids=[specialty_id], page_size=200)
    if not doctors_result["success"]:
        logger.error("find_best_doctor_in_specialty: get_doctors failed: status_code=%s error=%s", doctors_result.get("status_code"), doctors_result.get("error"))
        return {"status": "error"}

    doctors = [d for d in (doctors_result["data"] or {}).get("items", []) if d.get("hasSlots") is not False]
    if not doctors:
        return {"status": "not_found"}

    doctor_ids = [d.get("id") for d in doctors if d.get("id")]
    doctors_by_id = {d.get("id"): d for d in doctors}

    timezone_name = (state.get("templates") or {}).get("_timezone", DEFAULT_TIMEZONE)
    try:
        tz = ZoneInfo(timezone_name)
    except Exception:
        tz = ZoneInfo(DEFAULT_TIMEZONE)

    if criteria == "soonest":
        now = datetime.now(tz)
        horizon = now + timedelta(days=30)

        slots_result = api.get_doctor_schedule_slots(
            base_url, doctor_ids=doctor_ids,
            from_date=now.isoformat(), to_date=horizon.isoformat(),
            is_booked=False, page_size=1000,
        )
        if not slots_result["success"]:
            logger.error("find_best_doctor_in_specialty: get_doctor_schedule_slots failed: status_code=%s error=%s", slots_result.get("status_code"), slots_result.get("error"))
            return {"status": "error"}

        best = None
        for item in (slots_result["data"] or {}).get("items", []):
            if item.get("isBooked"):
                continue
            slot_start = to_riyadh(item.get("slotStart"), timezone_name)
            if not slot_start:
                continue
            try:
                dt = datetime.fromisoformat(slot_start)
            except ValueError:
                continue
            if dt <= now:
                continue
            if best is None or dt < best[0]:
                best = (dt, item)

        if not best:
            return {"status": "not_found"}

        dt, item = best
        doctor = doctors_by_id.get(item.get("doctorId"), {})

        return {
            "status": "found",
            "doctor": {
                "id": item.get("doctorId"),
                "formatedName": doctor.get("formatedName") or item.get("doctorName"),
                "degreeName": doctor.get("degreeName"),
            },
            "slot": {
                "slotStart": dt.isoformat(),
                "slotEnd": to_riyadh(item.get("slotEnd"), timezone_name),
                "date_display": _display_date(dt.isoformat()),
                "time_display": _display_time_12h(dt.isoformat()),
                "branchId": item.get("branchId"),
                "branchName": item.get("branchName"),
            },
        }

    # criteria == "cheapest" - queried per-doctor (confirmed request
    # shape only supports one doctor's fees at a time reliably; the
    # roster for a single specialty is small enough that this is fine).
    best_price = None
    best_doctor_id = None
    best_service = None

    for doctor_id in doctor_ids:
        fees_result = api.get_doctor_fees(base_url, doctor_ids=[doctor_id])
        if not fees_result["success"]:
            continue
        for item in (fees_result["data"] or {}).get("items", []):
            price = item.get("price")
            if price is None:
                continue
            if best_price is None or price < best_price:
                best_price = price
                best_doctor_id = doctor_id
                best_service = item.get("serviceName")

    if best_doctor_id is None:
        return {"status": "not_found"}

    doctor = doctors_by_id.get(best_doctor_id, {})
    return {
        "status": "found",
        "doctor": {
            "id": best_doctor_id,
            "formatedName": doctor.get("formatedName"),
            "degreeName": doctor.get("degreeName"),
        },
        "price": best_price,
        "service": best_service,
    }


ALL_TOOLS = [
    validate_phone_format,
    compare_phone,
    lookup_appointment,
    check_booking_status,
    cancel_appointment,
    send_otp,
    verify_otp,
    list_specialties,
    find_available_doctors,
    get_next_weekday_date,
    get_doctor_schedule,
    get_available_reschedule_slots,
    reschedule_appointment,
    answer_hospital_faq,
    match_entity_info,
    reset_booking_session,
    match_entity_for_booking,
    get_doctor_fees,
    get_patient_info,
    resolve_available_day,
    create_new_booking,
    get_doctor_schedule_for_booking,
    get_available_slots_for_booking,
    find_best_doctor_in_specialty,
]
