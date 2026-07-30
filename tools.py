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
    """Phone-path-only filter, applied when looking up bookings by phone
    number so the user can choose which one to cancel.

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

    The reference-number lookup path does NOT apply this filter at all -
    that asymmetry is unchanged, carried over from the original business
    logic.

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

    now = datetime.utcnow()

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
                dt = datetime.fromisoformat(raw_from.replace("Z", ""))
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

    if phone:
        # Phone path applies the active-only filter; ref path does not -
        # exact same asymmetry as the original business logic.
        items = _filter_active(items)
        if not items:
            return {"status": "not_found"}

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
    language: str = "en",
) -> dict:
    """Get the GENERAL RECURRING weekly schedule of the doctor on a given
    booking - which weekdays they work, their daily start/end times, and
    the date range this schedule is valid for. Call this BEFORE offering
    to reschedule, to know which days of the week are even worth
    checking - this does NOT return specific open time slots (use
    `get_available_reschedule_slots` for that once you've picked a
    target date). Returns:
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

    result = api.get_doctor_schedule(base_url, doctor_ids=[resolved["doctor_id"]])

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

    # Always chronological - the API's own return order was observed to
    # be scrambled in production (slots came back neither ascending nor
    # descending), and relying on the LLM to re-sort dozens of items
    # correctly by eye is not realistic. Sort here, once, in code.
    slots.sort(key=lambda s: s["slotStart"] or "")

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
]
