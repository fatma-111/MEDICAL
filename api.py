"""
Raw HTTP client layer.

Every external HTTP call in the system lives here, one function per call,
mirroring the n8n HTTP Request nodes 1:1:

  - GuestBookings/GetList (by ref)    <- f_lookup_appointment.json "HTTP Request"
                                         f_cancel_appointment.json "HTTP Request"
  - GuestBookings/GetList (by phone)  <- f_lookup_appointment.json "HTTP Request2"
                                         f_cancel_appointment.json "HTTP Request2"
  - GuestBookings/Cancel/{id}         <- f_cancel_appointment.json "HTTP Request1"/"HTTP Request3"/"HTTP Request4"
  - Authentica send-otp / verify-otp  <- langchain_cancellation.json "send_otp5"/"verify_otp5"

No business logic (filtering, selection, formatting) lives here - that's
tools.py's job. Every function catches network failures itself and
returns a structured result rather than raising, so graph nodes never
need a try/except around a tool call.
"""

import logging
from typing import Optional

import requests

from config import (
    AUTHENTICA_API_KEY,
    AUTHENTICA_BASE_URL,
    AUTHENTICA_FALLBACK_EMAIL,
    AUTHENTICA_TEMPLATE_ID,
    CLIENT_ID_HEADER,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)


# ==========================================================
# Result helper
# ==========================================================

def _result(success: bool, status_code: Optional[int] = None, data=None, error: Optional[str] = None) -> dict:
    return {"success": success, "status_code": status_code, "data": data, "error": error}


def _headers(client_id: Optional[str] = None, language: Optional[str] = None) -> dict:
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    if client_id:
        headers[CLIENT_ID_HEADER] = client_id
    if language:
        headers["accept-language"] = language
    return headers


# ==========================================================
# Guest Bookings API
# ==========================================================

def get_bookings_by_ref(base_url: str, ref_number: str, language: Optional[str] = None, client_id: Optional[str] = None) -> dict:
    """POST {base_url}/api/GuestBookings/GetList with bookingRefNum.

    Mirrors f_lookup_appointment.json "HTTP Request" / f_cancel_appointment.json "HTTP Request".
    """

    url = f"{base_url}/api/GuestBookings/GetList"
    payload = {"bookingRefNum": ref_number}

    return _post_bookings(url, payload, language, client_id)


def get_bookings_by_phone(
    base_url: str,
    phone: str,
    language: Optional[str] = None,
    client_id: Optional[str] = None,
    page_size: int = 1000,
    status_list: Optional[list] = None,
) -> dict:
    """POST {base_url}/api/GuestBookings/GetList with mobileNumber + pageSize.

    Mirrors f_lookup_appointment.json "HTTP Request2" (pageSize: 1000).

    `status_list`, when given, is sent as the API's own "statusList"
    filter field (confirmed from the Booking API's documented request
    schema) - e.g. [1, 2] for New+Confirmed only. This lets the server
    do the active-status filtering directly. tools.py's own client-side
    filtering (_filter_active) still runs afterward as a second,
    defense-in-depth layer regardless of whether this is used.
    """

    url = f"{base_url}/api/GuestBookings/GetList"
    payload = {"mobileNumber": phone, "pageSize": page_size}

    if status_list:
        payload["statusList"] = status_list

    return _post_bookings(url, payload, language, client_id)


def _post_bookings(url: str, payload: dict, language: Optional[str], client_id: Optional[str]) -> dict:
    logger.debug("POST %s payload=%s", url, payload)

    try:
        response = requests.post(
            url,
            json=payload,
            headers=_headers(client_id=client_id, language=language),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        logger.warning("Booking lookup timed out: %s", url)
        return _result(False, error="timeout")
    except requests.RequestException as exc:
        logger.exception("Booking lookup request failed: %s", url)
        return _result(False, error=str(exc))

    if response.status_code >= 500:
        return _result(False, response.status_code, error="server_error")

    if response.status_code >= 400:
        return _result(False, response.status_code, error="validation_error")

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body.get("data", {}))


def cancel_booking_by_guid(base_url: str, booking_guid: str, client_id: Optional[str] = None) -> dict:
    """PUT {base_url}/api/GuestBookings/Cancel/{booking_guid}.

    Mirrors f_cancel_appointment.json "HTTP Request1"/"HTTP Request3"
    (onError: continueErrorOutput -> here, a structured failure result
    instead of a raised exception achieves the same thing).
    """

    url = f"{base_url}/api/GuestBookings/Cancel/{booking_guid}"

    logger.debug("PUT %s", url)

    try:
        response = requests.put(
            url,
            headers=_headers(client_id=client_id),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        logger.warning("Cancel request timed out: %s", url)
        return _result(False, error="timeout")
    except requests.RequestException as exc:
        logger.exception("Cancel request failed: %s", url)
        return _result(False, error=str(exc))

    if response.status_code >= 500:
        return _result(False, response.status_code, error="server_error")

    if response.status_code >= 400:
        return _result(False, response.status_code, error="validation_error")

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body)


# ==========================================================
# Authentica OTP API (real provider - langchain_cancellation.json
# "send_otp5" / "verify_otp5"). Only used when config.OTP_PROVIDER ==
# "authentica"; see services in tools.py for the dummy alternative.
# ==========================================================

def authentica_send_otp(phone: str) -> dict:
    url = f"{AUTHENTICA_BASE_URL}/send-otp"

    payload = {
        "method": "sms",
        "template_id": AUTHENTICA_TEMPLATE_ID,
        "fallback_email": AUTHENTICA_FALLBACK_EMAIL,
        "phone": phone,
    }
    headers = {"Accept": "application/json", "X-Authorization": AUTHENTICA_API_KEY}

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.Timeout:
        return _result(False, error="timeout")
    except requests.RequestException as exc:
        logger.exception("Authentica send_otp failed")
        return _result(False, error=str(exc))

    if response.status_code >= 400:
        return _result(False, response.status_code, error="send_otp_failed")

    try:
        body = response.json()
    except ValueError:
        body = {}

    return _result(True, response.status_code, data=body)


def authentica_verify_otp(phone: str, otp: str, email: str = "") -> dict:
    url = f"{AUTHENTICA_BASE_URL}/verify-otp"

    payload = {"otp": otp, "email": email, "phone": phone}
    headers = {"Accept": "application/json", "X-Authorization": AUTHENTICA_API_KEY}

    try:
        response = requests.post(url, headers=headers, data=payload, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.Timeout:
        return _result(False, error="timeout")
    except requests.RequestException as exc:
        logger.exception("Authentica verify_otp failed")
        return _result(False, error=str(exc))

    if response.status_code >= 400:
        return _result(False, response.status_code, error="verify_otp_failed")

    try:
        body = response.json()
    except ValueError:
        body = {}

    verified = bool(body.get("isSuccess") or body.get("success") or body.get("verified"))

    return _result(verified, response.status_code, data=body)


# ==========================================================
# Doctors / Specialties API (Medical Concierge feature)
# ==========================================================
#
# Separate service from GuestBookings, confirmed on a different port
# (1102 vs 1101). Response shape (confirmed directly from the API's own
# Swagger "Execute" output): {"data": {"items": [...], ...},
# "statusCode": 200, "isSuccess": true, "messages": [...]} - handled the
# same way _post_bookings already handles GuestBookings' identical
# response envelope.

def _post_json(url: str, payload: dict, client_id: Optional[str] = None) -> dict:
    """Generic POST + envelope handling, shared by get_specialties/
    get_doctors. Mirrors _post_bookings' error handling exactly
    (timeout/5xx/4xx/empty/invalid JSON/isSuccess check), kept as a
    separate function so GuestBookings' own _post_bookings is untouched."""

    logger.debug("POST %s payload=%s", url, payload)

    try:
        response = requests.post(
            url,
            json=payload,
            headers=_headers(client_id=client_id),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        logger.warning("Request timed out: %s", url)
        return _result(False, error="timeout")
    except requests.RequestException as exc:
        logger.exception("Request failed: %s", url)
        return _result(False, error=str(exc))

    if response.status_code >= 500:
        logger.error("Doctors/Specialties API server error: %s status=%s body=%s", url, response.status_code, response.text[:1000])
        return _result(False, response.status_code, error="server_error")

    if response.status_code >= 400:
        logger.error("Doctors/Specialties API validation error: %s status=%s body=%s", url, response.status_code, response.text[:1000])
        return _result(False, response.status_code, error="validation_error")

    try:
        body = response.json()
    except ValueError:
        return _result(False, response.status_code, error="invalid_json_response")

    if not body:
        return _result(False, response.status_code, error="empty_response")

    if not body.get("isSuccess"):
        return _result(False, response.status_code, data=body, error="api_reported_failure")

    return _result(True, response.status_code, data=body.get("data", {}))


def get_specialties(base_url: str, page_size: int = 200, client_id: Optional[str] = None) -> dict:
    """POST {base_url}/api/Specialties/GetSpecialtiesPagedList.

    Returns every specialty this clinic offers (scoped by base_url alone,
    confirmed directly - no separate organizationId/branchId needed)."""

    url = f"{base_url}/api/Specialties/GetSpecialtiesPagedList"
    payload = {"pageNumber": 0, "pageSize": page_size}

    return _post_json(url, payload, client_id=client_id)


def get_doctors(
    base_url: str,
    specialty_ids: Optional[list] = None,
    has_published_service: bool = True,
    has_service_schedule: bool = True,
    intersection_start: Optional[str] = None,
    intersection_end: Optional[str] = None,
    page_size: int = 200,
    client_id: Optional[str] = None,
) -> dict:
    """POST {base_url}/api/Doctors/GetList.

    `has_published_service`/`has_service_schedule`/`intersection_start`/
    `intersection_end` are REQUEST filter fields (confirmed directly from
    the API's own request schema) - they narrow results to doctors who
    are actually bookable with an available schedule intersecting the
    given time window. The response itself then includes `hasSlots` per
    doctor reflecting that same filter."""

    url = f"{base_url}/api/Doctors/GetList"
    payload = {
        "pageNumber": 0,
        "pageSize": page_size,
        "hasPublishedService": has_published_service,
        "hasServiceSchedule": has_service_schedule,
    }

    if specialty_ids:
        payload["specialtyIds"] = specialty_ids
    if intersection_start:
        payload["intersectionStart"] = intersection_start
    if intersection_end:
        payload["intersectionEnd"] = intersection_end

    return _post_json(url, payload, client_id=client_id)
