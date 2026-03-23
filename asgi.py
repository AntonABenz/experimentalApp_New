# asgi.py
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse, RedirectResponse, PlainTextResponse
from starlette.requests import Request

from otree.asgi import app as otree_app
from prolific_webhook import prolific_webhook_view, _find_participant_by_prolific_id
from start import Player as StartPlayer

# NEW
from otree.models import Session  # Django ORM
from img_desc import (
    Player as ImgDescPlayer,
    clean_str,
    free_slot_for_participant,
    get_cohort_snapshot_data,
    get_participant_prolific_id,
    get_participant_slot_rows,
    get_participant_status,
    mark_participant_active,
    mark_participant_complete_in_cohort,
    mark_participant_drop_out,
    mark_participant_finished,
    maybe_expand_prolific_for_participant,
    reset_this_app_for_participant,
    safe_int,
)

logger = logging.getLogger("benzapp.admin")
PROLIFIC_CAPTURE_COOKIE = "prolific_capture"
PROLIFIC_CAPTURE_MAX_AGE = 6 * 60 * 60

async def ping(request):
    return JSONResponse({"ok": True})


def _cookie_secret() -> bytes:
    return (os.environ.get("OTREE_SECRET_KEY") or "dev-secret").encode("utf-8")


def _clean_cookie_value(value) -> str:
    return (value or "").strip()


def _sign_cookie_payload(payload: dict) -> str:
    raw = json.dumps(payload or {}, separators=(",", ":"), sort_keys=True)
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
    signature = hmac.new(_cookie_secret(), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _load_signed_cookie(value: str) -> dict:
    value = _clean_cookie_value(value)
    if "." not in value:
        return {}

    encoded, signature = value.rsplit(".", 1)
    expected = hmac.new(_cookie_secret(), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    if not secrets.compare_digest(signature, expected):
        return {}

    try:
        padded = encoded + ("=" * (-len(encoded) % 4))
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        payload = json.loads(raw or "{}")
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def _extract_prolific_cookie_payload(request: Request) -> dict:
    return {
        "prolific_id": _clean_cookie_value(
            request.query_params.get("PROLIFIC_PID")
            or request.query_params.get("prolific_pid")
            or request.query_params.get("prolific_id")
            or request.query_params.get("participant_id")
            or request.query_params.get("participant_label")
        ),
        "study_id": _clean_cookie_value(
            request.query_params.get("STUDY_ID")
            or request.query_params.get("study_id")
        ),
        "session_id": _clean_cookie_value(
            request.query_params.get("SESSION_ID")
            or request.query_params.get("session_id")
        ),
    }


def _participant_code_from_path(path: str) -> str:
    parts = [p for p in (path or "").split("/") if p]
    if len(parts) >= 2 and parts[0] == "InitializeParticipant":
        return parts[1]
    if len(parts) >= 2 and parts[0] == "p":
        return parts[1]
    return ""


def _find_participant_by_code(participant_code: str):
    participant_code = _clean_cookie_value(participant_code)
    if not participant_code:
        return None

    # Do not use Participant.objects here. In this deployed oTree runtime that API
    # is not available and caused 500s on /InitializeParticipant for Prolific users.
    # Find the participant through concrete app Player models instead.
    for model in (StartPlayer, ImgDescPlayer):
        try:
            player = (
                model.objects
                .filter(participant__code=participant_code)
                .order_by("-id")
                .first()
            )
            if player and getattr(player, "participant", None):
                return player.participant
        except Exception:
            continue

    # Minimal fallback for admin repair/testing and for runtimes where the ORM
    # relation lookup above silently misses. Keep this as a fallback only so we
    # do not change the normal entry behavior more than necessary.
    for model in (StartPlayer, ImgDescPlayer):
        try:
            rows = list(model.filter())
        except Exception:
            continue
        for player in reversed(rows[-5000:]):
            participant = getattr(player, "participant", None)
            if participant and _clean_cookie_value(getattr(participant, "code", "")) == participant_code:
                return participant

    return None


def _apply_prolific_cookie_to_participant(participant, payload: dict) -> bool:
    prolific_id = _clean_cookie_value(payload.get("prolific_id"))
    study_id = _clean_cookie_value(payload.get("study_id"))
    session_id = _clean_cookie_value(payload.get("session_id"))

    if not prolific_id:
        return False

    changed = False
    if participant.vars.get("prolific_id") != prolific_id:
        participant.vars["prolific_id"] = prolific_id
        changed = True

    # Stable identifier semantics, do not repurpose:
    # participant.label is reserved here as a fallback copy of the Prolific PID.
    # It must not be reused for spreadsheet participant ids or status labels.
    if getattr(participant, "label", "") != prolific_id:
        participant.label = prolific_id
        changed = True

    if study_id and participant.vars.get("study_id") != study_id:
        participant.vars["study_id"] = study_id
        changed = True

    if session_id and participant.vars.get("prolific_session_id") != session_id:
        participant.vars["prolific_session_id"] = session_id
        changed = True

    current_status = _clean_cookie_value(participant.vars.get("participant_status")).lower()
    if current_status not in {"finished", "drop_out"}:
        participant.vars["participant_status"] = "active"
        changed = True

    if changed:
        participant.save()
        logger.info(
            "Prolific cookie captured: participant=%s prolific_id=%s study_id=%s session_id=%s",
            participant.code,
            prolific_id,
            study_id,
            session_id,
        )
    return True


class ProlificCaptureCookieMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        should_clear_cookie = False
        payload = _load_signed_cookie(request.cookies.get(PROLIFIC_CAPTURE_COOKIE, ""))
        request_payload = _extract_prolific_cookie_payload(request)
        participant_code = _participant_code_from_path(request.url.path)

        if payload:
            if participant_code:
                participant = _find_participant_by_code(participant_code)
                if participant and _apply_prolific_cookie_to_participant(participant, payload):
                    should_clear_cookie = True

        response = await call_next(request)

        if payload and not should_clear_cookie and participant_code:
            participant = _find_participant_by_code(participant_code)
            if participant and _apply_prolific_cookie_to_participant(participant, payload):
                should_clear_cookie = True

        if request_payload.get("prolific_id") and not payload:
            response.set_cookie(
                PROLIFIC_CAPTURE_COOKIE,
                _sign_cookie_payload(request_payload),
                max_age=PROLIFIC_CAPTURE_MAX_AGE,
                httponly=True,
                samesite="lax",
                path="/",
            )

        if should_clear_cookie:
            response.delete_cookie(PROLIFIC_CAPTURE_COOKIE, path="/")

        return response

async def entry(request):
    """
    Prolific landing endpoint.
    Redirects to /join/<session_code>?<original_querystring>
    IMPORTANT: You must create at least one session in admin (full_study) beforehand.
    """
    qs = str(request.url.query or "")
    suffix = f"?{qs}" if qs else ""

    # pick the latest real session for config_name='full_study'
    s = (
        Session.objects
        .filter(is_demo=False, config_name="full_study")
        .order_by("-id")
        .first()
    )

    if not s:
        return PlainTextResponse("No active session found. Create a 'full_study' session in admin.", status_code=500)

    response = RedirectResponse(f"/join/{s.code}{suffix}", status_code=302)

    payload = _extract_prolific_cookie_payload(request)
    if payload.get("prolific_id"):
        response.set_cookie(
            PROLIFIC_CAPTURE_COOKIE,
            _sign_cookie_payload(payload),
            max_age=PROLIFIC_CAPTURE_MAX_AGE,
            httponly=True,
            samesite="lax",
            path="/",
        )

    return response


def _repair_token_from_request(request: Request) -> str:
    return (
        request.query_params.get("token")
        or request.headers.get("X-Cohort-Repair-Token")
        or ""
    ).strip()


def _serialize_participant_state(participant):
    slot_rows = get_participant_slot_rows(participant.session, participant.code)
    exp_nums = sorted({int(row["exp_num"]) for row in slot_rows if int(row["exp_num"]) > 0})

    exp_from_vars = safe_int(participant.vars.get("exp_target"), 0)
    if exp_from_vars > 0 and exp_from_vars not in exp_nums:
        exp_nums.append(exp_from_vars)
        exp_nums = sorted(exp_nums)
    if not exp_nums:
        exp_nums = [1]

    return dict(
        participant_code=participant.code,
        prolific_id=get_participant_prolific_id(participant),
        participant_label=clean_str(getattr(participant, "label", "")),
        participant_status=get_participant_status(participant) or "",
        prolific_submission_status=clean_str(participant.vars.get("prolific_submission_status", "")),
        returned_from_prolific=bool(participant.vars.get("returned_from_prolific")),
        exp_target=safe_int(participant.vars.get("exp_target"), 0),
        local_slot=safe_int(participant.vars.get("local_slot"), 0),
        slot_rows=slot_rows,
        cohort_snapshots=[get_cohort_snapshot_data(participant.session, exp_num) for exp_num in exp_nums],
    )


def _find_participant_for_repair(participant_code: str, prolific_id: str):
    participant_code = clean_str(participant_code)
    prolific_id = clean_str(prolific_id)

    if participant_code:
        participant = _find_participant_by_code(participant_code)
        if participant:
            return participant

    if prolific_id:
        return _find_participant_by_prolific_id(prolific_id)

    return None


async def cohort_repair(request: Request):
    expected_token = (os.environ.get("COHORT_REPAIR_TOKEN") or "").strip()
    if not expected_token:
        return JSONResponse(
            {"ok": False, "error": "COHORT_REPAIR_TOKEN not configured"},
            status_code=503,
        )

    provided_token = _repair_token_from_request(request)
    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        return JSONResponse({"ok": False, "error": "invalid token"}, status_code=403)

    body = {}
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}

    action = clean_str(body.get("action") or request.query_params.get("action") or "status").lower()
    participant_code = clean_str(body.get("participant_code") or request.query_params.get("participant_code") or "")
    prolific_id = clean_str(body.get("prolific_id") or request.query_params.get("prolific_id") or "")

    if action not in {"status", "drop_out", "finished", "free_slot", "active"}:
        return JSONResponse(
            {"ok": False, "error": "action must be one of status, drop_out, finished, free_slot, active"},
            status_code=400,
        )

    participant = _find_participant_for_repair(participant_code, prolific_id)
    if not participant:
        return JSONResponse(
            {"ok": False, "error": "participant not found", "participant_code": participant_code, "prolific_id": prolific_id},
            status_code=404,
        )

    before = _serialize_participant_state(participant)
    note = "status only"

    if action == "drop_out":
        mark_participant_drop_out(participant)
        free_slot_for_participant(participant.session, participant.code)
        reset_this_app_for_participant(participant)
        mark_participant_complete_in_cohort(participant.session, participant.code, completed=False)
        participant.save()
        note = "participant marked drop_out, slot freed, app state reset"
    elif action == "finished":
        mark_participant_finished(participant)
        mark_participant_complete_in_cohort(participant.session, participant.code, completed=True)
        participant.save()
        maybe_expand_prolific_for_participant(participant)
        note = "participant marked finished and cohort completion updated"
    elif action == "free_slot":
        free_slot_for_participant(participant.session, participant.code)
        reset_this_app_for_participant(participant)
        mark_participant_complete_in_cohort(participant.session, participant.code, completed=False)
        participant.save()
        note = "slot freed and app state reset; participant status unchanged"
    elif action == "active":
        mark_participant_active(participant)
        participant.save()
        note = "participant marked active"

    after = _serialize_participant_state(participant)
    logger.info(
        "CohortRepair: action=%s participant=%s prolific_id=%s status_before=%s status_after=%s",
        action,
        participant.code,
        get_participant_prolific_id(participant),
        before.get("participant_status", ""),
        after.get("participant_status", ""),
    )

    return JSONResponse(
        {
            "ok": True,
            "action": action,
            "note": note,
            "before": before,
            "after": after,
        }
    )

app = Starlette(
    middleware=[
        Middleware(ProlificCaptureCookieMiddleware),
    ],
    routes=[
        Route("/prolific/ping/", ping, methods=["GET"]),
        Route("/prolific/webhook/", prolific_webhook_view, methods=["POST", "GET", "HEAD"]),
        Route("/admin/cohort-repair", cohort_repair, methods=["GET", "POST"]),
        Route("/admin/cohort-repair/", cohort_repair, methods=["GET", "POST"]),
        # NEW:
        Route("/entry", entry, methods=["GET"]),
        Route("/entry/", entry, methods=["GET"]),
        Mount("/", app=otree_app),
    ]
)
