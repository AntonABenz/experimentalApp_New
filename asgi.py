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
    CohortSlot,
    Constants as ImgDescConstants,
    Player as ImgDescPlayer,
    ScheduleItem,
    clean_str,
    find_prolific_slot_map,
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

    try:
        sessions = Session.objects.order_by("-id")[:200]
    except Exception:
        sessions = []

    for session in sessions:
        try:
            for participant in session.get_participants():
                if _clean_cookie_value(getattr(participant, "code", "")) == participant_code:
                    return participant
        except Exception:
            continue

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

    mapping = find_prolific_slot_map(participant_code=participant_code, prefer_active=False)
    if mapping:
        participant = _find_participant_in_session(
            clean_str(getattr(mapping, "session_code", "")),
            participant_code=participant_code,
        )
        if participant:
            return participant

    # Compatibility fallback for deployed runtimes where relation filters can
    # miss existing participants. Scan recent concrete Player rows and compare
    # the resolved participant object directly.
    for model in (ImgDescPlayer, StartPlayer):
        try:
            recent_players = model.objects.order_by("-id")[:4000]
        except Exception:
            recent_players = []

        for player in recent_players:
            try:
                participant = getattr(player, "participant", None)
                if participant and _clean_cookie_value(getattr(participant, "code", "")) == participant_code:
                    return participant
            except Exception:
                continue

    return None


def _find_participant_in_session(session_code: str, participant_code: str = "", prolific_id: str = ""):
    session_code = _clean_cookie_value(session_code)
    participant_code = _clean_cookie_value(participant_code)
    prolific_id = _clean_cookie_value(prolific_id)
    if not session_code:
        return None

    try:
        session = Session.objects.filter(code=session_code).first()
    except Exception:
        session = None
    if not session:
        return None

    try:
        for participant in session.get_participants():
            if participant_code and _clean_cookie_value(getattr(participant, "code", "")) == participant_code:
                return participant
            if prolific_id and _clean_cookie_value(get_participant_prolific_id(participant)) == prolific_id:
                return participant
    except Exception:
        return None

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
    mapping = find_prolific_slot_map(
        participant_code=clean_str(getattr(participant, "code", "")),
        session_code=clean_str(getattr(participant.session, "code", "")),
        prefer_active=False,
    )

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
        prolific_slot_map=(
            dict(
                session_code=clean_str(getattr(mapping, "session_code", "")),
                participant_code=clean_str(getattr(mapping, "participant_code", "")),
                prolific_pid=clean_str(getattr(mapping, "prolific_pid", "")),
                exp_num=safe_int(getattr(mapping, "exp_num", 0), 0),
                slot=safe_int(getattr(mapping, "slot", 0), 0),
                active=bool(getattr(mapping, "active", False)),
                last_status=clean_str(getattr(mapping, "last_status", "")),
            )
            if mapping
            else {}
        ),
        slot_rows=slot_rows,
        cohort_snapshots=[get_cohort_snapshot_data(participant.session, exp_num) for exp_num in exp_nums],
    )


def _serialize_mapping_state(mapping):
    if not mapping:
        return {}

    session_code = clean_str(getattr(mapping, "session_code", ""))
    exp_num = safe_int(getattr(mapping, "exp_num", 0), 0)
    snapshots = []
    try:
        session = Session.objects.filter(code=session_code).first() if session_code else None
    except Exception:
        session = None
    if session and exp_num > 0:
        try:
            snapshots = [get_cohort_snapshot_data(session, exp_num)]
        except Exception:
            snapshots = []

    return dict(
        participant_code=clean_str(getattr(mapping, "participant_code", "")),
        prolific_id=clean_str(getattr(mapping, "prolific_pid", "")),
        participant_label=clean_str(getattr(mapping, "prolific_pid", "")),
        participant_status="",
        prolific_submission_status="",
        returned_from_prolific=False,
        exp_target=exp_num,
        local_slot=safe_int(getattr(mapping, "slot", 0), 0),
        prolific_slot_map=dict(
            session_code=session_code,
            participant_code=clean_str(getattr(mapping, "participant_code", "")),
            prolific_pid=clean_str(getattr(mapping, "prolific_pid", "")),
            exp_num=exp_num,
            slot=safe_int(getattr(mapping, "slot", 0), 0),
            active=bool(getattr(mapping, "active", False)),
            last_status=clean_str(getattr(mapping, "last_status", "")),
        ),
        slot_rows=[],
        cohort_snapshots=snapshots,
    )


def _recent_sessions(limit: int = 200):
    try:
        return list(Session.objects.order_by("-id")[: int(limit or 200)])
    except Exception:
        return []


def _recent_img_desc_roots(limit: int = 200):
    roots = []
    for session in _recent_sessions(limit):
        try:
            for sub in session.get_subsessions():
                if (
                    getattr(getattr(sub, "_meta", None), "app_label", "") == ImgDescConstants.name_in_url
                    and int(getattr(sub, "round_number", 0) or 0) == 1
                ):
                    roots.append(sub)
                    break
        except Exception:
            continue
    return roots


def _slot_rows_from_root(root, participant_code: str):
    participant_code = clean_str(participant_code)
    if not root or not participant_code:
        return []
    try:
        rows = CohortSlot.filter(subsession=root, participant_code=participant_code)
    except Exception:
        return []

    result = []
    for row in sorted(rows, key=lambda r: (int(getattr(r, "exp_num", 0) or 0), int(getattr(r, "slot", 0) or 0), not bool(getattr(r, "active", False)))):
        result.append(
            dict(
                exp_num=int(getattr(row, "exp_num", 0) or 0),
                slot=int(getattr(row, "slot", 0) or 0),
                active=bool(getattr(row, "active", False)),
                completed=bool(getattr(row, "completed", False)),
            )
        )
    return result


def _recent_img_desc_players(limit: int = 4000):
    try:
        return list(ImgDescPlayer.objects.order_by("-id")[: int(limit or 4000)])
    except Exception:
        return []


def _find_recent_img_desc_player(participant_code: str = "", prolific_id: str = ""):
    participant_code = clean_str(participant_code)
    prolific_id = clean_str(prolific_id)
    if not participant_code and not prolific_id:
        return None

    for player in _recent_img_desc_players():
        try:
            participant = getattr(player, "participant", None)
            player_code = clean_str(getattr(participant, "code", ""))
            player_pid = clean_str(getattr(player, "prolific_id_field", ""))
            if participant_code and player_code == participant_code:
                return player
            if prolific_id and player_pid == prolific_id:
                return player
        except Exception:
            continue
    return None


def _has_schedule_rows(root, participant_code: str) -> bool:
    participant_code = clean_str(participant_code)
    if not root or not participant_code:
        return False
    try:
        rows = ScheduleItem.filter(subsession=root, participant_code=participant_code)
    except Exception:
        return False
    return bool(rows)


def _find_repair_fallback_by_participant_code(participant_code: str):
    participant_code = clean_str(participant_code)
    if not participant_code:
        return None

    player = _find_recent_img_desc_player(participant_code=participant_code)
    if player:
        participant = getattr(player, "participant", None)
        session = getattr(participant, "session", None) or getattr(getattr(player, "subsession", None), "session", None)
        try:
            root = player.subsession.in_round(1)
        except Exception:
            root = None
        slot_rows = _slot_rows_from_root(root, participant_code)
        logger.info(
            "CohortRepair lookup: img_desc player match participant=%s root=%s slot_rows=%s",
            participant_code,
            bool(root),
            len(slot_rows),
        )
        return dict(
            participant_code=participant_code,
            prolific_id=clean_str(getattr(player, "prolific_id_field", "")),
            session=session,
            session_code=clean_str(getattr(session, "code", "")),
            slot_rows=slot_rows,
        )

    for root in _recent_img_desc_roots():
        if _has_schedule_rows(root, participant_code):
            session = getattr(root, "session", None)
            slot_rows = _slot_rows_from_root(root, participant_code)
            return dict(
                participant_code=participant_code,
                prolific_id="",
                session=session,
                session_code=clean_str(getattr(session, "code", "")),
                slot_rows=slot_rows,
            )
        slot_rows = _slot_rows_from_root(root, participant_code)
        if slot_rows:
            return dict(
                participant_code=participant_code,
                prolific_id="",
                session=getattr(root, "session", None),
                session_code=clean_str(getattr(getattr(root, "session", None), "code", "")),
                slot_rows=slot_rows,
            )
    return None


def _find_repair_fallback_by_prolific_id(prolific_id: str):
    prolific_id = clean_str(prolific_id)
    if not prolific_id:
        return None

    player = _find_recent_img_desc_player(prolific_id=prolific_id)
    if player:
        try:
            participant = getattr(player, "participant", None)
            participant_code = clean_str(getattr(participant, "code", ""))
            session = getattr(participant, "session", None) or getattr(getattr(player, "subsession", None), "session", None)
            try:
                root = player.subsession.in_round(1)
            except Exception:
                root = None
            slot_rows = _slot_rows_from_root(root, participant_code)
            logger.info(
                "CohortRepair lookup: img_desc player PID match prolific_id=%s participant=%s root=%s slot_rows=%s",
                prolific_id,
                participant_code,
                bool(root),
                len(slot_rows),
            )
            return dict(
                participant_code=participant_code,
                prolific_id=prolific_id,
                session=session,
                session_code=clean_str(getattr(session, "code", "")),
                slot_rows=slot_rows,
            )
        except Exception:
            pass

    try:
        players = list(ImgDescPlayer.objects.filter(prolific_id_field=prolific_id).order_by("-id")[:200])
    except Exception:
        players = []

    if not players:
        try:
            players = list(ImgDescPlayer.objects.order_by("-id")[:4000])
        except Exception:
            players = []
        players = [p for p in players if clean_str(getattr(p, "prolific_id_field", "")) == prolific_id]

    for player in players:
        try:
            participant = getattr(player, "participant", None)
            participant_code = clean_str(getattr(participant, "code", ""))
            session = getattr(participant, "session", None)
            if participant_code and session is not None:
                slot_rows = get_participant_slot_rows(session, participant_code)
                return dict(
                    participant_code=participant_code,
                    prolific_id=prolific_id,
                    session=session,
                    session_code=clean_str(getattr(session, "code", "")),
                    slot_rows=slot_rows,
                )
        except Exception:
            continue
    return None


def _find_repair_fallback(participant_code: str, prolific_id: str):
    fallback = _find_repair_fallback_by_participant_code(participant_code)
    if fallback:
        if not fallback.get("prolific_id") and prolific_id:
            fallback["prolific_id"] = clean_str(prolific_id)
        return fallback
    return _find_repair_fallback_by_prolific_id(prolific_id)


def _serialize_repair_fallback(fallback):
    if not fallback:
        return {}

    session = fallback.get("session")
    slot_rows = list(fallback.get("slot_rows") or [])
    exp_nums = sorted({int(row.get("exp_num", 0) or 0) for row in slot_rows if int(row.get("exp_num", 0) or 0) > 0})
    if not exp_nums:
        exp_nums = [1]

    snapshots = []
    for exp_num in exp_nums:
        try:
            snapshots.append(get_cohort_snapshot_data(session, exp_num))
        except Exception:
            continue

    local_slot = 0
    for row in slot_rows:
        if bool(row.get("active")):
            local_slot = safe_int(row.get("slot", 0), 0)
            break
    if not local_slot and slot_rows:
        local_slot = safe_int(slot_rows[-1].get("slot", 0), 0)

    return dict(
        participant_code=clean_str(fallback.get("participant_code", "")),
        prolific_id=clean_str(fallback.get("prolific_id", "")),
        participant_label=clean_str(fallback.get("prolific_id", "")),
        participant_status="",
        prolific_submission_status="",
        returned_from_prolific=False,
        exp_target=safe_int(exp_nums[0] if exp_nums else 0, 0),
        local_slot=local_slot,
        prolific_slot_map={},
        slot_rows=slot_rows,
        cohort_snapshots=snapshots,
    )


def _find_participant_for_repair(participant_code: str, prolific_id: str):
    participant_code = clean_str(participant_code)
    prolific_id = clean_str(prolific_id)

    if participant_code:
        participant = _find_participant_by_code(participant_code)
        if participant:
            return participant

    if prolific_id:
        mapping = find_prolific_slot_map(prolific_pid=prolific_id, prefer_active=False)
        if mapping:
            participant = _find_participant_in_session(
                clean_str(getattr(mapping, "session_code", "")),
                participant_code=clean_str(getattr(mapping, "participant_code", "")),
                prolific_id=prolific_id,
            )
            if participant:
                return participant
        return _find_participant_by_prolific_id(prolific_id)

    return None


def _find_mapping_for_repair(participant_code: str, prolific_id: str):
    participant_code = clean_str(participant_code)
    prolific_id = clean_str(prolific_id)
    if participant_code:
        mapping = find_prolific_slot_map(participant_code=participant_code, prefer_active=False)
        if mapping:
            return mapping
    if prolific_id:
        return find_prolific_slot_map(prolific_pid=prolific_id, prefer_active=False)
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
    mapping = None if participant else _find_mapping_for_repair(participant_code, prolific_id)
    fallback = None if (participant or mapping) else _find_repair_fallback(participant_code, prolific_id)
    if not participant and not mapping and not fallback:
        logger.warning(
            "CohortRepair lookup miss: participant_code=%s prolific_id=%s",
            participant_code,
            prolific_id,
        )
        return JSONResponse(
            {"ok": False, "error": "participant not found", "participant_code": participant_code, "prolific_id": prolific_id},
            status_code=404,
        )

    before = (
        _serialize_participant_state(participant)
        if participant
        else (_serialize_mapping_state(mapping) if mapping else _serialize_repair_fallback(fallback))
    )
    note = "status only"

    if participant and action == "drop_out":
        mark_participant_drop_out(participant)
        free_slot_for_participant(participant.session, participant.code)
        reset_this_app_for_participant(participant)
        mark_participant_complete_in_cohort(participant.session, participant.code, completed=False)
        participant.save()
        note = "participant marked drop_out, slot freed, app state reset"
    elif participant and action == "finished":
        mark_participant_finished(participant)
        mark_participant_complete_in_cohort(participant.session, participant.code, completed=True)
        participant.save()
        maybe_expand_prolific_for_participant(participant)
        note = "participant marked finished and cohort completion updated"
    elif participant and action == "free_slot":
        free_slot_for_participant(participant.session, participant.code)
        reset_this_app_for_participant(participant)
        mark_participant_complete_in_cohort(participant.session, participant.code, completed=False)
        participant.save()
        note = "slot freed and app state reset; participant status unchanged"
    elif participant and action == "active":
        mark_participant_active(participant)
        participant.save()
        note = "participant marked active"
    elif mapping and action in {"status", "drop_out", "free_slot"}:
        session_code = clean_str(getattr(mapping, "session_code", ""))
        mapped_code = clean_str(getattr(mapping, "participant_code", ""))
        try:
            session = Session.objects.filter(code=session_code).first() if session_code else None
        except Exception:
            session = None

        if action in {"drop_out", "free_slot"}:
            if not session or not mapped_code:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "mapping found but session/participant code unavailable",
                        "participant_code": participant_code,
                        "prolific_id": prolific_id,
                    },
                    status_code=409,
                )
            free_slot_for_participant(session, mapped_code)
            mark_participant_complete_in_cohort(session, mapped_code, completed=False)
            note = "mapping-only slot free completed; participant object unresolved"
        else:
            note = "mapping-only status; participant object unresolved"
    elif fallback and action in {"status", "drop_out", "free_slot"}:
        session = fallback.get("session")
        fallback_code = clean_str(fallback.get("participant_code", ""))
        if action in {"drop_out", "free_slot"}:
            if session is None or not fallback_code:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "repair fallback found but session/participant code unavailable",
                        "participant_code": participant_code,
                        "prolific_id": prolific_id,
                    },
                    status_code=409,
                )
            free_slot_for_participant(session, fallback_code)
            mark_participant_complete_in_cohort(session, fallback_code, completed=False)
            note = "repair-only slot free completed from slot/player fallback"
        else:
            note = "repair-only status from slot/player fallback"
    elif mapping:
        return JSONResponse(
            {
                "ok": False,
                "error": "participant object unresolved for requested action",
                "participant_code": participant_code,
                "prolific_id": prolific_id,
            },
            status_code=409,
        )
    elif fallback:
        return JSONResponse(
            {
                "ok": False,
                "error": "repair fallback unresolved for requested action",
                "participant_code": participant_code,
                "prolific_id": prolific_id,
            },
            status_code=409,
        )

    after = (
        _serialize_participant_state(participant)
        if participant
        else (_serialize_mapping_state(mapping) if mapping else _serialize_repair_fallback(fallback))
    )
    logger.info(
        "CohortRepair: action=%s participant=%s prolific_id=%s status_before=%s status_after=%s",
        action,
        clean_str(getattr(participant, "code", ""))
        or clean_str(getattr(mapping, "participant_code", ""))
        or clean_str((fallback or {}).get("participant_code", "")),
        get_participant_prolific_id(participant)
        if participant
        else (
            clean_str(getattr(mapping, "prolific_pid", ""))
            or clean_str((fallback or {}).get("prolific_id", ""))
        ),
        before.get("participant_status", "") or clean_str(getattr(mapping, "last_status", "")),
        after.get("participant_status", "") or clean_str(getattr(mapping, "last_status", "")),
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
