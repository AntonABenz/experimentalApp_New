# asgi.py
import logging
import os
import secrets

from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse, RedirectResponse, PlainTextResponse
from starlette.requests import Request

from otree.asgi import app as otree_app
from prolific_webhook import prolific_webhook_view, _find_participant_by_prolific_id

# NEW
from otree.models import Participant, Session  # Django ORM
from img_desc import (
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

async def ping(request):
    return JSONResponse({"ok": True})

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

    return RedirectResponse(f"/join/{s.code}{suffix}", status_code=302)


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
        participant = Participant.objects.filter(code=participant_code).order_by("-id").first()
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
