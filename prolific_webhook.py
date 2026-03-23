from __future__ import annotations

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from otree.models import Session

from img_desc.utils import verify_prolific_webhook
from img_desc import (
    Player as ImgDescPlayer,
    reset_this_app_for_participant,
    free_slot_for_participant,
    mark_participant_active,
    mark_participant_complete_in_cohort,
    mark_participant_drop_out,
    mark_participant_finished,
    maybe_expand_prolific_for_participant,
    Constants,
    clean_str,
    get_participant_status,
)

logger = logging.getLogger("benzapp.prolific_webhook")


def _iter_recent_participants(limit_sessions: int = 50):
    try:
        sessions = Session.objects.filter(is_demo=False).order_by("-id")[:limit_sessions]
    except Exception:
        sessions = []

    for session in sessions:
        try:
            participants = session.get_participants()
        except Exception:
            continue
        for participant in participants:
            yield participant


def normalize_prolific_status(raw) -> str:
    return clean_str(raw).upper().replace("_", "-")


def _find_participant_by_prolific_id(prolific_pid: str):
    """
    Best-effort lookup.
    Primary: vars['prolific_id'].
    Fallbacks: participant.label, then recent-participant scan.
    """
    prolific_pid = (prolific_pid or "").strip()
    if not prolific_pid:
        return None

    for participant in _iter_recent_participants():
        try:
            if participant.vars.get("prolific_id") == prolific_pid or getattr(participant, "label", "") == prolific_pid:
                return participant
        except Exception:
            continue

    try:
        for player in reversed(ImgDescPlayer.filter(prolific_id_field=prolific_pid)):
            participant = getattr(player, "participant", None)
            if participant:
                return participant
    except Exception:
        pass

    return None


async def prolific_webhook_view(request: Request):
    if request.method == "GET":
        return PlainTextResponse("ok", status_code=200)

    if request.method != "POST":
        return PlainTextResponse("POST required", status_code=400)

    raw_body = await request.body()

    ts = request.headers.get("X-Prolific-Request-Timestamp", "")
    sig = request.headers.get("X-Prolific-Request-Signature", "")

    ok = verify_prolific_webhook(raw_body=raw_body, timestamp=ts, signature=sig)
    if not ok:
        logger.warning("Webhook signature verification failed.")
        return PlainTextResponse("Bad signature", status_code=403)

    try:
        payload = json.loads(raw_body.decode("utf-8") or "{}")
    except Exception:
        return PlainTextResponse("Invalid JSON", status_code=400)

    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        data = payload if isinstance(payload, dict) else {}

    status = normalize_prolific_status(
        data.get("status") or data.get("submission_status") or ""
    )

    prolific_pid = (
        data.get("participant_id")
        or data.get("prolific_pid")
        or data.get("prolific_id")
        or ""
    ).strip()

    study_id = (data.get("study_id") or "").strip()
    submission_id = (data.get("submission_id") or data.get("id") or "").strip()

    if not prolific_pid or not status:
        return PlainTextResponse("Missing prolific_pid or status", status_code=400)

    participant = _find_participant_by_prolific_id(prolific_pid)

    if participant is None:
        logger.warning(
            "Webhook: participant_not_found prolific_pid=%s status=%s study_id=%s submission_id=%s",
            prolific_pid,
            status,
            study_id,
            submission_id,
        )
        return JSONResponse({"ok": True, "note": "participant_not_found"})

    if not get_participant_status(participant):
        mark_participant_active(participant)

    participant.vars["prolific_submission_status"] = status
    participant.vars["prolific_submission_id"] = submission_id
    participant.vars["study_id_from_webhook"] = study_id

    if status in Constants.BAD_PROLIFIC_STATUSES:
        mark_participant_drop_out(participant)
    elif status in Constants.APPROVED_PROLIFIC_STATUSES:
        mark_participant_finished(participant)
        mark_participant_complete_in_cohort(
            participant.session,
            participant.code,
            completed=True,
        )
    elif status in Constants.ACTIVE_PROLIFIC_STATUSES:
        mark_participant_active(participant)

    try:
        participant.save()
    except Exception:
        logger.exception(
            "Webhook: failed saving participant status. participant=%s prolific_pid=%s",
            participant.code,
            prolific_pid,
        )

    if status in Constants.BAD_PROLIFIC_STATUSES:
        try:
            session = participant.session

            free_slot_for_participant(session, participant.code)
            reset_this_app_for_participant(participant)
            mark_participant_complete_in_cohort(
                session,
                participant.code,
                completed=False,
            )
            participant.save()

            logger.info(
                "Webhook: freed slot without deleting schedule for participant=%s status=%s prolific_pid=%s",
                participant.code,
                status,
                prolific_pid,
            )
        except Exception as e:
            logger.error(
                "Webhook: failed to free slot safely participant=%s prolific_pid=%s error=%s",
                participant.code,
                prolific_pid,
                e,
                exc_info=True,
            )

    if status in Constants.APPROVED_PROLIFIC_STATUSES:
        maybe_expand_prolific_for_participant(participant)
        logger.info(
            "Webhook: marked participant finished participant=%s status=%s prolific_pid=%s",
            participant.code,
            status,
            prolific_pid,
        )

    return JSONResponse({"ok": True})
