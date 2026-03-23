from __future__ import annotations

import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from otree.models import Participant, Session

from img_desc.utils import verify_prolific_webhook
from start import Player as StartPlayer
from img_desc import (
    Player as ImgDescPlayer,
    reset_this_app_for_participant,
    find_prolific_slot_map,
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


def normalize_prolific_status(raw) -> str:
    return clean_str(raw).upper().replace("_", "-")


def _find_participant_in_session(session_code: str, participant_code: str = "", prolific_pid: str = ""):
    session_code = clean_str(session_code)
    participant_code = clean_str(participant_code)
    prolific_pid = clean_str(prolific_pid)
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
            if participant_code and clean_str(getattr(participant, "code", "")) == participant_code:
                return participant
            if prolific_pid and (
                clean_str(getattr(participant, "label", "")) == prolific_pid
                or clean_str(participant.vars.get("prolific_id", "")) == prolific_pid
            ):
                return participant
    except Exception:
        return None

    return None


def _find_participant_by_prolific_id(prolific_pid: str):
    """
    Best-effort lookup.
    Primary: participant.label (canonical Prolific identifier).
    Fallbacks: vars['prolific_id'], recent Player-row scans, then
    player.prolific_id_field in img_desc.
    """
    prolific_pid = (prolific_pid or "").strip()
    if not prolific_pid:
        return None

    mapping = find_prolific_slot_map(prolific_pid=prolific_pid, prefer_active=False)
    if mapping:
        participant = _find_participant_in_session(
            clean_str(getattr(mapping, "session_code", "")),
            participant_code=clean_str(getattr(mapping, "participant_code", "")),
            prolific_pid=prolific_pid,
        )
        if participant:
            return participant

    try:
        qs = Participant.objects.filter(label=prolific_pid)
        if qs.exists():
            return qs.order_by("-id").first()
    except Exception:
        pass

    try:
        qs = Participant.objects.filter(vars__contains={"prolific_id": prolific_pid})
        if qs.exists():
            return qs.order_by("-id").first()
    except Exception:
        pass

    try:
        qs = ImgDescPlayer.objects.filter(prolific_id_field=prolific_pid).order_by("-id")
        for player in qs[:50]:
            participant = getattr(player, "participant", None)
            if participant:
                return participant
    except Exception:
        pass

    for model in (ImgDescPlayer, StartPlayer):
        try:
            recent_players = model.objects.order_by("-id")[:4000]
        except Exception:
            recent_players = []

        for player in recent_players:
            try:
                participant = getattr(player, "participant", None)
                if not participant:
                    continue
                if (
                    getattr(participant, "label", "") == prolific_pid
                    or participant.vars.get("prolific_id") == prolific_pid
                    or clean_str(getattr(player, "prolific_id_field", "")) == prolific_pid
                ):
                    return participant
            except Exception:
                continue

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
        mapping = find_prolific_slot_map(prolific_pid=prolific_pid, prefer_active=False)
        if mapping and status in Constants.BAD_PROLIFIC_STATUSES:
            session_code = clean_str(getattr(mapping, "session_code", ""))
            mapped_code = clean_str(getattr(mapping, "participant_code", ""))
            try:
                session = Session.objects.filter(code=session_code).first() if session_code else None
            except Exception:
                session = None

            if session and mapped_code:
                free_slot_for_participant(session, mapped_code)
                mark_participant_complete_in_cohort(session, mapped_code, completed=False)
                logger.info(
                    "Webhook: freed slot from mapping prolific_pid=%s participant=%s status=%s study_id=%s submission_id=%s",
                    prolific_pid,
                    mapped_code,
                    status,
                    study_id,
                    submission_id,
                )
                return JSONResponse({"ok": True, "note": "mapping_slot_freed"})
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
