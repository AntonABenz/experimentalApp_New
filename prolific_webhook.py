# prolific_webhook.py
import json
import logging
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from otree.models import Participant

from img_desc.utils import verify_prolific_webhook
from img_desc import (
    reset_this_app_for_participant,
    Constants as ImgDescConstants,
    free_slot_for_participant,
    delete_schedule_for_participant,
)

logger = logging.getLogger("benzapp.prolific_webhook")


def _find_participant_by_prolific_id(prolific_pid: str):
    """
    Prefer a DB-level filter if Participant.vars is a JSONField.
    Fallback to scanning a limited recent window.
    """
    prolific_pid = (prolific_pid or "").strip()
    if not prolific_pid:
        return None

    # Try JSONField contains lookup (works on Django JSONField/Postgres).
    try:
        qs = Participant.objects.filter(vars__contains={"prolific_id": prolific_pid}).order_by("-id")
        return qs.first()
    except Exception:
        pass

    # Fallback: scan recent participants (bounded)
    for p in Participant.objects.all().order_by("-id")[:5000]:
        try:
            if p.vars.get("prolific_id") == prolific_pid:
                return p
        except Exception:
            continue
    return None


async def prolific_webhook_view(request: Request):
    # Health check / endpoint presence
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

    status = (data.get("status") or data.get("submission_status") or "").strip()
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
        # Acknowledge Prolific but nothing to do.
        return JSONResponse({"ok": True, "note": "participant_not_found"})

    # Store latest webhook info
    participant.vars["prolific_submission_status"] = status
    participant.vars["prolific_submission_id"] = submission_id
    participant.vars["study_id_from_webhook"] = study_id
    participant.save()

    # Only free slot/reset if participant did NOT complete img_desc.
    # Otherwise a completed participant could later "return" and we'd incorrectly free a completed slot.
    completed = bool(participant.vars.get(f"{ImgDescConstants.name_in_url}_completed", False))

    if (status in ImgDescConstants.BAD_PROLIFIC_STATUSES) and (not completed):
        try:
            # Free their cohort slot and delete schedule immediately
            # (IMPORTANT: do it here, because they might never revisit your oTree pages).
            session = getattr(participant, "session", None)
            if session is not None:
                free_slot_for_participant(session, participant.code)
                delete_schedule_for_participant(session, participant.code)

            reset_this_app_for_participant(participant)
            participant.save()

            logger.info(
                f"Freed slot + reset for participant={participant.code} prolific_id={prolific_pid} status={status}"
            )
        except Exception as e:
            logger.error(
                f"Failed to free slot/reset participant {participant.code}: {e}",
                exc_info=True,
            )

    return JSONResponse({"ok": True})
