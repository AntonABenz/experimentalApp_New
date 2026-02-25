import json
import logging
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from otree.models import Participant
from img_desc.utils import verify_prolific_webhook
from img_desc import (
    reset_this_app_for_participant,
    delete_schedule_for_participant,
    free_slot_for_participant,
    Constants,
)

logger = logging.getLogger("benzapp.prolific_webhook")

BLOCK_FLAG = "blocked_due_to_return"


def _find_participant_by_prolific_id(prolific_pid: str):
    """
    Best-effort lookup. Primary: vars['prolific_id'].
    Fallback: scan recent participants if needed (optional).
    """
    prolific_pid = (prolific_pid or "").strip()
    if not prolific_pid:
        return None

    # Preferred: JSON vars contains
    try:
        qs = Participant.objects.filter(vars__contains={"prolific_id": prolific_pid})
        if qs.exists():
            return qs.order_by("-id").first()
    except Exception:
        pass

    # Fallback: scan recent participants (keeps it from silently failing)
    try:
        for p in Participant.objects.all().order_by("-id")[:3000]:
            if p.vars.get("prolific_id") == prolific_pid:
                return p
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
        logger.warning(
            f"Webhook: participant_not_found prolific_pid={prolific_pid} "
            f"status={status} study_id={study_id} submission_id={submission_id}"
        )
        return JSONResponse({"ok": True, "note": "participant_not_found"})

    # record status details
    participant.vars["prolific_submission_status"] = status
    participant.vars["prolific_submission_id"] = submission_id
    participant.vars["study_id_from_webhook"] = study_id

    # HARD BLOCK for bad statuses to prevent zombie re-assignment
    if status in Constants.BAD_PROLIFIC_STATUSES:
        participant.vars[BLOCK_FLAG] = True

    participant.save()

    # If they already completed the app, don't free their slot/schedule
    already_completed = bool(participant.vars.get(f"{Constants.name_in_url}_completed", False))
    if already_completed:
        logger.info(
            f"Webhook: bad_status={status} but participant already completed app; "
            f"skip freeing slot. participant={participant.code} prolific_pid={prolific_pid}"
        )
        return JSONResponse({"ok": True, "note": "already_completed"})

    # For bad statuses: free slot + delete schedule so replacement can join immediately
    if status in Constants.BAD_PROLIFIC_STATUSES:
        try:
            session = participant.session
            free_slot_for_participant(session, participant.code)
            delete_schedule_for_participant(session, participant.code)
            reset_this_app_for_participant(participant)
            logger.info(
                f"Webhook: freed slot + deleted schedule for participant={participant.code} "
                f"status={status} prolific_pid={prolific_pid}"
            )
        except Exception as e:
            logger.error(
                f"Webhook: failed to free slot/delete schedule participant={participant.code}: {e}",
                exc_info=True,
            )

    return JSONResponse({"ok": True})
