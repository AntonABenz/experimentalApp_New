import json
import logging

from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse

from otree.models import Participant

from img_desc.utils import verify_prolific_webhook
from img_desc import reset_this_app_for_participant  # uses participant object

logger = logging.getLogger("benzapp.prolific_webhook")


async def prolific_webhook_view(request: Request):
    """
    Prolific webhook receiver (Starlette endpoint).

    Expected headers:
      - X-Prolific-Request-Timestamp
      - X-Prolific-Request-Signature
    Body: JSON payload containing submission status changes.
    """
    if request.method == "GET":
        # Useful for manual testing
        return JSONResponse({"ok": True, "note": "webhook endpoint alive"})

    if request.method != "POST":
        return PlainTextResponse("POST required", status_code=400)

    raw_body = await request.body()
    if not raw_body:
        raw_body = b"{}"

    ts = request.headers.get("X-Prolific-Request-Timestamp", "")
    sig = request.headers.get("X-Prolific-Request-Signature", "")

    ok = verify_prolific_webhook(raw_body=raw_body, timestamp=ts, signature=sig)
    if not ok:
        logger.warning("Webhook signature verification failed (or secret not set).")
        return PlainTextResponse("Bad signature", status_code=403)

    try:
        payload = json.loads(raw_body.decode("utf-8"))
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
    )
    study_id = data.get("study_id") or ""
    submission_id = data.get("submission_id") or data.get("id") or ""

    if not prolific_pid or not status:
        logger.warning(f"Webhook missing prolific_pid or status. payload={payload}")
        return PlainTextResponse("Missing prolific_pid or status", status_code=400)

    # ---- Find Participant in oTree ----
    participant = None
    for p in Participant.objects.all().order_by("-id")[:2000]:
        if p.vars.get("prolific_id") == prolific_pid:
            participant = p
            break

    if participant is None:
        logger.warning(f"No participant found for prolific_id={prolific_pid}")
        return JSONResponse({"ok": True, "note": "participant_not_found"})

    participant.vars["prolific_submission_status"] = status
    participant.vars["prolific_submission_id"] = submission_id
    participant.vars["study_id_from_webhook"] = study_id
    participant.save()

    if status == "TIMED-OUT":
        logger.warning(
            f"TIMED-OUT: prolific_id={prolific_pid} "
            f"participant_code={participant.code} "
            f"study_id={study_id} submission_id={submission_id} "
            f"exp_target={participant.vars.get('exp_target')} "
            f"local_slot={participant.vars.get('local_slot')}"
        )
        try:
            reset_this_app_for_participant(participant)
        except Exception as e:
            logger.error(f"Failed to reset participant {participant.code}: {e}", exc_info=True)

    return JSONResponse({"ok": True})
