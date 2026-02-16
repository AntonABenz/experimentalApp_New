import json
import logging

from django.http import JsonResponse, HttpResponseBadRequest, HttpResponseForbidden
from django.views.decorators.csrf import csrf_exempt

from otree.models import Participant

from img_desc.utils import verify_prolific_webhook
from img_desc import reset_this_app_for_participant  # uses participant object


logger = logging.getLogger("benzapp.prolific_webhook")


def _get_header(request, name: str) -> str:
    # Django stores headers as HTTP_<UPPER_SNAKE>
    key = "HTTP_" + name.upper().replace("-", "_")
    return request.META.get(key, "")


@csrf_exempt
def prolific_webhook_view(request):
    """
    Prolific webhook receiver.
    Expected headers:
      - X-Prolific-Request-Timestamp
      - X-Prolific-Request-Signature
    Body: JSON payload containing submission status changes.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    raw_body = request.body or b"{}"

    ts = _get_header(request, "X-Prolific-Request-Timestamp")
    sig = _get_header(request, "X-Prolific-Request-Signature")

    # Verify signature (if PROLIFIC_WEBHOOK_SECRET is configured)
    ok = verify_prolific_webhook(raw_body=raw_body, timestamp=ts, signature=sig)
    if not ok:
        # If secret missing, verify_prolific_webhook returns False.
        # In production, you should set PROLIFIC_WEBHOOK_SECRET.
        logger.warning("Webhook signature verification failed (or secret not set).")
        return HttpResponseForbidden("Bad signature")

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return HttpResponseBadRequest("Invalid JSON")

    # ---- Extract fields (payload shape can vary by event type) ----
    # We try a few likely locations.
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
        return HttpResponseBadRequest("Missing prolific_pid or status")

    # ---- Find Participant in oTree ----
    # You must ensure you store prolific_id into participant.vars["prolific_id"]
    # when participant enters the study.
    participant = None
    for p in Participant.objects.all().order_by("-id")[:2000]:
        if p.vars.get("prolific_id") == prolific_pid:
            participant = p
            break

    if participant is None:
        logger.warning(f"No participant found for prolific_id={prolific_pid}")
        return JsonResponse({"ok": True, "note": "participant_not_found"})

    # ---- Update participant status ----
    participant.vars["prolific_submission_status"] = status
    participant.vars["prolific_submission_id"] = submission_id
    participant.vars["study_id_from_webhook"] = study_id
    participant.save()

    # ---- If timed out: reset participant so slot is reusable ----
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

    return JsonResponse({"ok": True})
