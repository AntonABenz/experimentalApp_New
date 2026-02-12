import base64
import hashlib
import hmac
import json
import logging
import os

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

from otree.models import Participant  # oTree's Participant model

logger = logging.getLogger("benzapp.prolific_webhook")


def _verify_prolific_signature(raw_body: bytes, timestamp: str, signature: str, secret: str) -> bool:
    """
    Prolific verification (HMAC-SHA256 over timestamp + body, base64 digest). :contentReference[oaicite:3]{index=3}
    """
    if not (timestamp and signature and secret):
        return False

    calculated = base64.b64encode(
        hmac.new(
            secret.encode(),
            (timestamp + raw_body.decode("utf-8")).encode("utf-8"),
            hashlib.sha256,
        ).digest()
    )

    # signature header is the base64 string
    return hmac.compare_digest(calculated, signature.encode("utf-8"))


@csrf_exempt
def prolific_webhook(request):
    secret = os.environ.get("PROLIFIC_WEBHOOK_SECRET", "")
    signature = request.headers.get("X-Prolific-Request-Signature", "")
    timestamp = request.headers.get("X-Prolific-Request-Timestamp", "")

    raw_body = request.body or b"{}"

    if not _verify_prolific_signature(raw_body, timestamp, signature, secret):
        logger.warning("Invalid Prolific webhook signature")
        return HttpResponse("invalid signature", status=401)

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return HttpResponse("bad json", status=400)

    event_type = payload.get("event_type")
    if event_type != "submission.status.change":
        return JsonResponse({"ok": True, "ignored": True})

    # Receiving payload schema: resource_id, event_type, participant_id, status :contentReference[oaicite:4]{index=4}
    prolific_participant_id = payload.get("participant_id")
    status = payload.get("status")
    submission_id = payload.get("resource_id")

    # Find matching oTree participant(s) by prolific_id stored in participant.vars["prolific_id"]
    # NOTE: JSONField lookup syntax can differ depending on oTree/Django versions.
    # This "contains" pattern is the most portable.
    matches = Participant.objects.filter(_vars__contains={"prolific_id": prolific_participant_id})

    if not matches.exists():
        logger.warning(
            "Webhook: no matching participant for prolific_id=%s status=%s submission=%s",
            prolific_participant_id, status, submission_id
        )
        return JsonResponse({"ok": True, "matched": 0})

    for part in matches:
        # Update vars so app can block them if they come back
        part.vars["prolific_submission_status"] = status
        part.vars["prolific_submission_id"] = submission_id
        part.save()

        exp_target = part.vars.get("exp_target", "")
        local_slot = part.vars.get("local_slot", "")

        logger.warning(
            "Prolific status update: prolific_id=%s status=%s submission=%s otree_participant=%s exp=%s slot=%s",
            prolific_participant_id, status, submission_id, part.code, exp_target, local_slot
        )

        # Only discard data if timed-out (or add RETURNED etc if you want)
        # TIMED-OUT is an official submission status. :contentReference[oaicite:5]{index=5}
        if status == "TIMED-OUT":
            from img_desc import reset_this_app_for_participant  # import your helper
            reset_this_app_for_participant(part)

            logger.warning(
                "Discarded oTree data due to TIMED-OUT: prolific_id=%s otree_participant=%s exp=%s slot=%s",
                prolific_participant_id, part.code, exp_target, local_slot
            )

    return JsonResponse({"ok": True, "matched": matches.count()})
