import json
import logging

from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from otree.models import Participant

from img_desc.utils import verify_prolific_webhook
from img_desc import reset_this_app_for_participant  # uses your existing function

logger = logging.getLogger("benzapp.prolific_webhook")


@csrf_exempt
def prolific_webhook(request):
    raw_body = request.body or b"{}"

    signature = request.headers.get("X-Prolific-Request-Signature", "")
    timestamp = request.headers.get("X-Prolific-Request-Timestamp", "")

    if not verify_prolific_webhook(raw_body, timestamp, signature):
        logger.warning("Rejected webhook: invalid signature")
        return HttpResponse("invalid signature", status=401)

    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return HttpResponse("bad json", status=400)

    event_type = payload.get("event_type")
    if event_type != "submission.status.change":
        return JsonResponse({"ok": True, "ignored": True})

    prolific_participant_id = payload.get("participant_id")
    status = payload.get("status")          # e.g. TIMED-OUT
    submission_id = payload.get("resource_id")

    # We assume you store Prolific participant_id in participant.vars["prolific_id"]
    qs = Participant.objects.filter(_vars__contains={"prolific_id": prolific_participant_id})

    if not qs.exists():
        logger.warning(
            "Webhook: no participant matched prolific_id=%s status=%s submission=%s",
            prolific_participant_id, status, submission_id
        )
        return JsonResponse({"ok": True, "matched": 0})

    for part in qs:
        exp_target = part.vars.get("exp_target", "")
        local_slot = part.vars.get("local_slot", "")

        part.vars["prolific_submission_status"] = status
        part.vars["prolific_submission_id"] = submission_id
        part.save()

        logger.warning(
            "PROLIFIC_STATUS: prolific_id=%s status=%s submission=%s otree_participant=%s exp=%s slot=%s",
            prolific_participant_id, status, submission_id, part.code, exp_target, local_slot
        )

        if status == "TIMED-OUT":
            reset_this_app_for_participant(part)
            logger.warning(
                "DISCARDED: TIMED-OUT prolific_id=%s otree_participant=%s exp=%s slot=%s",
                prolific_participant_id, part.code, exp_target, local_slot
            )

    return JsonResponse({"ok": True, "matched": qs.count()})
