# img_desc/utils.py
import requests
import json
import logging
from os import environ
from otree.models import Participant
from django.http import JsonResponse, HttpResponse
from django.views.generic import View
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.shortcuts import redirect
from django.utils import timezone
import pandas as pd

logger = logging.getLogger("img_desc.utils")

# -------------------------------------------------------------------
#  PROLIFIC API KEY
# -------------------------------------------------------------------

PROLIFIC_API_KEY = environ.get("PROLIFIC_API_KEY")
if not PROLIFIC_API_KEY:
    raise ValueError("PROLIFIC_API_KEY not set in environment variables!")

STUBURL = "https://app.prolific.co/submissions/complete?cc="

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Token {PROLIFIC_API_KEY}",
}

# -------------------------------------------------------------------
#  IMAGE URL BUILDER — S3
# -------------------------------------------------------------------

def get_url_for_image(player, img, extension=None):
    """
    Called by Player.get_image_url()
    """
    s3path = player.session.vars.get("s3path")
    extension = extension or player.session.vars.get("extension")

    if not img:
        return "https://picsum.photos/400/300?text=No+Image"

    if img.endswith(f".{extension}"):
        return f"{s3path}{img}"
    return f"{s3path}{img}.{extension}"


# -------------------------------------------------------------------
#  PROLIFIC API HELPERS
# -------------------------------------------------------------------

def get_study(study_id):
    url = f"https://api.prolific.co/api/v1/studies/{study_id}/"
    try:
        res = requests.get(url, headers=HEADERS)
        if res.status_code != 200:
            logger.warning(f"Could not fetch study {study_id}: {res.status_code}")
            return None
        return res.json()
    except Exception as e:
        logger.error(f"Error fetching study {study_id}: {e}")
        return None


def get_completion_info(study_id):
    """
    Called once per study (only first participant triggers this).
    Fetches completion code + auto redirect URL.
    """
    study_data = get_study(study_id)
    if not study_data:
        return dict(
            completion_code="API_ERROR",
            full_return_url=f"{STUBURL}API_ERROR",
        )

    completion_codes = study_data.get("completion_codes", [])
    completed_code = next(
        (c["code"] for c in completion_codes if c.get("code_type") == "COMPLETED"),
        "NO_CODE"
    )

    full_return_url = f"{STUBURL}{completed_code}"

    logger.info(f"Completion code fetched: {completed_code}")
    return dict(
        completion_code=completed_code,
        full_return_url=full_return_url,
    )


def increase_space(study_id, num_extra, max_users):
    """
    Uses Prolific API to increase available slots.
    Called at batch rollover.
    """
    study_info = get_study(study_id)
    if not study_info:
        return None

    current_places = int(study_info.get("total_available_places", 0))
    if current_places >= max_users:
        logger.warning("Slot increase rejected: maximum user count reached.")
        return None

    url = f"https://api.prolific.co/api/v1/studies/{study_id}/"
    payload = json.dumps({"total_available_places": current_places + num_extra})

    logger.info(
        f"PATCH: Increasing Prolific places from {current_places} → "
        f"{current_places + num_extra}"
    )

    try:
        response = requests.patch(url, headers=HEADERS, data=payload)
        if response.status_code != 200:
            logger.warning(f"Error increasing places: {response.status_code}")
            return None
        return response.json()
    except Exception as e:
        logger.error(f"Error calling Prolific PATCH: {e}")
        return None


# -------------------------------------------------------------------
#  PROLIFIC WEBHOOK → FREE SLOTS WHEN RETURNED / REJECTED
# -------------------------------------------------------------------

RETURNED_STATUSES = ["RETURNED", "TIMED-OUT", "REJECTED"]
STATUS_CHANGE = "submission.status.change"


@method_decorator(csrf_exempt, name="dispatch")
class HookView(View):
    """
    Accessible at:  /img_desc/prolific_hook
    """

    def post(self, request, *args, **kwargs):
        try:
            body = json.loads(request.body.decode("utf-8"))
        except Exception:
            return JsonResponse({"message": "invalid json"})

        logger.info("Webhook received:")
        logger.info(body)

        if body.get("event_type") != STATUS_CHANGE:
            return JsonResponse({"message": "ignored"})

        if body.get("status") not in RETURNED_STATUSES:
            return JsonResponse({"message": "ignored event"})

        session_id = body.get("resource_id")
        pid = body.get("participant_id")

        participants = Participant.objects.filter(label=session_id)
        if not participants.exists():
            msg = f"Webhook: No participant with session_id {session_id}"
            logger.warning(msg)
            return JsonResponse({"message": msg})

        messages = []

        for p in participants:
            if p.vars.get("full_study_completed"):
                msg = f"{p.code} completed study; slot not freed"
                messages.append(msg)
                continue

            freed = p.infos.update(busy=False, owner=None)
            if freed:
                msg = f"Slot freed for player {p.code} (Prolific PID={pid})"
            else:
                msg = f"No slot to free for player {p.code}"

            logger.info(msg)
            messages.append(msg)

        return JsonResponse({"message": messages})


# -------------------------------------------------------------------
#  EXPORT (OPTIONAL, SAME AS OLD VERSION)
# -------------------------------------------------------------------

class PandasExport(View):
    content_type = "text/csv"
    url_name = None

    def get(self, request, *args, **kwargs):
        df = self.get_data({})
        if df is None or df.empty:
            return JsonResponse({"error": "no data"})

        timestamp = timezone.now().strftime("%Y_%m_%d_%H_%M_%S")
        csv_data = df.to_csv(index=False)
        resp = HttpResponse(csv_data, content_type=self.content_type)
        resp["Content-Disposition"] = f'attachment; filename="{self.url_name}_{timestamp}.csv"'
        return resp
