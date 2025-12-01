# img_desc/utils.py

import json
import logging
import os

import requests

logger = logging.getLogger("benzapp.utils")

# -------------------------------------------------------------------
#  PROLIFIC CONFIG
# -------------------------------------------------------------------

PROLIFIC_API_KEY = os.environ.get("PROLIFIC_API_KEY")

if not PROLIFIC_API_KEY:
    # IMPORTANT: do NOT crash if the key is missing
    logger.warning(
        "PROLIFIC_API_KEY not set; Prolific API features (balance, "
        "increase_space, completion lookup) are disabled."
    )

STUBURL = "https://app.prolific.co/submissions/complete?cc="

BASE_HEADERS = {
    "Content-Type": "application/json",
}
if PROLIFIC_API_KEY:
    BASE_HEADERS["Authorization"] = f"Token {PROLIFIC_API_KEY}"

BALANCE_URL = (
    "https://api.prolific.co/api/v1/workspaces/"
    "647787b9fe04daac6e2e944e/balance/"
)


# -------------------------------------------------------------------
#  IMAGE URL HELPER (used by Player.get_image_url)
# -------------------------------------------------------------------

def get_url_for_image(player, img: str, extension: str | None = None) -> str:
    """
    Build full URL for an image, using session.vars entries that were
    loaded from the Excel sheet in creating_session.
    """
    s3path = player.session.vars.get("s3path") or ""
    ext = extension or player.session.vars.get("extension") or "png"

    # if img already has an extension, don't duplicate
    if img and not img.lower().endswith(f".{ext}"):
        img = f"{img}.{ext}"

    return f"{s3path}{img}"


# -------------------------------------------------------------------
#  PROLIFIC API HELPERS
# -------------------------------------------------------------------

def _can_call_prolific() -> bool:
    if not PROLIFIC_API_KEY:
        logger.warning(
            "Attempted to call Prolific API but PROLIFIC_API_KEY is not set. "
            "Skipping API call."
        )
        return False
    return True


def get_balance():
    """
    Returns Prolific workspace balance JSON, or None if unavailable.
    """
    if not _can_call_prolific():
        return None

    try:
        resp = requests.get(BALANCE_URL, headers=BASE_HEADERS, timeout=10)
    except Exception as e:
        logger.warning(f"Error calling Prolific balance API: {e}")
        return None

    if resp.status_code != 200:
        logger.warning(
            f"Get error trying to get balance. Status code: {resp.status_code}"
        )
        return None

    return resp.json()


def get_study(study_id: str):
    """
    Fetches study info from Prolific. Returns JSON or None.
    """
    if not _can_call_prolific():
        return None

    url = f"https://api.prolific.co/api/v1/studies/{study_id}/"

    try:
        resp = requests.get(url, headers=BASE_HEADERS, timeout=10)
    except Exception as e:
        logger.warning(f"Error calling Prolific get_study({study_id}): {e}")
        return None

    if resp.status_code != 200:
        logger.warning(
            f"Get error trying to get data for study {study_id}. "
            f"Status code: {resp.status_code}"
        )
        return None

    return resp.json()


def get_completion_info(study_id: str):
    """
    Returns dict(completion_code=..., full_return_url=...) or None.

    If the Prolific API is unavailable, caller should fall back to
    'API_ERROR' codes (which your img_desc app already does).
    """
    study_data = get_study(study_id)
    if not study_data:
        logger.warning(f"Failed to get study data for {study_id}")
        return None

    if isinstance(study_data, dict) and study_data.get("error"):
        logger.warning(f"Get error trying to get data for study {study_id}")
        return None

    completion_codes = study_data.get("completion_codes", [])

    # First COMPLETED code, or "NO_CODE"
    completed_code = next(
        (
            cinfo.get("code")
            for cinfo in completion_codes
            if cinfo.get("code_type") == "COMPLETED"
        ),
        "NO_CODE",
    )

    full_return_url = f"{STUBURL}{completed_code}"
    logger.info(
        f"full_return_url: {full_return_url}; completed_code: {completed_code}"
    )

    return dict(
        completion_code=completed_code,
        full_return_url=full_return_url,
    )


def increase_space(study_id: str, num_extra: int, max_users: int):
    """
    Attempts to increase 'total_available_places' on Prolific, but
    only if PROLIFIC_API_KEY is set.

    Returns JSON response or None.
    """
    if not _can_call_prolific():
        return None

    study = get_study(study_id)
    if not study:
        logger.warning("Something wrong with response when getting study data.")
        return None

    try:
        num_current_places = int(study.get("total_available_places"))
    except (TypeError, ValueError):
        logger.warning(
            "SOMETHING WRONG WITH RESPONSE OF DATA GETTING OF THE STUDY"
        )
        return None

    if num_current_places >= max_users:
        logger.warning(
            f"QUOTA EXCEEDED. Num of current places: {num_current_places}. "
            f"Max users: {max_users}"
        )
        return None

    new_places = num_current_places + num_extra
    url = f"https://api.prolific.co/api/v1/studies/{study_id}/"
    payload = json.dumps({"total_available_places": new_places})

    logger.info(
        f"calling prolific api requesting to increase places to {new_places} "
        f"in study {study_id}"
    )
    logger.info(f"payload: {payload}; url: {url}")

    try:
        resp = requests.patch(url, headers=BASE_HEADERS, data=payload, timeout=10)
    except Exception as e:
        logger.warning(f"Error calling Prolific increase_space({study_id}): {e}")
        return None

    if resp.status_code != 200:
        logger.warning(
            f"Get error trying to increase places in study {study_id}. "
            f"Status code: {resp.status_code}"
        )
        return None

    logger.info(f"response.status_code: {resp.status_code}")
    logger.info(f"response.text: {resp.text}")

    try:
        return resp.json()
    except Exception:
        return None
