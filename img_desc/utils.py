# img_desc/utils.py

import base64
import hashlib
import hmac
import json
import logging
import os

import requests

logger = logging.getLogger("benzapp.utils")

# -------------------------------------------------------------------
#  PROLIFIC CONFIG
# -------------------------------------------------------------------

PROLIFIC_API_KEY = os.environ.get("PROLIFIC_API_KEY")
PROLIFIC_WEBHOOK_SECRET = os.environ.get("PROLIFIC_WEBHOOK_SECRET")

if not PROLIFIC_API_KEY:
    # IMPORTANT: do NOT crash if the key is missing
    logger.warning(
        "PROLIFIC_API_KEY not set; Prolific API features (balance, "
        "increase_space, completion lookup) are disabled."
    )

if not PROLIFIC_WEBHOOK_SECRET:
    logger.warning(
        "PROLIFIC_WEBHOOK_SECRET not set; Prolific webhook verification is disabled."
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
        logger.warning(f"Error getting balance. Status code: {resp.status_code}")
        return None

    try:
        return resp.json()
    except Exception:
        return None


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
            f"Error getting data for study {study_id}. Status code: {resp.status_code}"
        )
        return None

    try:
        return resp.json()
    except Exception:
        return None


def get_completion_info(study_id: str):
    """
    Returns dict(completion_code=..., full_return_url=...) or None.
    If the Prolific API is unavailable, caller should fall back to
    'API_ERROR' codes.
    """
    study_data = get_study(study_id)
    if not study_data:
        logger.warning(f"Failed to get study data for {study_id}")
        return None

    if isinstance(study_data, dict) and study_data.get("error"):
        logger.warning(f"Prolific API returned error for study {study_id}")
        return None

    completion_codes = study_data.get("completion_codes", [])

    completed_code = next(
        (
            cinfo.get("code")
            for cinfo in completion_codes
            if cinfo.get("code_type") == "COMPLETED"
        ),
        "NO_CODE",
    )

    full_return_url = f"{STUBURL}{completed_code}"
    logger.info(f"Completion URL: {full_return_url}; code: {completed_code}")

    return dict(
        completion_code=completed_code,
        full_return_url=full_return_url,
    )


def increase_space(study_id: str, num_extra: int, max_users: int):
    """
    Attempts to increase 'total_available_places' on Prolific.
    Returns JSON response or None.
    """
    if not _can_call_prolific():
        return None

    study = get_study(study_id)
    if not study:
        logger.warning("Could not fetch study data.")
        return None

    try:
        num_current_places = int(study.get("total_available_places"))
    except (TypeError, ValueError):
        logger.warning("Unexpected study response format: total_available_places invalid.")
        return None

    if num_current_places >= max_users:
        logger.warning(
            f"Quota exceeded. Current places: {num_current_places}. Max users: {max_users}"
        )
        return None

    new_places = num_current_places + num_extra
    url = f"https://api.prolific.co/api/v1/studies/{study_id}/"
    payload = json.dumps({"total_available_places": new_places})

    logger.info(f"Increasing Prolific places to {new_places} for study {study_id}")

    try:
        resp = requests.patch(url, headers=BASE_HEADERS, data=payload, timeout=10)
    except Exception as e:
        logger.warning(f"Error calling Prolific increase_space({study_id}): {e}")
        return None

    if resp.status_code != 200:
        logger.warning(
            f"Error increasing places in study {study_id}. Status code: {resp.status_code}"
        )
        logger.warning(f"Response text: {resp.text}")
        return None

    try:
        return resp.json()
    except Exception:
        return None


# -------------------------------------------------------------------
#  PROLIFIC WEBHOOK VERIFICATION
# -------------------------------------------------------------------


def verify_prolific_webhook(raw_body: bytes, timestamp: str, signature: str) -> bool:
    """
    Verify Prolific webhook signature.

    Prolific sends headers:
      - X-Prolific-Request-Timestamp
      - X-Prolific-Request-Signature

    Expected signature:
      base64(hmac_sha256(secret, timestamp + body_as_text))
    """
    if not PROLIFIC_WEBHOOK_SECRET:
        return False

    ts = (timestamp or "").strip()
    sig = (signature or "").strip()
    if not ts or not sig:
        return False

    body_text = raw_body.decode("utf-8")  # should be UTF-8 JSON
    message = (ts + body_text).encode("utf-8")

    digest = hmac.new(
        PROLIFIC_WEBHOOK_SECRET.encode("utf-8"),
        message,
        hashlib.sha256,
    ).digest()

    expected = base64.b64encode(digest).decode("utf-8")
    return hmac.compare_digest(expected, sig)
