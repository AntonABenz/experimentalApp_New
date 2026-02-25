from typing import Optional
import os
import logging
import requests

logger = logging.getLogger(__name__)

PROLIFIC_API_BASE = "https://api.prolific.com/api/v1"

def _get_env(name: str) -> str:
    v = (os.environ.get(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _get_study(token: str, study_id: str) -> dict:
    # GET https://api.prolific.com/api/v1/studies/:id/
    # :contentReference[oaicite:1]{index=1}
    url = f"{PROLIFIC_API_BASE}/studies/{study_id}/"
    headers = {"Authorization": f"Token {token}"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _patch_study(token: str, study_id: str, payload: dict) -> dict:
    # PATCH https://api.prolific.com/api/v1/studies/:id/
    # :contentReference[oaicite:2]{index=2}
    url = f"{PROLIFIC_API_BASE}/studies/{study_id}/"
    headers = {
        "Authorization": f"Token {token}",
        "Content-Type": "application/json",
    }
    resp = requests.patch(url, json=payload, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def maybe_expand_slots(enabled: bool, batch_done: bool) -> None:
    """
    Increase Prolific study places when a batch is done.

    Required env vars:
      - PROLIFIC_API_TOKEN: your Prolific API token (researcher account)
      - PROLIFIC_STUDY_ID: the Prolific study ID to expand

    Optional env vars:
      - PROLIFIC_EXPAND_BY: how many places to add each time (default 4)

    Notes:
      - Prolific allows increasing total_available_places on a published study (not decreasing).
        :contentReference[oaicite:3]{index=3}
      - This function is NOT idempotent: if you call it twice for the same "batch_done",
        it will add places twice. Ideally call it only once per batch completion.
    """
    if not (enabled and batch_done):
        return

    try:
        token = _get_env("PROLIFIC_API_TOKEN")
        study_id = _get_env("PROLIFIC_STUDY_ID")
        expand_by = int((os.environ.get("PROLIFIC_EXPAND_BY") or "4").strip())
        if expand_by <= 0:
            logger.warning("PROLIFIC_EXPAND_BY <= 0, skipping expansion.")
            return

        study = _get_study(token, study_id)
        current = int(study.get("total_available_places") or 0)
        new_total = current + expand_by

        updated = _patch_study(token, study_id, {"total_available_places": new_total})
        logger.info(
            "Expanded Prolific study places: study_id=%s %s -> %s",
            study_id,
            current,
            int(updated.get("total_available_places") or new_total),
        )

    except requests.HTTPError as e:
        # Helpful error body for debugging
        body = getattr(e.response, "text", "") if getattr(e, "response", None) else ""
        logger.error("Prolific expansion HTTP error: %s; body=%s", e, body)
    except Exception as e:
        logger.exception("Prolific expansion failed: %s", e)
