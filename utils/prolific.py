from typing import Optional
import os
import logging
import threading
import requests

logger = logging.getLogger(__name__)

PROLIFIC_API_BASE = "https://api.prolific.com/api/v1"

# Module-level lock to prevent race conditions when multiple workers
# trigger expansion simultaneously.
_expansion_lock = threading.Lock()


def _get_env(name: str) -> str:
    v = (os.environ.get(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _get_study(token: str, study_id: str) -> dict:
    """GET /api/v1/studies/:id/"""
    url = f"{PROLIFIC_API_BASE}/studies/{study_id}/"
    headers = {"Authorization": f"Token {token}"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _patch_study(token: str, study_id: str, payload: dict) -> dict:
    """PATCH /api/v1/studies/:id/"""
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
      - PROLIFIC_API_TOKEN : your Prolific API token (researcher account)
      - PROLIFIC_STUDY_ID  : the Prolific study ID to expand

    Optional env vars:
      - PROLIFIC_EXPAND_BY : how many places to add each time (default: 4)

    Notes:
      - Only expands if the study is in ACTIVE status.
      - Thread-safe: uses a lock to prevent double-expansion from concurrent workers.
      - Prolific only allows *increasing* total_available_places on a published study.
    """
    if not (enabled and batch_done):
        return

    # Acquire lock to prevent concurrent expansions (race condition fix)
    if not _expansion_lock.acquire(blocking=False):
        logger.warning("Prolific expansion already in progress, skipping this call.")
        return

    try:
        token = _get_env("PROLIFIC_API_TOKEN")
        study_id = _get_env("PROLIFIC_STUDY_ID")
        expand_by = int((os.environ.get("PROLIFIC_EXPAND_BY") or "4").strip())

        if expand_by <= 0:
            logger.warning("PROLIFIC_EXPAND_BY <= 0, skipping expansion.")
            return

        study = _get_study(token, study_id)

        # Guard: only expand if the study is actively running
        status = study.get("status")
        if status != "ACTIVE":
            logger.warning(
                "Prolific study %s is not ACTIVE (status=%s), skipping expansion.",
                study_id,
                status,
            )
            return

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
        body = getattr(e.response, "text", "") if getattr(e, "response", None) else ""
        logger.error(
            "Prolific expansion HTTP error: %s; body=%s", e, body
        )
    except RuntimeError as e:
        logger.error("Prolific expansion config error: %s", e)
    except Exception as e:
        logger.exception("Prolific expansion failed: %s", e)
    finally:
        _expansion_lock.release()
