
from __future__ import annotations

import os
import logging
import threading
from typing import Optional

import requests

logger = logging.getLogger(__name__)

PROLIFIC_API_BASE = "https://api.prolific.com/api/v1"

# Prevent double-expansion inside ONE Python process (not across multiple dynos).
_expansion_lock = threading.Lock()


def _get_env(name: str) -> str:
    v = (os.environ.get(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _get_study(token: str, study_id: str) -> dict:
    url = f"{PROLIFIC_API_BASE}/studies/{study_id}/"
    headers = {"Authorization": f"Token {token}"}
    resp = requests.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _patch_study(token: str, study_id: str, payload: dict) -> dict:
    url = f"{PROLIFIC_API_BASE}/studies/{study_id}/"
    headers = {"Authorization": f"Token {token}", "Content-Type": "application/json"}
    resp = requests.patch(url, json=payload, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def maybe_expand_slots(enabled: bool, batch_done: bool) -> None:
    """
    Increase Prolific study places when a batch is done.

    Required env vars:
      - PROLIFIC_API_TOKEN
      - PROLIFIC_STUDY_ID

    Optional env vars:
      - PROLIFIC_EXPAND_BY (default: 4)
      - PROLIFIC_MAX_PLACES (optional cap, default: unset)

    Notes:
      - This does NOT control concurrency. Your concurrency is controlled by cohort_size=4 in oTree.
      - Thread-safe only within one process. For multi-worker idempotency, add a DB marker.
    """
    if not (enabled and batch_done):
        return

    # lock is only a best-effort guard inside one process
    got_lock = _expansion_lock.acquire(blocking=False)
    if not got_lock:
        logger.warning("Prolific expansion already in progress in this process; skipping.")
        return

    try:
        token = _get_env("PROLIFIC_API_TOKEN")
        study_id = _get_env("PROLIFIC_STUDY_ID")

        expand_by = int((os.environ.get("PROLIFIC_EXPAND_BY") or "4").strip())
        if expand_by <= 0:
            logger.warning("PROLIFIC_EXPAND_BY <= 0; skipping expansion.")
            return

        max_places_raw = (os.environ.get("PROLIFIC_MAX_PLACES") or "").strip()
        max_places: Optional[int] = int(max_places_raw) if max_places_raw else None

        study = _get_study(token, study_id)

        # Guard: only expand if the study is active (you can relax this if Prolific uses other states)
        status = (study.get("status") or "").strip()
        if status and status != "ACTIVE":
            logger.warning("Study not ACTIVE (status=%s); skipping expansion.", status)
            return

        current = int(study.get("total_available_places") or 0)
        new_total = current + expand_by

        if max_places is not None:
            if current >= max_places:
                logger.info("Already at/above PROLIFIC_MAX_PLACES=%s; skipping.", max_places)
                return
            new_total = min(new_total, max_places)

        updated = _patch_study(token, study_id, {"total_available_places": new_total})
        updated_total = int(updated.get("total_available_places") or new_total)

        logger.info("Expanded Prolific places: study_id=%s %s -> %s", study_id, current, updated_total)

    except RuntimeError as e:
        # Missing env var, etc. Keep the app running.
        logger.error("Prolific expansion config error: %s", e)
    except requests.HTTPError as e:
        body = getattr(e.response, "text", "") if getattr(e, "response", None) else ""
        logger.error("Prolific expansion HTTP error: %s; body=%s", e, body)
    except Exception as e:
        logger.exception("Prolific expansion failed: %s", e)
    finally:
        try:
            _expansion_lock.release()
        except Exception:
            pass
