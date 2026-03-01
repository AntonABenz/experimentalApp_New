from __future__ import annotations

import os
import logging
import threading
from typing import Optional, Tuple

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


def _auth_headers(token: str) -> dict:
    return {"Authorization": f"Token {token}"}


def _get_study(token: str, study_id: str) -> dict:
    url = f"{PROLIFIC_API_BASE}/studies/{study_id}/"
    resp = requests.get(url, headers=_auth_headers(token), timeout=20)
    resp.raise_for_status()
    return resp.json()


def _patch_study(token: str, study_id: str, payload: dict) -> dict:
    url = f"{PROLIFIC_API_BASE}/studies/{study_id}/"
    headers = {**_auth_headers(token), "Content-Type": "application/json"}
    resp = requests.patch(url, json=payload, headers=headers, timeout=20)
    resp.raise_for_status()
    return resp.json()


def _allowed_statuses() -> set[str]:
    """
    Optional env var:
      - PROLIFIC_ALLOWED_STATUSES (default: "ACTIVE")
        Example: "ACTIVE,PAUSED"
    """
    raw = (os.environ.get("PROLIFIC_ALLOWED_STATUSES") or "ACTIVE").strip()
    parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
    return set(parts) if parts else {"ACTIVE"}


def maybe_expand_slots(enabled: bool, batch_done: bool) -> Tuple[bool, str]:
    """
    Increase Prolific study places when a batch is done.

    Required env vars:
      - PROLIFIC_API_TOKEN
      - PROLIFIC_STUDY_ID

    Optional env vars:
      - PROLIFIC_EXPAND_BY (default: 4)
      - PROLIFIC_MAX_PLACES (optional cap, default: unset)
      - PROLIFIC_ALLOWED_STATUSES (default: "ACTIVE") e.g. "ACTIVE,PAUSED"

    Returns:
      (ok, message)
    """
    if not (enabled and batch_done):
        return False, "disabled_or_batch_not_done"

    got_lock = _expansion_lock.acquire(blocking=False)
    if not got_lock:
        logger.warning("Prolific expansion already in progress in this process; skipping.")
        return False, "lock_busy"

    try:
        token = _get_env("PROLIFIC_API_TOKEN")
        study_id = _get_env("PROLIFIC_STUDY_ID")

        expand_by_raw = (os.environ.get("PROLIFIC_EXPAND_BY") or "4").strip()
        expand_by = int(expand_by_raw)
        if expand_by <= 0:
            logger.warning("PROLIFIC_EXPAND_BY <= 0; skipping expansion.")
            return False, "expand_by_non_positive"

        max_places_raw = (os.environ.get("PROLIFIC_MAX_PLACES") or "").strip()
        max_places: Optional[int] = int(max_places_raw) if max_places_raw else None

        study = _get_study(token, study_id)

        status = (study.get("status") or "").strip().upper()
        allowed = _allowed_statuses()
        if status and status not in allowed:
            logger.warning("Study status=%s not in allowed=%s; skipping expansion.", status, sorted(allowed))
            return False, f"status_not_allowed:{status}"

        current = int(study.get("total_available_places") or 0)
        new_total = current + expand_by

        if max_places is not None:
            if current >= max_places:
                logger.info("Already at/above PROLIFIC_MAX_PLACES=%s; skipping.", max_places)
                return False, "at_or_above_max_places"
            new_total = min(new_total, max_places)

        if new_total == current:
            logger.info("No change in places (current=%s new_total=%s); skipping.", current, new_total)
            return False, "no_change"

        logger.info(
            "Attempting Prolific expansion: study_id=%s status=%s current=%s expand_by=%s new_total=%s",
            study_id, status, current, expand_by, new_total
        )

        updated = _patch_study(token, study_id, {"total_available_places": new_total})
        updated_total = int(updated.get("total_available_places") or new_total)

        logger.info("Expanded Prolific places: study_id=%s %s -> %s", study_id, current, updated_total)
        return True, f"expanded:{current}->{updated_total}"

    except RuntimeError as e:
        logger.error("Prolific expansion config error: %s", e)
        return False, f"config_error:{e}"
    except requests.HTTPError as e:
        body = getattr(e.response, "text", "") if getattr(e, "response", None) else ""
        logger.error("Prolific expansion HTTP error: %s; body=%s", e, body)
        return False, f"http_error:{e}"
    except Exception as e:
        logger.exception("Prolific expansion failed: %s", e)
        return False, f"exception:{e}"
    finally:
        try:
            _expansion_lock.release()
        except Exception:
            pass
