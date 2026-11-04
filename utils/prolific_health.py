"""Prolific connection status for the admin dashboard.

Answers three questions without showing any secret:
  - are the Prolific API token and the webhook secret set (yes / no only)?
  - when did Prolific last call the webhook, and did the signature match?
  - why did the last automatic increase of Prolific places fail?

Webhook calls are noted in a small table (one row per kind). Everything here is
best effort: a database problem must never break the webhook or the dashboard.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timezone

logger = logging.getLogger("benzapp.prolific_health")

TABLE = "public.prolific_webhook_seen"
KIND_OK = "signature_ok"
KIND_BAD_SIGNATURE = "bad_signature"


def _database_url() -> str:
    url = (os.environ.get("DATABASE_URL") or os.environ.get("OTREE_DB_URL") or "").strip()
    return url if url.startswith("postgres") else ""


def _connect():
    url = _database_url()
    if not url:
        return None
    import psycopg2

    return psycopg2.connect(url)


def _ensure_table(cur):
    cur.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            kind TEXT PRIMARY KEY,
            ts DOUBLE PRECISION NOT NULL,
            status TEXT NOT NULL DEFAULT '',
            note TEXT NOT NULL DEFAULT '',
            count INTEGER NOT NULL DEFAULT 0
        )
        """
    )


def record_webhook(kind: str, status: str = "", note: str = "") -> None:
    """Note that Prolific called the webhook (kind = KIND_OK or KIND_BAD_SIGNATURE)."""
    conn = None
    try:
        conn = _connect()
        if conn is None:
            return
        with conn.cursor() as cur:
            _ensure_table(cur)
            cur.execute(
                f"""
                INSERT INTO {TABLE} (kind, ts, status, note, count) VALUES (%s, %s, %s, %s, 1)
                ON CONFLICT (kind) DO UPDATE
                SET ts = EXCLUDED.ts, status = EXCLUDED.status, note = EXCLUDED.note, count = {TABLE}.count + 1
                """,
                (kind, time.time(), (status or "")[:40], (note or "")[:80]),
            )
        conn.commit()
    except Exception:
        logger.warning("Could not note the webhook call", exc_info=True)
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _seen_webhooks() -> dict:
    conn = None
    try:
        conn = _connect()
        if conn is None:
            return {}
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (TABLE,))
            if not cur.fetchone()[0]:
                return {}
            cur.execute(f"SELECT kind, ts, status, note, count FROM {TABLE}")
            return {
                row[0]: dict(ts=float(row[1] or 0), status=row[2] or "", note=row[3] or "", count=int(row[4] or 0))
                for row in cur.fetchall()
            }
    except Exception:
        logger.warning("Could not read the webhook calls", exc_info=True)
        return {}
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def _when(ts: float) -> str:
    if not ts:
        return ""
    minutes = int(max(0, time.time() - ts) // 60)
    if minutes < 1:
        ago = "just now"
    elif minutes < 120:
        ago = f"{minutes} min ago"
    elif minutes < 48 * 60:
        ago = f"{minutes // 60} h ago"
    else:
        ago = f"{minutes // (24 * 60)} days ago"
    stamp = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%d %b %H:%M UTC")
    return f"{stamp} ({ago})"


def explain_expansion_message(message: str) -> str:
    """Plain words for the reason saved with the last place-increase attempt."""
    m = (message or "").strip()
    if not m:
        return ""
    if m.startswith("expanded:"):
        before, _, after = m[len("expanded:"):].partition("->")
        return f"places increased from {before} to {after}"
    if m.startswith("config_error:"):
        if "PROLIFIC_API_TOKEN" in m or "PROLIFIC_API_KEY" in m:
            return "no Prolific API token on the server"
        return "settings missing on the server (" + m[len("config_error:"):].strip() + ")"
    if m.startswith("status_not_allowed:"):
        return "the Prolific study was not active (status " + m.split(":", 1)[1].strip() + ")"
    if m.startswith("http_error:"):
        detail = m.split(":", 1)[1].strip()
        if detail.startswith(("401", "403")):
            return "Prolific refused the API token (" + detail[:3] + ")"
        if detail.startswith("404"):
            return "Prolific does not know this study id (404)"
        return "Prolific answered with an error (" + detail[:60] + ")"
    if m.startswith("study_id_unavailable"):
        return "the participants had no Prolific study id"
    if m == "at_or_above_max_places":
        return "already at the maximum number of places (PROLIFIC_MAX_PLACES)"
    if m == "attempt_started":
        return "the attempt stopped before it finished"
    if m == "no_next_experiment":
        return "no further cohort to recruit for"
    return m[:80]


def prolific_health(expansion_message: str = "") -> dict:
    seen = _seen_webhooks()
    ok = seen.get(KIND_OK, {})
    bad = seen.get(KIND_BAD_SIGNATURE, {})
    return dict(
        token_set=bool((os.environ.get("PROLIFIC_API_TOKEN") or os.environ.get("PROLIFIC_API_KEY") or "").strip()),
        secret_set=bool((os.environ.get("PROLIFIC_WEBHOOK_SECRET") or "").strip()),
        last_ok=_when(ok.get("ts", 0)),
        last_ok_status=ok.get("status", ""),
        ok_count=ok.get("count", 0),
        last_bad=_when(bad.get("ts", 0)),
        bad_count=bad.get("count", 0),
        expansion_reason=explain_expansion_message(expansion_message),
    )
