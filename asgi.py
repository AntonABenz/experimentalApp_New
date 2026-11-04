# asgi.py
import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse, RedirectResponse, PlainTextResponse
from starlette.requests import Request

from otree.asgi import app as otree_app
from otree.database import db
from otree.middleware import lock2 as _otree_request_lock
from prolific_webhook import prolific_webhook_view, _find_participant_by_prolific_id
from start import Player as StartPlayer
import psycopg2

# NEW
from otree.models import Participant, Session
from otree.database import db as otree_db
from otree.middleware import lock2 as otree_lock2
from img_desc import (
    CohortSlot,
    Constants as ImgDescConstants,
    Player as ImgDescPlayer,
    ScheduleItem,
    _current_exp_num_for_participant,
    _get_expansion_state,
    clean_str,
    cohort_complete,
    cohort_debug_snapshot,
    experiment_exists,
    find_prolific_slot_map,
    free_slot_from_prolific_slot_map,
    free_slot_for_participant,
    get_cohort_snapshot_data,
    get_participant_prolific_id,
    get_participant_slot_rows,
    get_schedule_item,
    get_participant_status,
    mark_finished_from_prolific_slot_map,
    mark_participant_active,
    mark_participant_complete_in_cohort,
    mark_participant_drop_out,
    mark_participant_finished,
    maybe_expand_prolific_when_cohort_complete,
    maybe_expand_prolific_for_participant,
    _recent_root_subsessions,
    reset_this_app_for_participant,
    safe_int,
)

logger = logging.getLogger("benzapp.admin")
PROLIFIC_CAPTURE_COOKIE = "prolific_capture"
PROLIFIC_CAPTURE_MAX_AGE = 6 * 60 * 60
_start_prolific_intake_table_available_cache = None

def _with_db_transaction(view):
    """Run a route that lives outside oTree's own app the way oTree runs its pages.

    oTree 5 has no model.save(): changes are written when oTree's
    CommitTransactionMiddleware commits at the end of a request, and the next
    request replaces the DB session, dropping anything left uncommitted. Routes
    mounted next to otree_app never pass through that middleware, so commit here
    (under the same lock) or their changes are lost.
    """

    async def wrapped(request):
        async with _otree_request_lock:
            db.new_session()
            try:
                response = await view(request)
            except Exception:
                db.rollback()
                raise
            if response.status_code < 500:
                db.commit()
            else:
                db.rollback()
            return response

    return wrapped


async def ping(request):
    return JSONResponse({"ok": True})


def _cookie_secret() -> bytes:
    return (os.environ.get("OTREE_SECRET_KEY") or "dev-secret").encode("utf-8")


def _clean_cookie_value(value) -> str:
    return (value or "").strip()


def _normalize_prolific_identity_pair(prolific_id: str, participant_label: str) -> tuple[str, str, bool]:
    prolific_id = _clean_cookie_value(prolific_id)
    participant_label = _clean_cookie_value(participant_label)
    mismatch = bool(prolific_id and participant_label and prolific_id != participant_label)
    if not prolific_id:
        prolific_id = participant_label
    if not participant_label:
        participant_label = prolific_id
    return prolific_id, participant_label, mismatch


def _sign_cookie_payload(payload: dict) -> str:
    raw = json.dumps(payload or {}, separators=(",", ":"), sort_keys=True)
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
    signature = hmac.new(_cookie_secret(), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"{encoded}.{signature}"


def _load_signed_cookie(value: str) -> dict:
    value = _clean_cookie_value(value)
    if "." not in value:
        return {}

    encoded, signature = value.rsplit(".", 1)
    expected = hmac.new(_cookie_secret(), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    if not secrets.compare_digest(signature, expected):
        return {}

    try:
        padded = encoded + ("=" * (-len(encoded) % 4))
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        payload = json.loads(raw or "{}")
    except Exception:
        return {}

    return payload if isinstance(payload, dict) else {}


def _extract_prolific_cookie_payload(request: Request) -> dict:
    prolific_id_raw = _clean_cookie_value(
        request.query_params.get("PROLIFIC_PID")
        or request.query_params.get("prolific_pid")
        or request.query_params.get("prolific_id")
        or request.query_params.get("participant_id")
    )
    participant_label_raw = _clean_cookie_value(
        request.query_params.get("participant_label")
        or request.query_params.get("PARTICIPANT_LABEL")
    )
    prolific_id, participant_label, mismatch = _normalize_prolific_identity_pair(
        prolific_id_raw,
        participant_label_raw,
    )
    return {
        "prolific_id": prolific_id,
        "participant_label": participant_label,
        "study_id": _clean_cookie_value(
            request.query_params.get("STUDY_ID")
            or request.query_params.get("study_id")
        ),
        "session_id": _clean_cookie_value(
            request.query_params.get("SESSION_ID")
            or request.query_params.get("session_id")
        ),
        "identity_mismatch": mismatch,
        "raw_prolific_id": prolific_id_raw,
        "raw_participant_label": participant_label_raw,
    }


def reset_start_prolific_intake_cache():
    """Forget a previous availability result so the next call re-checks."""
    global _start_prolific_intake_table_available_cache
    _start_prolific_intake_table_available_cache = None


def _start_prolific_intake_table_available() -> bool:
    global _start_prolific_intake_table_available_cache
    # Only a confirmed-available result is cached. A failure (no DB, a
    # connection blip, a create error) is never cached, so the next call
    # retries and the table can self-heal instead of staying disabled for the
    # life of the process.
    if _start_prolific_intake_table_available_cache:
        return True

    database_url = _clean_cookie_value(os.environ.get("DATABASE_URL", "") or os.environ.get("OTREE_DB_URL", ""))
    if not database_url:
        return False

    try:
        conn = psycopg2.connect(database_url)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.start_prolificintake')")
                table_exists = bool(cur.fetchone()[0])
                if not table_exists:
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS public.start_prolificintake (
                            id BIGSERIAL PRIMARY KEY,
                            participant_code TEXT NOT NULL UNIQUE,
                            session_code TEXT NOT NULL DEFAULT '',
                            prolific_pid TEXT NOT NULL DEFAULT '',
                            participant_label TEXT NOT NULL DEFAULT '',
                            study_id TEXT NOT NULL DEFAULT '',
                            session_id TEXT NOT NULL DEFAULT '',
                            last_seen_ts DOUBLE PRECISION NOT NULL DEFAULT 0
                        )
                        """
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_start_pi_participant ON public.start_prolificintake (participant_code)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_start_pi_pid ON public.start_prolificintake (prolific_pid)"
                    )
                    cur.execute(
                        "CREATE INDEX IF NOT EXISTS idx_start_pi_session ON public.start_prolificintake (session_code)"
                    )
                    conn.commit()
                    logger.warning("ASGI StartProlificIntake bootstrap: created table start_prolificintake")
                    table_exists = True
                if table_exists:
                    _start_prolific_intake_table_available_cache = True
                return bool(table_exists)
        finally:
            conn.close()
    except Exception as e:
        logger.warning("ASGI StartProlificIntake availability check failed: %s", e)
        return False


def _is_missing_intake_table_error(exc: Exception) -> bool:
    """True when a query failed because start_prolificintake is gone (e.g. a
    database reset dropped it while the availability cache still said True)."""
    message = _clean_cookie_value(str(exc)).lower()
    return "start_prolificintake" in message and "does not exist" in message


def _recreate_intake_table_after_missing() -> bool:
    """Forget the stale availability result and re-check, recreating the table
    when it is missing."""
    reset_start_prolific_intake_cache()
    return _start_prolific_intake_table_available()


def _sync_start_prolific_intake_from_participant(participant, payload: dict, _retry: bool = True) -> bool:
    if participant is None or not _start_prolific_intake_table_available():
        return False

    participant_code = _clean_cookie_value(getattr(participant, "code", ""))
    session_code = _clean_cookie_value(getattr(getattr(participant, "session", None), "code", ""))
    payload_prolific_id = payload.get("prolific_id") or getattr(participant, "label", "") or participant.vars.get("prolific_id")
    payload_participant_label = payload.get("participant_label") or payload_prolific_id
    prolific_id, participant_label, _ = _normalize_prolific_identity_pair(
        payload_prolific_id,
        payload_participant_label,
    )
    study_id = _clean_cookie_value(payload.get("study_id") or participant.vars.get("study_id"))
    session_id = _clean_cookie_value(payload.get("session_id") or participant.vars.get("prolific_session_id"))

    if not participant_code or not prolific_id:
        return False

    database_url = _clean_cookie_value(os.environ.get("DATABASE_URL", "") or os.environ.get("OTREE_DB_URL", ""))
    if not database_url:
        return False

    try:
        conn = psycopg2.connect(database_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.start_prolificintake
                        (participant_code, session_code, prolific_pid, participant_label, study_id, session_id, last_seen_ts)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (participant_code) DO UPDATE
                    SET session_code = COALESCE(NULLIF(EXCLUDED.session_code, ''), public.start_prolificintake.session_code),
                        prolific_pid = COALESCE(NULLIF(EXCLUDED.prolific_pid, ''), public.start_prolificintake.prolific_pid),
                        participant_label = COALESCE(NULLIF(EXCLUDED.participant_label, ''), public.start_prolificintake.participant_label),
                        study_id = COALESCE(NULLIF(EXCLUDED.study_id, ''), public.start_prolificintake.study_id),
                        session_id = COALESCE(NULLIF(EXCLUDED.session_id, ''), public.start_prolificintake.session_id),
                        last_seen_ts = EXCLUDED.last_seen_ts
                    """,
                    (participant_code, session_code, prolific_id, participant_label, study_id, session_id, time.time()),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        if _retry and _is_missing_intake_table_error(e):
            logger.warning("ASGI sync_start_prolific_intake: table missing, recreating and retrying participant=%s", participant_code)
            _recreate_intake_table_after_missing()
            return _sync_start_prolific_intake_from_participant(participant, payload, _retry=False)
        logger.warning(
            "ASGI sync_start_prolific_intake failed: participant=%s session=%s prolific_id=%s study_id=%s session_id=%s error=%s",
            participant_code,
            session_code,
            prolific_id,
            study_id,
            session_id,
            e,
        )
        return False

    logger.warning(
        "ASGI sync_start_prolific_intake: participant=%s session=%s prolific_id=%s participant_label=%s study_id=%s session_id=%s",
        participant_code,
        session_code,
        prolific_id,
        participant_label,
        study_id,
        session_id,
    )
    return True


def _join_session_code_from_path(path: str) -> str:
    parts = [p for p in (path or "").split("/") if p]
    if len(parts) >= 2 and parts[0] == "join":
        return _clean_cookie_value(parts[1])
    return ""


def _pending_start_intake_code(payload: dict) -> str:
    prolific_id = _clean_cookie_value(payload.get("prolific_id"))
    study_id = _clean_cookie_value(payload.get("study_id"))
    session_id = _clean_cookie_value(payload.get("session_id"))
    if not prolific_id:
        return ""
    return f"pending:{prolific_id}:{study_id}:{session_id}"


def _sync_start_prolific_intake_pending(session_code: str, payload: dict, _retry: bool = True) -> bool:
    if not _start_prolific_intake_table_available():
        return False

    pending_code = _pending_start_intake_code(payload)
    prolific_id, participant_label, _ = _normalize_prolific_identity_pair(
        payload.get("prolific_id"),
        payload.get("participant_label"),
    )
    study_id = _clean_cookie_value(payload.get("study_id"))
    session_id = _clean_cookie_value(payload.get("session_id"))
    session_code = _clean_cookie_value(session_code)

    if not pending_code or not prolific_id:
        return False

    database_url = _clean_cookie_value(os.environ.get("DATABASE_URL", "") or os.environ.get("OTREE_DB_URL", ""))
    if not database_url:
        return False

    try:
        conn = psycopg2.connect(database_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO public.start_prolificintake
                        (participant_code, session_code, prolific_pid, participant_label, study_id, session_id, last_seen_ts)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (participant_code) DO UPDATE
                    SET session_code = COALESCE(NULLIF(EXCLUDED.session_code, ''), public.start_prolificintake.session_code),
                        prolific_pid = COALESCE(NULLIF(EXCLUDED.prolific_pid, ''), public.start_prolificintake.prolific_pid),
                        participant_label = COALESCE(NULLIF(EXCLUDED.participant_label, ''), public.start_prolificintake.participant_label),
                        study_id = COALESCE(NULLIF(EXCLUDED.study_id, ''), public.start_prolificintake.study_id),
                        session_id = COALESCE(NULLIF(EXCLUDED.session_id, ''), public.start_prolificintake.session_id),
                        last_seen_ts = EXCLUDED.last_seen_ts
                    """,
                    (pending_code, session_code, prolific_id, participant_label, study_id, session_id, time.time()),
                )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        if _retry and _is_missing_intake_table_error(e):
            logger.warning("ASGI sync_start_prolific_intake pending: table missing, recreating and retrying session=%s", session_code)
            _recreate_intake_table_after_missing()
            return _sync_start_prolific_intake_pending(session_code, payload, _retry=False)
        logger.warning(
            "ASGI sync_start_prolific_intake pending failed: pending=%s session=%s prolific_id=%s study_id=%s session_id=%s error=%s",
            pending_code,
            session_code,
            prolific_id,
            study_id,
            session_id,
            e,
        )
        return False

    logger.warning(
        "ASGI sync_start_prolific_intake pending: pending=%s session=%s prolific_id=%s participant_label=%s study_id=%s session_id=%s",
        pending_code,
        session_code,
        prolific_id,
        participant_label,
        study_id,
        session_id,
    )
    return True


def _bind_start_prolific_intake_participant_code(participant_code: str, payload: dict, _retry: bool = True) -> bool:
    if not _start_prolific_intake_table_available():
        return False

    participant_code = _clean_cookie_value(participant_code)
    pending_code = _pending_start_intake_code(payload)
    prolific_id = _clean_cookie_value(payload.get("prolific_id"))
    study_id = _clean_cookie_value(payload.get("study_id"))
    session_id = _clean_cookie_value(payload.get("session_id"))

    if not participant_code or not pending_code or not prolific_id:
        return False

    database_url = _clean_cookie_value(os.environ.get("DATABASE_URL", "") or os.environ.get("OTREE_DB_URL", ""))
    if not database_url:
        return False

    try:
        conn = psycopg2.connect(database_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT participant_code, session_code, prolific_pid, participant_label, study_id, session_id
                    FROM public.start_prolificintake
                    WHERE participant_code = %s
                    LIMIT 1
                    """,
                    (pending_code,),
                )
                pending_row = cur.fetchone()
                if not pending_row:
                    return False

                cur.execute(
                    """
                    SELECT participant_code, session_code, prolific_pid, participant_label, study_id, session_id
                    FROM public.start_prolificintake
                    WHERE participant_code = %s
                    LIMIT 1
                    """,
                    (participant_code,),
                )
                actual_row = cur.fetchone()

                if actual_row:
                    cur.execute(
                        """
                        UPDATE public.start_prolificintake
                        SET session_code = %s,
                            prolific_pid = %s,
                            participant_label = %s,
                            study_id = %s,
                            session_id = %s,
                            last_seen_ts = %s
                        WHERE participant_code = %s
                        """,
                        (
                            _clean_cookie_value(actual_row[1]) or _clean_cookie_value(pending_row[1]),
                            _clean_cookie_value(actual_row[2]) or _clean_cookie_value(pending_row[2]) or prolific_id,
                            _clean_cookie_value(actual_row[3]) or _clean_cookie_value(pending_row[3]) or prolific_id,
                            _clean_cookie_value(actual_row[4]) or _clean_cookie_value(pending_row[4]) or study_id,
                            _clean_cookie_value(actual_row[5]) or _clean_cookie_value(pending_row[5]) or session_id,
                            time.time(),
                            participant_code,
                        ),
                    )
                    cur.execute("DELETE FROM public.start_prolificintake WHERE participant_code = %s", (pending_code,))
                else:
                    cur.execute(
                        """
                        UPDATE public.start_prolificintake
                        SET participant_code = %s,
                            last_seen_ts = %s
                        WHERE participant_code = %s
                        """,
                        (participant_code, time.time(), pending_code),
                    )
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        if _retry and _is_missing_intake_table_error(e):
            logger.warning("ASGI bind_start_prolific_intake: table missing, recreating and retrying participant=%s", participant_code)
            _recreate_intake_table_after_missing()
            return _bind_start_prolific_intake_participant_code(participant_code, payload, _retry=False)
        logger.warning(
            "ASGI bind_start_prolific_intake failed: participant=%s pending=%s prolific_id=%s study_id=%s session_id=%s error=%s",
            participant_code,
            pending_code,
            prolific_id,
            study_id,
            session_id,
            e,
        )
        return False

    logger.warning(
        "ASGI bind_start_prolific_intake: participant=%s pending=%s prolific_id=%s study_id=%s session_id=%s",
        participant_code,
        pending_code,
        prolific_id,
        study_id,
        session_id,
    )
    return True


def _participant_code_from_path(path: str) -> str:
    parts = [p for p in (path or "").split("/") if p]
    if len(parts) >= 2 and parts[0] == "InitializeParticipant":
        return parts[1]
    if len(parts) >= 2 and parts[0] == "p":
        return parts[1]
    return ""


def _find_participant_by_code(participant_code: str):
    participant_code = _clean_cookie_value(participant_code)
    if not participant_code:
        return None

    try:
        sessions = Session.objects_filter().order_by(Session.id.desc()).limit(200).all()
    except Exception:
        sessions = []

    for session in sessions:
        try:
            for participant in session.get_participants():
                if _clean_cookie_value(getattr(participant, "code", "")) == participant_code:
                    logger.warning("CohortRepair code lookup: matched via session.get_participants participant=%s", participant_code)
                    return participant
        except Exception:
            continue

    # oTree 5 has no Model.objects (that was the Django-era API and caused 500s on
    # /InitializeParticipant). Find the participant through the concrete app Player
    # models, joining to Participant to match on its code.
    for model in (StartPlayer, ImgDescPlayer):
        try:
            player = (
                model.objects_filter()
                .join(Participant, model.participant_id == Participant.id)
                .filter(Participant.code == participant_code)
                .order_by(model.id.desc())
                .first()
            )
            if player and getattr(player, "participant", None):
                logger.warning(
                    "CohortRepair code lookup: matched via %s relation filter participant=%s",
                    getattr(model, "__name__", str(model)),
                    participant_code,
                )
                return player.participant
        except Exception:
            continue

    mapping = find_prolific_slot_map(participant_code=participant_code, prefer_active=False)
    if mapping:
        participant = _find_participant_in_session(
            clean_str(getattr(mapping, "session_code", "")),
            participant_code=participant_code,
        )
        if participant:
            logger.warning("CohortRepair code lookup: matched via slot-map session fallback participant=%s", participant_code)
            return participant

    # Compatibility fallback for deployed runtimes where relation filters can
    # miss existing participants. Scan recent concrete Player rows and compare
    # the resolved participant object directly.
    for model in (ImgDescPlayer, StartPlayer):
        try:
            recent_players = model.objects_filter().order_by(model.id.desc()).limit(4000).all()
        except Exception:
            recent_players = []

        for player in recent_players:
            try:
                participant = getattr(player, "participant", None)
                if participant and _clean_cookie_value(getattr(participant, "code", "")) == participant_code:
                    logger.warning(
                        "CohortRepair code lookup: matched via %s recent player scan participant=%s",
                        getattr(model, "__name__", str(model)),
                        participant_code,
                    )
                    return participant
            except Exception:
                continue

    logger.warning("CohortRepair code lookup: no participant object match participant=%s", participant_code)
    return None


def _find_participant_in_session(session_code: str, participant_code: str = "", prolific_id: str = ""):
    session_code = _clean_cookie_value(session_code)
    participant_code = _clean_cookie_value(participant_code)
    prolific_id = _clean_cookie_value(prolific_id)
    if not session_code:
        return None

    session = _find_session_by_code(session_code)
    if not session:
        return None

    try:
        for participant in session.get_participants():
            if participant_code and _clean_cookie_value(getattr(participant, "code", "")) == participant_code:
                return participant
            if prolific_id and _clean_cookie_value(get_participant_prolific_id(participant)) == prolific_id:
                return participant
    except Exception:
        return None

    return None


def _apply_prolific_cookie_to_participant(participant, payload: dict) -> bool:
    prolific_id = _clean_cookie_value(payload.get("prolific_id"))
    study_id = _clean_cookie_value(payload.get("study_id"))
    session_id = _clean_cookie_value(payload.get("session_id"))

    if not prolific_id:
        return False

    changed = False
    if participant.vars.get("prolific_id") != prolific_id:
        participant.vars["prolific_id"] = prolific_id
        changed = True

    # Stable identifier semantics, do not repurpose:
    # participant.label is reserved here as a fallback copy of the Prolific PID.
    # It must not be reused for spreadsheet participant ids or status labels.
    if getattr(participant, "label", "") != prolific_id:
        participant.label = prolific_id
        changed = True

    if study_id and participant.vars.get("study_id") != study_id:
        participant.vars["study_id"] = study_id
        changed = True

    if session_id and participant.vars.get("prolific_session_id") != session_id:
        participant.vars["prolific_session_id"] = session_id
        changed = True

    current_status = _clean_cookie_value(participant.vars.get("participant_status")).lower()
    if current_status not in {"finished", "drop_out"}:
        participant.vars["participant_status"] = "active"
        changed = True

    if changed:
        logger.info(
            "Prolific cookie captured: participant=%s prolific_id=%s study_id=%s session_id=%s",
            participant.code,
            prolific_id,
            study_id,
            session_id,
        )
    return True


async def _capture_prolific_cookie_for_participant(participant_code: str, payload: dict) -> bool:
    """Apply the captured Prolific ids to the participant and commit.

    This middleware runs outside oTree's commit middleware, so commit here (see
    _with_db_transaction). Best effort: a failure is logged and must not turn the
    participant's page into a server error.
    """
    async with _otree_request_lock:
        db.new_session()
        try:
            _bind_start_prolific_intake_participant_code(participant_code, payload)
            participant = _find_participant_by_code(participant_code)
            applied = False
            if participant:
                synced = _sync_start_prolific_intake_from_participant(participant, payload)
                applied = _apply_prolific_cookie_to_participant(participant, payload)
                applied = bool(synced or applied)
            db.commit()
            return applied
        except Exception:
            db.rollback()
            logger.exception("Prolific cookie capture failed participant=%s", participant_code)
            return False


class ProlificCaptureCookieMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        should_clear_cookie = False
        payload = _load_signed_cookie(request.cookies.get(PROLIFIC_CAPTURE_COOKIE, ""))
        request_payload = _extract_prolific_cookie_payload(request)
        participant_code = _participant_code_from_path(request.url.path)
        join_session_code = _join_session_code_from_path(request.url.path)

        if request_payload.get("identity_mismatch") and join_session_code:
            logger.warning(
                "Rejected mismatched Prolific join: session=%s raw_prolific_id=%s raw_participant_label=%s study_id=%s session_id=%s path=%s",
                join_session_code,
                request_payload.get("raw_prolific_id", ""),
                request_payload.get("raw_participant_label", ""),
                request_payload.get("study_id", ""),
                request_payload.get("session_id", ""),
                request.url.path,
            )
            return PlainTextResponse(
                "We could not safely admit you to the experiment because the incoming participant identifiers were inconsistent. Please return the study and contact the researcher.",
                status_code=400,
            )

        if request_payload.get("prolific_id") and join_session_code:
            _sync_start_prolific_intake_pending(join_session_code, request_payload)

        if payload:
            if participant_code:
                if await _capture_prolific_cookie_for_participant(participant_code, payload):
                    should_clear_cookie = True

        response = await call_next(request)

        if payload and not should_clear_cookie and participant_code:
            if await _capture_prolific_cookie_for_participant(participant_code, payload):
                should_clear_cookie = True

        if request_payload.get("prolific_id") and not payload:
            response.set_cookie(
                PROLIFIC_CAPTURE_COOKIE,
                _sign_cookie_payload(request_payload),
                max_age=PROLIFIC_CAPTURE_MAX_AGE,
                httponly=True,
                samesite="lax",
                path="/",
            )

        if should_clear_cookie:
            response.delete_cookie(PROLIFIC_CAPTURE_COOKIE, path="/")

        return response

async def entry(request):
    """
    Prolific landing endpoint.
    Redirects to /join/<session_code>?<original_querystring>
    IMPORTANT: You must create at least one session in admin (full_study) beforehand.
    """
    qs = str(request.url.query or "")
    suffix = f"?{qs}" if qs else ""

    # pick the latest real session for config_name='full_study'. A session created
    # with "for_prolific" switched off is a test session and never gets Prolific
    # participants, so the study can be tested on the live server at any time.
    # (oTree 5 has no Session.objects and keeps the config name inside the config.)
    # This route is outside oTree's own pages, so open the database the way oTree's
    # request middleware does: a fresh session, under the same lock.
    async with otree_lock2:
        otree_db.new_session()
        try:
            s = next(
                (
                    sess
                    for sess in Session.objects_filter(is_demo=False)
                    .order_by(Session.id.desc())
                    .limit(200)
                    if (sess.config or {}).get("name") == "full_study"
                    and (sess.config or {}).get("for_prolific", True)
                ),
                None,
            )
            session_code = s.code if s else ""
            # the session-wide link /join/<code> takes the join code, not the session code
            join_code = s._anonymous_code if s else ""
        finally:
            otree_db.rollback()

    if not session_code:
        return PlainTextResponse("No active session found. Create a 'full_study' session in admin.", status_code=500)

    payload = _extract_prolific_cookie_payload(request)
    if payload.get("identity_mismatch"):
        logger.warning(
            "Rejected mismatched Prolific entry: session=%s raw_prolific_id=%s raw_participant_label=%s study_id=%s session_id=%s",
            session_code,
            payload.get("raw_prolific_id", ""),
            payload.get("raw_participant_label", ""),
            payload.get("study_id", ""),
            payload.get("session_id", ""),
        )
        return PlainTextResponse(
            "We could not safely admit you to the experiment because the incoming participant identifiers were inconsistent. Please return the study and contact the researcher.",
            status_code=400,
        )

    response = RedirectResponse(f"/join/{join_code}{suffix}", status_code=302)
    if payload.get("prolific_id"):
        _sync_start_prolific_intake_pending(session_code, payload)
        response.set_cookie(
            PROLIFIC_CAPTURE_COOKIE,
            _sign_cookie_payload(payload),
            max_age=PROLIFIC_CAPTURE_MAX_AGE,
            httponly=True,
            samesite="lax",
            path="/",
        )

    return response


def _repair_token_from_request(request: Request) -> str:
    return (
        request.query_params.get("token")
        or request.headers.get("X-Cohort-Repair-Token")
        or ""
    ).strip()


def _serialize_participant_state(participant):
    slot_rows = get_participant_slot_rows(participant.session, participant.code)
    exp_nums = sorted({int(row["exp_num"]) for row in slot_rows if int(row["exp_num"]) > 0})
    mapping = find_prolific_slot_map(
        participant_code=clean_str(getattr(participant, "code", "")),
        session_code=clean_str(getattr(participant.session, "code", "")),
        prefer_active=False,
    )

    exp_from_vars = safe_int(participant.vars.get("exp_target"), 0)
    if exp_from_vars > 0 and exp_from_vars not in exp_nums:
        exp_nums.append(exp_from_vars)
        exp_nums = sorted(exp_nums)
    if not exp_nums:
        exp_nums = [1]

    return dict(
        participant_code=participant.code,
        prolific_id=get_participant_prolific_id(participant),
        participant_label=clean_str(getattr(participant, "label", "")),
        participant_status=get_participant_status(participant) or "",
        prolific_submission_status=clean_str(participant.vars.get("prolific_submission_status", "")),
        returned_from_prolific=bool(participant.vars.get("returned_from_prolific")),
        exp_target=safe_int(participant.vars.get("exp_target"), 0),
        local_slot=safe_int(participant.vars.get("local_slot"), 0),
        prolific_slot_map=(
            dict(
                session_code=clean_str(getattr(mapping, "session_code", "")),
                participant_code=clean_str(getattr(mapping, "participant_code", "")),
                prolific_pid=clean_str(getattr(mapping, "prolific_pid", "")),
                exp_num=safe_int(getattr(mapping, "exp_num", 0), 0),
                slot=safe_int(getattr(mapping, "slot", 0), 0),
                active=bool(getattr(mapping, "active", False)),
                last_status=clean_str(getattr(mapping, "last_status", "")),
            )
            if mapping
            else {}
        ),
        slot_rows=slot_rows,
        cohort_snapshots=[get_cohort_snapshot_data(participant.session, exp_num) for exp_num in exp_nums],
    )


def _serialize_mapping_state(mapping):
    if not mapping:
        return {}

    session_code = clean_str(getattr(mapping, "session_code", ""))
    exp_num = safe_int(getattr(mapping, "exp_num", 0), 0)
    snapshots = []
    session = _find_session_by_code(session_code)
    if session and exp_num > 0:
        try:
            snapshots = [get_cohort_snapshot_data(session, exp_num)]
        except Exception:
            snapshots = []

    return dict(
        participant_code=clean_str(getattr(mapping, "participant_code", "")),
        prolific_id=clean_str(getattr(mapping, "prolific_pid", "")),
        participant_label=clean_str(getattr(mapping, "prolific_pid", "")),
        participant_status="",
        prolific_submission_status="",
        returned_from_prolific=False,
        exp_target=exp_num,
        local_slot=safe_int(getattr(mapping, "slot", 0), 0),
        prolific_slot_map=dict(
            session_code=session_code,
            participant_code=clean_str(getattr(mapping, "participant_code", "")),
            prolific_pid=clean_str(getattr(mapping, "prolific_pid", "")),
            study_id=clean_str(getattr(mapping, "study_id", "")),
            exp_num=exp_num,
            slot=safe_int(getattr(mapping, "slot", 0), 0),
            active=bool(getattr(mapping, "active", False)),
            last_status=clean_str(getattr(mapping, "last_status", "")),
        ),
        slot_rows=[],
        cohort_snapshots=snapshots,
    )


def _recent_sessions(limit: int = 200):
    try:
        return list(Session.objects_filter().order_by(Session.id.desc()).limit(int(limit or 200)))
    except Exception:
        return []


def _find_session_by_code(session_code: str, limit: int = 200):
    session_code = clean_str(session_code)
    if not session_code:
        return None
    for session in _recent_sessions(limit=limit):
        try:
            if clean_str(getattr(session, "code", "")) == session_code:
                return session
        except Exception:
            continue
    return None


def _recent_img_desc_roots(limit: int = 200):
    try:
        roots = list(_recent_root_subsessions(limit=int(limit or 200)))
    except Exception:
        roots = []
    return roots


def _slot_rows_from_root(root, participant_code: str):
    participant_code = clean_str(participant_code)
    if not root or not participant_code:
        return []
    try:
        rows = CohortSlot.filter(subsession=root, participant_code=participant_code)
    except Exception:
        return []

    result = []
    for row in sorted(rows, key=lambda r: (int(getattr(r, "exp_num", 0) or 0), int(getattr(r, "slot", 0) or 0), not bool(getattr(r, "active", False)))):
        result.append(
            dict(
                exp_num=int(getattr(row, "exp_num", 0) or 0),
                slot=int(getattr(row, "slot", 0) or 0),
                active=bool(getattr(row, "active", False)),
                completed=bool(getattr(row, "completed", False)),
            )
        )
    return result


def _player_prolific_id(player) -> str:
    # oTree 5 raises on reading a null field via getattr, so use field_maybe_none.
    try:
        return clean_str(player.field_maybe_none("prolific_id_field"))
    except Exception:
        return ""


def _recent_img_desc_players(limit: int = 4000):
    try:
        return list(ImgDescPlayer.objects_filter().order_by(ImgDescPlayer.id.desc()).limit(int(limit or 4000)))
    except Exception:
        return []


def _find_recent_img_desc_player(participant_code: str = "", prolific_id: str = ""):
    participant_code = clean_str(participant_code)
    prolific_id = clean_str(prolific_id)
    if not participant_code and not prolific_id:
        return None

    for player in _recent_img_desc_players():
        try:
            participant = getattr(player, "participant", None)
            player_code = clean_str(getattr(participant, "code", ""))
            player_pid = _player_prolific_id(player)
            if participant_code and player_code == participant_code:
                return player
            if prolific_id and player_pid == prolific_id:
                return player
        except Exception:
            continue
    return None


def _has_schedule_rows(root, participant_code: str) -> bool:
    participant_code = clean_str(participant_code)
    if not root or not participant_code:
        return False
    try:
        row = get_schedule_item(root, participant_code, 1)
    except Exception:
        return False
    return bool(row)


def _find_repair_fallback_by_participant_code(participant_code: str):
    participant_code = clean_str(participant_code)
    if not participant_code:
        return None

    player = _find_recent_img_desc_player(participant_code=participant_code)
    if player:
        participant = getattr(player, "participant", None)
        session = getattr(participant, "session", None) or getattr(getattr(player, "subsession", None), "session", None)
        try:
            root = player.subsession.in_round(1)
        except Exception:
            root = None
        slot_rows = _slot_rows_from_root(root, participant_code)
        logger.warning(
            "CohortRepair lookup: img_desc player match participant=%s root=%s slot_rows=%s",
            participant_code,
            bool(root),
            len(slot_rows),
        )
        return dict(
            participant_code=participant_code,
            prolific_id=_player_prolific_id(player),
            session=session,
            session_code=clean_str(getattr(session, "code", "")),
            slot_rows=slot_rows,
        )

    logger.warning("CohortRepair fallback: no recent img_desc player match participant=%s", participant_code)
    for root in _recent_img_desc_roots():
        if _has_schedule_rows(root, participant_code):
            session = getattr(root, "session", None)
            slot_rows = _slot_rows_from_root(root, participant_code)
            logger.warning(
                "CohortRepair fallback: schedule-item root match participant=%s session=%s slot_rows=%s",
                participant_code,
                clean_str(getattr(session, "code", "")),
                len(slot_rows),
            )
            return dict(
                participant_code=participant_code,
                prolific_id="",
                session=session,
                session_code=clean_str(getattr(session, "code", "")),
                slot_rows=slot_rows,
            )
        slot_rows = _slot_rows_from_root(root, participant_code)
        if slot_rows:
            logger.warning(
                "CohortRepair fallback: cohort-slot root match participant=%s session=%s slot_rows=%s",
                participant_code,
                clean_str(getattr(getattr(root, "session", None), "code", "")),
                len(slot_rows),
            )
            return dict(
                participant_code=participant_code,
                prolific_id="",
                session=getattr(root, "session", None),
                session_code=clean_str(getattr(getattr(root, "session", None), "code", "")),
                slot_rows=slot_rows,
            )
    logger.warning("CohortRepair fallback: no schedule/cohort root match participant=%s", participant_code)
    return None


def _find_repair_fallback_by_prolific_id(prolific_id: str):
    prolific_id = clean_str(prolific_id)
    if not prolific_id:
        return None

    player = _find_recent_img_desc_player(prolific_id=prolific_id)
    if player:
        try:
            participant = getattr(player, "participant", None)
            participant_code = clean_str(getattr(participant, "code", ""))
            session = getattr(participant, "session", None) or getattr(getattr(player, "subsession", None), "session", None)
            try:
                root = player.subsession.in_round(1)
            except Exception:
                root = None
            slot_rows = _slot_rows_from_root(root, participant_code)
            logger.warning(
                "CohortRepair lookup: img_desc player PID match prolific_id=%s participant=%s root=%s slot_rows=%s",
                prolific_id,
                participant_code,
                bool(root),
                len(slot_rows),
            )
            return dict(
                participant_code=participant_code,
                prolific_id=prolific_id,
                session=session,
                session_code=clean_str(getattr(session, "code", "")),
                slot_rows=slot_rows,
            )
        except Exception:
            pass
    logger.warning("CohortRepair fallback: no recent img_desc player PID match prolific_id=%s", prolific_id)

    try:
        players = list(
            ImgDescPlayer.objects_filter(prolific_id_field=prolific_id)
            .order_by(ImgDescPlayer.id.desc())
            .limit(200)
        )
    except Exception:
        players = []

    if not players:
        try:
            players = list(ImgDescPlayer.objects_filter().order_by(ImgDescPlayer.id.desc()).limit(4000))
        except Exception:
            players = []
        players = [p for p in players if _player_prolific_id(p) == prolific_id]

    for player in players:
        try:
            participant = getattr(player, "participant", None)
            participant_code = clean_str(getattr(participant, "code", ""))
            session = getattr(participant, "session", None)
            if participant_code and session is not None:
                slot_rows = get_participant_slot_rows(session, participant_code)
                return dict(
                    participant_code=participant_code,
                    prolific_id=prolific_id,
                    session=session,
                    session_code=clean_str(getattr(session, "code", "")),
                    slot_rows=slot_rows,
                )
        except Exception:
            continue
    logger.warning("CohortRepair fallback: no player-based PID match prolific_id=%s", prolific_id)
    return None


def _find_repair_fallback(participant_code: str, prolific_id: str):
    fallback = _find_repair_fallback_by_participant_code(participant_code)
    if fallback:
        if not fallback.get("prolific_id") and prolific_id:
            fallback["prolific_id"] = clean_str(prolific_id)
        return fallback
    return _find_repair_fallback_by_prolific_id(prolific_id)


def _serialize_repair_fallback(fallback):
    if not fallback:
        return {}

    session = fallback.get("session")
    slot_rows = list(fallback.get("slot_rows") or [])
    exp_nums = sorted({int(row.get("exp_num", 0) or 0) for row in slot_rows if int(row.get("exp_num", 0) or 0) > 0})
    if not exp_nums:
        exp_nums = [1]

    snapshots = []
    for exp_num in exp_nums:
        try:
            snapshots.append(get_cohort_snapshot_data(session, exp_num))
        except Exception:
            continue

    local_slot = 0
    for row in slot_rows:
        if bool(row.get("active")):
            local_slot = safe_int(row.get("slot", 0), 0)
            break
    if not local_slot and slot_rows:
        local_slot = safe_int(slot_rows[-1].get("slot", 0), 0)

    return dict(
        participant_code=clean_str(fallback.get("participant_code", "")),
        prolific_id=clean_str(fallback.get("prolific_id", "")),
        participant_label=clean_str(fallback.get("prolific_id", "")),
        participant_status="",
        prolific_submission_status="",
        returned_from_prolific=False,
        exp_target=safe_int(exp_nums[0] if exp_nums else 0, 0),
        local_slot=local_slot,
        prolific_slot_map={},
        slot_rows=slot_rows,
        cohort_snapshots=snapshots,
    )


def _serialize_expansion_state(row):
    if not row:
        return {}
    return dict(
        exp_num=safe_int(getattr(row, "exp_num", 0), 0),
        status=clean_str(getattr(row, "status", "")),
        attempts=safe_int(getattr(row, "attempts", 0), 0),
        last_attempt_ts=float(getattr(row, "last_attempt_ts", 0) or 0),
        last_message=clean_str(getattr(row, "last_message", "")),
    )


def _expansion_status_payload(participant, mapping, fallback, participant_code: str, prolific_id: str):
    player = None
    if participant is not None:
        player = _find_recent_img_desc_player(
            participant_code=clean_str(getattr(participant, "code", "")),
            prolific_id=clean_str(get_participant_prolific_id(participant)),
        )
    if player is None:
        player = _find_recent_img_desc_player(participant_code=participant_code, prolific_id=prolific_id)

    session = None
    if participant is not None:
        session = getattr(participant, "session", None)
    if session is None and fallback:
        session = fallback.get("session")
    if session is None and mapping:
        session = _find_session_by_code(clean_str(getattr(mapping, "session_code", "")))

    effective_code = clean_str(
        (getattr(participant, "code", "") if participant is not None else "")
        or participant_code
        or clean_str(getattr(mapping, "participant_code", ""))
        or clean_str((fallback or {}).get("participant_code", ""))
    )
    effective_pid = clean_str(
        (get_participant_prolific_id(participant) if participant is not None else "")
        or prolific_id
        or clean_str(getattr(mapping, "prolific_pid", ""))
        or clean_str((fallback or {}).get("prolific_id", ""))
    )
    participant_label = clean_str(
        (getattr(participant, "label", "") if participant is not None else "")
        or effective_pid
    )

    exp_num = 0
    if player is not None:
        exp_num = safe_int(_current_exp_num_for_participant(player), 0)
    if exp_num <= 0 and mapping:
        exp_num = safe_int(getattr(mapping, "exp_num", 0), 0)

    expansion_state = _serialize_expansion_state(
        _get_expansion_state(player, exp_num) if player is not None and exp_num > 0 else None
    )

    slot_rows = []
    cohort_snapshots = []
    if session is not None and effective_code:
        try:
            slot_rows = get_participant_slot_rows(session, effective_code)
        except Exception:
            slot_rows = []

    if session is not None:
        exp_candidates = []
        if exp_num > 0:
            exp_candidates.append(exp_num)
        for row in slot_rows:
            row_exp = safe_int(row.get("exp_num", 0), 0)
            if row_exp > 0 and row_exp not in exp_candidates:
                exp_candidates.append(row_exp)
        for candidate in exp_candidates[:2]:
            try:
                cohort_snapshots.append(get_cohort_snapshot_data(session, candidate))
            except Exception:
                try:
                    cohort_snapshots.append(
                        dict(exp_num=int(candidate), snapshot=cohort_debug_snapshot(session, candidate))
                    )
                except Exception:
                    pass

    return dict(
        participant_code=effective_code,
        prolific_id=effective_pid,
        participant_label=participant_label,
        session_code=clean_str(getattr(session, "code", "")),
        for_prolific=bool(session and session.config.get("for_prolific")),
        expand_slots=bool(session and session.config.get("expand_slots")),
        player_found=bool(player),
        participant_found=bool(participant),
        mapping_found=bool(mapping),
        fallback_found=bool(fallback),
        current_exp_num=safe_int(exp_num, 0),
        cohort_complete=bool(session and exp_num > 0 and cohort_complete(session, exp_num)),
        next_experiment_exists=bool(session and exp_num > 0 and experiment_exists(session, exp_num + 1)),
        expansion_state=expansion_state,
        participant_status=get_participant_status(participant) if participant is not None else "",
        prolific_slot_map=_serialize_mapping_state(mapping).get("prolific_slot_map", {}) if mapping else {},
        slot_rows=slot_rows,
        cohort_snapshots=cohort_snapshots,
    )


def _find_participant_for_repair(participant_code: str, prolific_id: str):
    participant_code = clean_str(participant_code)
    prolific_id = clean_str(prolific_id)

    if participant_code:
        participant = _find_participant_by_code(participant_code)
        if participant:
            return participant

    if prolific_id:
        mapping = find_prolific_slot_map(prolific_pid=prolific_id, prefer_active=False)
        if mapping:
            participant = _find_participant_in_session(
                clean_str(getattr(mapping, "session_code", "")),
                participant_code=clean_str(getattr(mapping, "participant_code", "")),
                prolific_id=prolific_id,
            )
            if participant:
                return participant
        return _find_participant_by_prolific_id(prolific_id)

    return None


def _find_mapping_for_repair(participant_code: str, prolific_id: str):
    participant_code = clean_str(participant_code)
    prolific_id = clean_str(prolific_id)
    if participant_code:
        mapping = find_prolific_slot_map(participant_code=participant_code, prefer_active=False)
        if mapping:
            return mapping
    if prolific_id:
        return find_prolific_slot_map(prolific_pid=prolific_id, prefer_active=False)
    return None


async def cohort_repair(request: Request):
    expected_token = (os.environ.get("COHORT_REPAIR_TOKEN") or "").strip()
    if not expected_token:
        return JSONResponse(
            {"ok": False, "error": "COHORT_REPAIR_TOKEN not configured"},
            status_code=503,
        )

    provided_token = _repair_token_from_request(request)
    if not provided_token or not secrets.compare_digest(provided_token, expected_token):
        return JSONResponse({"ok": False, "error": "invalid token"}, status_code=403)

    body = {}
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}

    action = clean_str(body.get("action") or request.query_params.get("action") or "status").lower()
    participant_code = clean_str(body.get("participant_code") or request.query_params.get("participant_code") or "")
    prolific_id = clean_str(body.get("prolific_id") or request.query_params.get("prolific_id") or "")

    if action not in {"status", "drop_out", "finished", "free_slot", "active", "expansion_status"}:
        return JSONResponse(
            {"ok": False, "error": "action must be one of status, drop_out, finished, free_slot, active, expansion_status"},
            status_code=400,
        )

    participant = None
    fallback = None
    mapping = _find_mapping_for_repair(participant_code, prolific_id)
    if mapping:
        mapped_code = clean_str(getattr(mapping, "participant_code", ""))
        mapped_pid = clean_str(getattr(mapping, "prolific_pid", ""))
        mapped_session = clean_str(getattr(mapping, "session_code", ""))
        logger.warning(
            "CohortRepair slot-map hit: participant=%s prolific_id=%s session=%s exp=%s slot=%s active=%s",
            mapped_code,
            mapped_pid,
            mapped_session,
            safe_int(getattr(mapping, "exp_num", 0), 0),
            safe_int(getattr(mapping, "slot", 0), 0),
            bool(getattr(mapping, "active", False)),
        )
        if not participant_code:
            participant_code = mapped_code
        if not prolific_id:
            prolific_id = mapped_pid
        participant = _find_participant_in_session(
            mapped_session,
            participant_code=mapped_code,
            prolific_id=mapped_pid,
        )
    else:
        logger.warning(
            "CohortRepair slot-map miss: participant=%s prolific_id=%s",
            participant_code,
            prolific_id,
        )

    if not participant:
        participant = _find_participant_for_repair(participant_code, prolific_id)

    if not participant and not fallback and not mapping:
        fallback = _find_repair_fallback(participant_code, prolific_id)
    if not participant and not mapping and not fallback:
        logger.warning(
            "CohortRepair lookup miss: participant_code=%s prolific_id=%s",
            participant_code,
            prolific_id,
        )
        return JSONResponse(
            {"ok": False, "error": "participant not found", "participant_code": participant_code, "prolific_id": prolific_id},
            status_code=404,
        )

    before = (
        _serialize_participant_state(participant)
        if participant
        else (_serialize_mapping_state(mapping) if mapping else _serialize_repair_fallback(fallback))
    )
    note = "status only"

    if action == "expansion_status":
        note = "read-only expansion diagnostics"
        before = _expansion_status_payload(participant, mapping, fallback, participant_code, prolific_id)
        after = before
    elif participant and action == "drop_out":
        mark_participant_drop_out(participant)
        free_slot_for_participant(participant.session, participant.code)
        reset_this_app_for_participant(participant)
        mark_participant_complete_in_cohort(participant.session, participant.code, completed=False)
        note = "participant marked drop_out, slot freed, app state reset"
    elif participant and action == "finished":
        mark_participant_finished(participant)
        mark_participant_complete_in_cohort(participant.session, participant.code, completed=True)
        maybe_expand_prolific_for_participant(participant)
        note = "participant marked finished and cohort completion updated"
    elif participant and action == "free_slot":
        free_slot_for_participant(participant.session, participant.code)
        reset_this_app_for_participant(participant)
        mark_participant_complete_in_cohort(participant.session, participant.code, completed=False)
        note = "slot freed and app state reset; participant status unchanged"
    elif participant and action == "active":
        mark_participant_active(participant)
        note = "participant marked active"
    elif mapping and action in {"status", "drop_out", "free_slot", "finished"}:
        session_code = clean_str(getattr(mapping, "session_code", ""))
        mapped_code = clean_str(getattr(mapping, "participant_code", ""))
        session = _find_session_by_code(session_code)

        if action in {"drop_out", "free_slot"}:
            if session and mapped_code:
                free_slot_for_participant(session, mapped_code)
                mark_participant_complete_in_cohort(session, mapped_code, completed=False)
                note = "mapping-only slot free completed via session lookup; participant object unresolved"
            else:
                ok = free_slot_from_prolific_slot_map(mapping, status=("drop_out" if action == "drop_out" else "slot_freed"))
                if not ok:
                    return JSONResponse(
                        {
                            "ok": False,
                            "error": "mapping found but session/participant code unavailable",
                            "participant_code": participant_code,
                            "prolific_id": prolific_id,
                        },
                        status_code=409,
                    )
                note = "mapping-only slot free completed directly from slot map; participant object unresolved"
        elif action == "finished":
            ok = mark_finished_from_prolific_slot_map(mapping, status="finished")
            if not ok:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "mapping found but finished status could not be persisted",
                        "participant_code": participant_code,
                        "prolific_id": prolific_id,
                    },
                    status_code=409,
                )
            if session and mapped_code:
                mark_participant_complete_in_cohort(session, mapped_code, completed=True)
            recent_player = _find_recent_img_desc_player(
                participant_code=mapped_code,
                prolific_id=clean_str(getattr(mapping, "prolific_pid", "")),
            )
            if recent_player is not None:
                try:
                    maybe_expand_prolific_when_cohort_complete(recent_player)
                    note = "mapping-only finished completed directly from slot map; expansion check executed from recent player"
                except Exception:
                    note = "mapping-only finished completed directly from slot map; expansion check failed"
            else:
                note = "mapping-only finished completed directly from slot map; no recent player for expansion check"
        else:
            note = "mapping-only status; participant object unresolved"
    elif fallback and action in {"status", "drop_out", "free_slot"}:
        session = fallback.get("session")
        fallback_code = clean_str(fallback.get("participant_code", ""))
        if action in {"drop_out", "free_slot"}:
            if session is None or not fallback_code:
                return JSONResponse(
                    {
                        "ok": False,
                        "error": "repair fallback found but session/participant code unavailable",
                        "participant_code": participant_code,
                        "prolific_id": prolific_id,
                    },
                    status_code=409,
                )
            free_slot_for_participant(session, fallback_code)
            mark_participant_complete_in_cohort(session, fallback_code, completed=False)
            note = "repair-only slot free completed from slot/player fallback"
        else:
            note = "repair-only status from slot/player fallback"
    elif mapping:
        return JSONResponse(
            {
                "ok": False,
                "error": "participant object unresolved for requested action",
                "participant_code": participant_code,
                "prolific_id": prolific_id,
            },
            status_code=409,
        )
    elif fallback:
        return JSONResponse(
            {
                "ok": False,
                "error": "repair fallback unresolved for requested action",
                "participant_code": participant_code,
                "prolific_id": prolific_id,
            },
            status_code=409,
        )

    if action != "expansion_status":
        after = (
            _serialize_participant_state(participant)
            if participant
            else (_serialize_mapping_state(mapping) if mapping else _serialize_repair_fallback(fallback))
        )
    logger.info(
        "CohortRepair: action=%s participant=%s prolific_id=%s status_before=%s status_after=%s",
        action,
        clean_str(getattr(participant, "code", ""))
        or clean_str(getattr(mapping, "participant_code", ""))
        or clean_str((fallback or {}).get("participant_code", "")),
        get_participant_prolific_id(participant)
        if participant
        else (
            clean_str(getattr(mapping, "prolific_pid", ""))
            or clean_str((fallback or {}).get("prolific_id", ""))
        ),
        before.get("participant_status", "") or clean_str(getattr(mapping, "last_status", "")),
        after.get("participant_status", "") or clean_str(getattr(mapping, "last_status", "")),
    )

    return JSONResponse(
        {
            "ok": True,
            "action": action,
            "note": note,
            "before": before,
            "after": after,
        }
    )

app = Starlette(
    middleware=[
        Middleware(ProlificCaptureCookieMiddleware),
    ],
    routes=[
        Route("/prolific/ping/", ping, methods=["GET"]),
        Route("/prolific/webhook/", _with_db_transaction(prolific_webhook_view), methods=["POST", "GET", "HEAD"]),
        Route("/admin/cohort-repair", _with_db_transaction(cohort_repair), methods=["GET", "POST"]),
        Route("/admin/cohort-repair/", _with_db_transaction(cohort_repair), methods=["GET", "POST"]),
        # NEW:
        Route("/entry", entry, methods=["GET"]),
        Route("/entry/", entry, methods=["GET"]),
        Mount("/", app=otree_app),
    ]
)
