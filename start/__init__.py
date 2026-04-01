from otree.api import *
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time

import psycopg2

from reading_xls.get_data import get_data

logger = logging.getLogger(__name__)

PARTICIPANT_STATUS_FIELD = "participant_status"
STATUS_ACTIVE = "active"
STATUS_FINISHED = "finished"
STATUS_DROP_OUT = "drop_out"
PROLIFIC_CAPTURE_COOKIE = "prolific_capture"
_start_prolific_intake_table_available_cache = None

# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------

def clean_str(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return s


def _maybe_cast(v):
    """Cast numbers/bools coming from sheets into Python types where sensible."""
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            return s
    return v


def _parse_querystring(qs: str) -> dict:
    """
    Minimal querystring parser (no urllib dependency).
    Assumes qs like "a=1&b=2".
    """
    out = {}
    if not qs:
        return out
    for kv in str(qs).split("&"):
        if "=" not in kv:
            continue
        k, v = kv.split("=", 1)
        out[k] = v
    return out


def _capture_cookie_secret() -> bytes:
    secret = (
        os.environ.get("OTREE_SECRET_KEY")
        or os.environ.get("SECRET_KEY")
        or os.environ.get("ADMIN_PASSWORD")
        or "otree-secret"
    )
    return str(secret).encode("utf-8")


def _load_prolific_capture_cookie_from_player(player) -> dict:
    req = getattr(player, "request", None)
    if req is None:
        return {}

    cookies = getattr(req, "COOKIES", None) or getattr(req, "cookies", None) or {}
    raw = clean_str(cookies.get(PROLIFIC_CAPTURE_COOKIE, ""))
    if "." not in raw:
        return {}

    encoded, signature = raw.rsplit(".", 1)
    expected = hmac.new(_capture_cookie_secret(), encoded.encode("utf-8"), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, expected):
        return {}

    try:
        padded = encoded + "=" * (-len(encoded) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8"))
    except Exception:
        return {}

    if not isinstance(payload, dict):
        return {}

    return {
        "prolific_id": clean_str(payload.get("prolific_id", "")),
        "participant_label": clean_str(payload.get("participant_label", "")),
        "study_id": clean_str(payload.get("study_id", "")),
        "session_id": clean_str(payload.get("session_id", "")),
    }


def _start_prolific_intake_table_available() -> bool:
    global _start_prolific_intake_table_available_cache
    if _start_prolific_intake_table_available_cache is not None:
        return _start_prolific_intake_table_available_cache

    database_url = clean_str(os.environ.get("DATABASE_URL", "") or os.environ.get("OTREE_DB_URL", ""))
    if not database_url:
        _start_prolific_intake_table_available_cache = False
        return False

    try:
        conn = psycopg2.connect(database_url)
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT to_regclass('public.start_prolificintake')")
                row = cur.fetchone()
                table_exists = bool(row and row[0])
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
                    logger.warning("StartProlificIntake bootstrap: created table start_prolificintake")
                    table_exists = True
                _start_prolific_intake_table_available_cache = table_exists
        finally:
            conn.close()
    except Exception as e:
        _start_prolific_intake_table_available_cache = False
        logger.warning("StartProlificIntake availability check failed: %s", e)

    return bool(_start_prolific_intake_table_available_cache)


def _start_intake_row_from_db(row):
    if not row:
        return {}
    return dict(
        participant_code=clean_str(row[0]),
        session_code=clean_str(row[1]),
        prolific_pid=clean_str(row[2]),
        participant_label=clean_str(row[3]),
        study_id=clean_str(row[4]),
        session_id=clean_str(row[5]),
        last_seen_ts=float(row[6] or 0),
    )


def find_start_prolific_intake(participant_code: str = "", prolific_pid: str = "") -> dict:
    participant_code = clean_str(participant_code)
    prolific_pid = clean_str(prolific_pid)
    if not participant_code and not prolific_pid:
        return {}
    if not _start_prolific_intake_table_available():
        return {}

    database_url = clean_str(os.environ.get("DATABASE_URL", "") or os.environ.get("OTREE_DB_URL", ""))
    if not database_url:
        return {}

    row = None
    try:
        conn = psycopg2.connect(database_url)
        try:
            with conn.cursor() as cur:
                # participant_code is the primary stable key for this intake row once
                # the first real start page exists. Prefer it over PID-based lookup.
                if participant_code:
                    cur.execute(
                        """
                        SELECT participant_code, session_code, prolific_pid, participant_label, study_id, session_id, last_seen_ts
                        FROM public.start_prolificintake
                        WHERE participant_code = %s
                        LIMIT 1
                        """,
                        (participant_code,),
                    )
                    row = cur.fetchone()

                if row is None and prolific_pid:
                    cur.execute(
                        """
                        SELECT participant_code, session_code, prolific_pid, participant_label, study_id, session_id, last_seen_ts
                        FROM public.start_prolificintake
                        WHERE prolific_pid = %s OR participant_label = %s
                        ORDER BY last_seen_ts DESC, participant_code DESC
                        LIMIT 1
                        """,
                        (prolific_pid, prolific_pid),
                    )
                    row = cur.fetchone()
        finally:
            conn.close()
    except Exception as e:
        logger.warning("find_start_prolific_intake: lookup failed participant=%s prolific_id=%s error=%s", participant_code, prolific_pid, e)
        return {}

    return _start_intake_row_from_db(row)


def sync_start_prolific_intake(player, pid: str = "", participant_label: str = "", study_id: str = "", sess_id: str = "") -> dict:
    participant = getattr(player, "participant", None)
    if participant is None:
        return {}
    if not _start_prolific_intake_table_available():
        return {}

    participant_code = clean_str(getattr(participant, "code", ""))
    session_code = clean_str(getattr(getattr(player, "session", None), "code", ""))
    pid = clean_str(pid)
    participant_label = clean_str(participant_label)
    study_id = clean_str(study_id)
    sess_id = clean_str(sess_id)

    if not pid:
        pid = participant_label
    if not participant_label:
        participant_label = pid
    if not participant_code:
        return {}

    database_url = clean_str(os.environ.get("DATABASE_URL", "") or os.environ.get("OTREE_DB_URL", ""))
    if not database_url:
        return {}

    row = {}
    try:
        conn = psycopg2.connect(database_url)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT participant_code, session_code, prolific_pid, participant_label, study_id, session_id, last_seen_ts
                    FROM public.start_prolificintake
                    WHERE participant_code = %s
                    LIMIT 1
                    """,
                    (participant_code,),
                )
                existing = cur.fetchone()
                if existing:
                    existing_row = _start_intake_row_from_db(existing)
                    effective_pid = pid or clean_str(existing_row.get("prolific_pid", "")) or clean_str(existing_row.get("participant_label", ""))
                    effective_label = participant_label or effective_pid or clean_str(existing_row.get("participant_label", "")) or clean_str(existing_row.get("prolific_pid", ""))
                    effective_study_id = study_id or clean_str(existing_row.get("study_id", ""))
                    effective_session_id = sess_id or clean_str(existing_row.get("session_id", ""))
                    effective_session_code = session_code or clean_str(existing_row.get("session_code", ""))
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
                        RETURNING participant_code, session_code, prolific_pid, participant_label, study_id, session_id, last_seen_ts
                        """,
                        (effective_session_code, effective_pid, effective_label, effective_study_id, effective_session_id, time.time(), participant_code),
                    )
                    row = _start_intake_row_from_db(cur.fetchone())
                else:
                    cur.execute(
                        """
                        INSERT INTO public.start_prolificintake
                            (participant_code, session_code, prolific_pid, participant_label, study_id, session_id, last_seen_ts)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING participant_code, session_code, prolific_pid, participant_label, study_id, session_id, last_seen_ts
                        """,
                        (participant_code, session_code, pid, participant_label or pid, study_id, sess_id, time.time()),
                    )
                    row = _start_intake_row_from_db(cur.fetchone())
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        logger.warning(
            "sync_start_prolific_intake: SQL write failed participant=%s prolific_id=%s study_id=%s session_id=%s error=%s",
            participant_code,
            pid,
            study_id,
            sess_id,
            e,
        )
        return {}

    logger.info(
        "sync_start_prolific_intake: participant=%s prolific_id=%s participant_label=%s study_id=%s session_id=%s",
        participant_code,
        clean_str(row.get("prolific_pid", "")),
        clean_str(row.get("participant_label", "")),
        clean_str(row.get("study_id", "")),
        clean_str(row.get("session_id", "")),
    )
    return row


def _extract_prolific_params(player) -> tuple[str, str, str, str]:
    pid = participant_label = study_id = sess_id = ""
    try:
        req = getattr(player, "request", None)
        if req is not None:
            get_params = getattr(req, "GET", None) or {}
            pid = (
                get_params.get("PROLIFIC_PID")
                or get_params.get("prolific_pid")
                or get_params.get("prolific_id")
                or get_params.get("participant_id")
                or get_params.get("participant_label")
                or ""
            ).strip()
            participant_label = (get_params.get("participant_label") or get_params.get("PARTICIPANT_LABEL") or "").strip()
            study_id = (get_params.get("STUDY_ID") or get_params.get("study_id") or "").strip()
            sess_id = (get_params.get("SESSION_ID") or get_params.get("session_id") or "").strip()
    except Exception:
        pass
    try:
        url = player.participant._url_i_should_be_on()
        if (not pid or not participant_label or not study_id or not sess_id) and "?" in url:
            params = _parse_querystring(url.split("?", 1)[1])
            if not pid:
                pid = (
                    params.get("PROLIFIC_PID")
                    or params.get("prolific_pid")
                    or params.get("prolific_id")
                    or params.get("participant_id")
                    or params.get("participant_label")
                    or ""
                ).strip()
            if not participant_label:
                participant_label = (params.get("participant_label") or params.get("PARTICIPANT_LABEL") or "").strip()
            if not study_id:
                study_id = (params.get("STUDY_ID") or params.get("study_id") or "").strip()
            if not sess_id:
                sess_id = (params.get("SESSION_ID") or params.get("session_id") or "").strip()
    except Exception:
        pass
    if not pid:
        pid = participant_label
    if not participant_label:
        participant_label = pid

    intake_payload = find_start_prolific_intake(
        participant_code=clean_str(getattr(player.participant, "code", "")),
        prolific_pid=pid or participant_label,
    )
    if intake_payload:
        if not pid:
            pid = clean_str(intake_payload.get("prolific_pid", "") or intake_payload.get("participant_label", ""))
        if not participant_label:
            participant_label = clean_str(intake_payload.get("participant_label", "") or intake_payload.get("prolific_pid", ""))
        if not study_id:
            study_id = clean_str(intake_payload.get("study_id", ""))
        if not sess_id:
            sess_id = clean_str(intake_payload.get("session_id", ""))

    try:
        if not pid:
            pid = clean_str(
                player.participant.vars.get("prolific_id", "")
                or getattr(player.participant, "label", "")
            )
        if not participant_label:
            participant_label = clean_str(getattr(player.participant, "label", "") or pid)
        if not study_id:
            study_id = clean_str(player.participant.vars.get("study_id", ""))
        if not sess_id:
            sess_id = clean_str(player.participant.vars.get("prolific_session_id", ""))
    except Exception:
        pass

    cookie_payload = _load_prolific_capture_cookie_from_player(player)
    if not pid:
        pid = clean_str(cookie_payload.get("prolific_id", "") or cookie_payload.get("participant_label", ""))
    if not participant_label:
        participant_label = clean_str(cookie_payload.get("participant_label", "") or cookie_payload.get("prolific_id", "") or pid)
    if not study_id:
        study_id = clean_str(cookie_payload.get("study_id", ""))
    if not sess_id:
        sess_id = clean_str(cookie_payload.get("session_id", ""))

    if not pid:
        pid = participant_label
    if not participant_label:
        participant_label = pid

    return pid, participant_label, study_id, sess_id


def _store_prolific_on_participant(player, pid: str, study_id: str = "", sess_id: str = "", participant_label: str = "") -> None:
    """
    Store Prolific identifiers robustly WITHOUT adding DB fields.
    Stores into:
      - participant.vars["prolific_id"]
      - participant.label  (handy for admin lookup & exports)
      - participant.vars["study_id"]
      - participant.vars["prolific_session_id"]

    Stable semantics, do not repurpose:
      - participant.label is the canonical Prolific PID once available
      - participant.vars["prolific_id"] mirrors that PID for backward compatibility
      - neither may be reused for spreadsheet participant numbers
    """
    p = player.participant

    pid = clean_str(pid)
    participant_label = clean_str(participant_label)
    study_id = clean_str(study_id)
    sess_id = clean_str(sess_id)

    if not pid:
        pid = participant_label
    if not participant_label:
        participant_label = pid

    if pid:
        p.vars["prolific_id"] = pid
        try:
            p.label = participant_label or pid
        except Exception:
            pass

    if study_id:
        p.vars["study_id"] = study_id

    if sess_id:
        p.vars["prolific_session_id"] = sess_id

    current_status = clean_str(p.vars.get(PARTICIPANT_STATUS_FIELD))
    if current_status not in {STATUS_FINISHED, STATUS_DROP_OUT}:
        p.vars[PARTICIPANT_STATUS_FIELD] = STATUS_ACTIVE

    try:
        p.save()
    except Exception:
        pass

    sync_start_prolific_intake(player, pid=pid, participant_label=participant_label, study_id=study_id, sess_id=sess_id)

    logger.info(
        "Captured Prolific params: pid=%s participant_label=%s study_id=%s session_id=%s participant_code=%s",
        pid or "",
        participant_label or "",
        study_id or "",
        sess_id or "",
        getattr(p, "code", ""),
    )


def _cohort_entry_state(player) -> dict:
    if not player.session.config.get("for_prolific"):
        return dict(blocked=False, exp_target=1, local_slot=1, waiting_for_prev=False)

    from img_desc import preview_slot_for_participant, cohort_complete

    exp_target, local_slot = preview_slot_for_participant(player)
    waiting_for_prev = bool(
        int(exp_target or 1) > 1 and not cohort_complete(player.session, int(exp_target) - 1)
    )

    # Stable semantics, do not repurpose:
    # local_slot is only a preview of the spreadsheet participant number here.
    return dict(
        blocked=bool(int(local_slot or 0) == 0 or waiting_for_prev),
        exp_target=int(exp_target or 0),
        local_slot=int(local_slot or 0),
        waiting_for_prev=waiting_for_prev,
    )


def build_image_url(player, filename: str) -> str:
    """
    Constructs the full S3 URL for a practice image.
    Assumes practice images are under {s3_base}/practice/{filename}.{ext}
    """
    filename = clean_str(filename)
    if not filename:
        return ""

    s3_base = clean_str(player.session.vars.get("s3path_base", ""))
    ext = clean_str(player.session.vars.get("extension", "png")) or "png"

    if not filename.lower().endswith(f".{ext}"):
        filename = f"{filename}.{ext}"

    if not s3_base:
        return filename  # fallback

    base = s3_base.rstrip("/")
    return f"{base}/practice/{filename}"


def _get_right_answers_list(practice_dict: dict) -> list[str]:
    """Reads right_answer_1, right_answer_2... into a list of strings."""
    keys = [k for k in (practice_dict or {}).keys() if str(k).lower().startswith("right_answer_")]

    def extract_num(k):
        nums = re.findall(r"\d+", str(k))
        return int(nums[0]) if nums else 999

    keys.sort(key=extract_num)

    out = []
    for k in keys:
        v = clean_str((practice_dict or {}).get(k))
        if v:
            out.append(v)
    return out


def _parse_kv_sheet(rows: list[dict]) -> dict:
    """
    Practice tabs are key/value sheets:
      name | value | comment
    Convert to dict {name: value}
    """
    out = {}
    for r in rows or []:
        name = clean_str(r.get("name") or r.get("Name"))
        if not name:
            continue
        val = r.get("value") if "value" in r else r.get("Value")
        out[name] = _maybe_cast(val)
    return out


def _find_practice_tab(practices_payload: dict, practice_id: int):
    """
    Accept many tab naming styles:
      practice_5, Practice_5, PRACTICE_5, practice5, Practice5
    """
    if not isinstance(practices_payload, dict):
        return None

    pid = str(practice_id)
    candidates = [
        f"practice_{pid}", f"Practice_{pid}", f"PRACTICE_{pid}",
        f"practice{pid}", f"Practice{pid}", f"PRACTICE{pid}",
    ]

    for c in candidates:
        if c in practices_payload:
            return practices_payload[c]

    lower_map = {str(k).lower(): k for k in practices_payload.keys()}
    for c in candidates:
        k = lower_map.get(c.lower())
        if k is not None:
            return practices_payload[k]

    return None


# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------

class C(BaseConstants):
    NAME_IN_URL = "start"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # IMPORTANT: NO prolific_id field here (Option 2 avoids DB schema changes)
    survey_data = models.LongStringField(blank=True)
    practice_response = models.LongStringField(blank=True)


# -------------------------------------------------------------------
# Session creation
# -------------------------------------------------------------------

def creating_session(subsession: BaseSubsession):
    session = subsession.session

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Session config must include 'filename'")

    payload = get_data(filename)

    settings = payload.get("settings", {}) or {}
    session.vars["sheet_settings"] = settings

    # Globals used across practice templates
    session.vars["allowed_values"] = settings.get("allowed_values", []) or []
    session.vars["allowed_regex"] = settings.get("allowed_regex", []) or []
    session.vars["suffixes"] = settings.get("suffixes", ["solve/d", "exercises"]) or ["solve/d", "exercises"]
    session.vars["interpreter_choices"] = settings.get("interpreter_choices", []) or []
    session.vars["interpreter_title"] = settings.get("interpreter_title", "Interpretation") or "Interpretation"
    session.vars["EndOfIntroText"] = settings.get("EndOfIntroText", "") or ""

    # S3 base + extension
    raw_s3 = clean_str(settings.get("s3path") or settings.get("s3path_base") or "")
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            bucket = raw_s3.split("buckets/")[1].split("?")[0].strip("/")
            raw_s3 = f"https://{bucket}.s3.eu-central-1.amazonaws.com"
        except Exception:
            pass

    session.vars["s3path_base"] = raw_s3
    session.vars["extension"] = clean_str(settings.get("extension", "png")) or "png"

    # Practice settings: build Practice1..Practice7
    practice_settings = {}

    # Case A: get_data embedded PracticeX dicts inside settings
    for k, v in settings.items():
        if str(k).startswith("Practice") and isinstance(v, dict):
            p = v.copy()
            p["right_answer"] = _get_right_answers_list(p)
            practice_settings[str(k)] = p

    # Case B: get_data provides separate practice tabs (payload["practices"])
    practices_payload = payload.get("practices")
    if isinstance(practices_payload, dict):
        for pid in range(1, 8):
            key = f"Practice{pid}"
            if key in practice_settings:
                continue

            tab_rows = _find_practice_tab(practices_payload, pid)
            if not tab_rows:
                continue

            p = _parse_kv_sheet(tab_rows)
            p["right_answer"] = _get_right_answers_list(p)
            practice_settings[key] = p

    session.vars["practice_settings"] = practice_settings


# -------------------------------------------------------------------
# Pages
# -------------------------------------------------------------------

class _BasePage(Page):
    pass


class _ProlificCaptureMixin:
    @staticmethod
    def _capture(player):
        if player.session.config.get("for_prolific"):
            pid, participant_label, study_id, sess_id = _extract_prolific_params(player)
            if pid or participant_label or study_id or sess_id:
                _store_prolific_on_participant(player, pid, study_id, sess_id, participant_label=participant_label)

    @staticmethod
    def vars_for_template(player):
        # runs on GET, so it captures immediately without needing any button/POST
        _ProlificCaptureMixin._capture(player)
        return {}

    @staticmethod
    def before_next_page(player, timeout_happened):
        # safety retry when any Prolific identifier is still missing
        if (
            player.session.config.get("for_prolific")
            and (
                not clean_str(player.participant.vars.get("prolific_id"))
                or not clean_str(player.participant.vars.get("study_id"))
                or not clean_str(player.participant.vars.get("prolific_session_id"))
            )
        ):
            _ProlificCaptureMixin._capture(player)


class _PracticePage(_BasePage):
    practice_id = None
    template_name = None
    form_model = "player"
    form_fields = ["practice_response"]

    @classmethod
    def is_displayed(cls, player: Player):
        key = f"Practice{cls.practice_id}"
        return key in (player.session.vars.get("practice_settings") or {})

    @classmethod
    def _settings(cls, player: Player):
        key = f"Practice{cls.practice_id}"
        s = (player.session.vars.get("practice_settings") or {}).get(key, {}).copy()

        s["full_image_path"] = build_image_url(player, s.get("image", ""))

        if "required_rows" not in s:
            if cls.practice_id == 7:
                s["required_rows"] = 5
            elif cls.practice_id in {4, 6}:
                s["required_rows"] = 3

        return s

    @classmethod
    def vars_for_template(cls, player: Player):
        s = cls._settings(player)
        return dict(
            settings=s,
            js_regex=json.dumps(player.session.vars.get("allowed_regex", [])),
            allowed_values=player.session.vars.get("allowed_values", []),
            suffixes=player.session.vars.get("suffixes", []),
        )


class Consent(_ProlificCaptureMixin, _BasePage):
    template_name = "start/Consent.html"


class CohortEntryGate(_ProlificCaptureMixin, _BasePage):
    template_name = "start/CohortEntryGate.html"

    @staticmethod
    def is_displayed(player):
        if not player.session.config.get("for_prolific"):
            return False
        _ProlificCaptureMixin._capture(player)
        return _cohort_entry_state(player)["blocked"]

    @staticmethod
    def vars_for_template(player):
        _ProlificCaptureMixin._capture(player)
        return _cohort_entry_state(player)


class Demographics(_BasePage):
    form_model = "player"
    form_fields = ["survey_data"]


class Instructions(_BasePage):
    # (Optional) expose these for templates that use them
    show_instructions = True
    instructions_path = "start/Instructions.html"


class Practice1(_PracticePage):
    practice_id = 1
    template_name = "start/Practice1.html"


class Practice2(_PracticePage):
    practice_id = 2
    template_name = "start/Practice1.html"


class Practice3(_PracticePage):
    practice_id = 3
    template_name = "start/Practice1.html"


class Practice4(_PracticePage):
    practice_id = 4
    template_name = "start/Practice4.html"


class Practice5(_PracticePage):
    practice_id = 5
    template_name = "start/Practice5.html"


class Practice6(_PracticePage):
    practice_id = 6
    template_name = "start/Practice6.html"


class Practice7(_PracticePage):
    practice_id = 7
    template_name = "start/Practice7.html"


class EndOfIntro(_BasePage):
    @staticmethod
    def vars_for_template(player):
        return dict(end_of_intro_text=player.session.vars.get("EndOfIntroText", ""))


page_sequence = [
    Consent,
    CohortEntryGate,
    Demographics,
    Instructions,
    Practice1,
    Practice2,
    Practice3,
    Practice4,
    Practice5,
    Practice6,
    Practice7,
    EndOfIntro,
]


def custom_export(players):
    # Stable export API, do not repurpose:
    # - "participant_code" is the opaque oTree participant code
    # - "prolific_id" is the Prolific PID, with participant.label only as fallback
    def _prolific_id_for_export(participant):
        pid = participant.vars.get("prolific_id", "") or getattr(participant, "label", "")
        if pid:
            return pid
        try:
            for pp in participant.get_players():
                pid = clean_str(getattr(pp, "prolific_id_field", ""))
                if pid:
                    return pid
        except Exception:
            pass
        return ""

    def _participant_status_for_export(participant):
        try:
            from img_desc import get_participant_status

            status = clean_str(get_participant_status(participant)).lower()
            if status in {STATUS_ACTIVE, STATUS_FINISHED, STATUS_DROP_OUT}:
                return status
        except Exception:
            pass
        status = clean_str(participant.vars.get(PARTICIPANT_STATUS_FIELD, "")).lower()
        if status in {STATUS_ACTIVE, STATUS_FINISHED, STATUS_DROP_OUT}:
            return status
        return ""

    yield [
        "session_code",
        "participant_code",
        "prolific_id",
        "study_id",
        "prolific_session_id",
        "participant_status",
        "survey_data",
        "practice_response",
    ]

    for player in players:
        participant = player.participant
        session_obj = player.session
        yield [
            getattr(session_obj, "code", ""),
            getattr(participant, "code", ""),
            _prolific_id_for_export(participant),
            participant.vars.get("study_id", ""),
            participant.vars.get("prolific_session_id", ""),
            _participant_status_for_export(participant),
            player.survey_data or "",
            player.practice_response or "",
        ]
