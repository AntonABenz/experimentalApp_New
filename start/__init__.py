from otree.api import *
import logging
import json
import re

from reading_xls.get_data import get_data

logger = logging.getLogger(__name__)

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


def _extract_prolific_params(player) -> tuple[str, str, str]:
    pid = study_id = sess_id = ""
    try:
        url = player.participant._url_i_should_be_on()
        if "?" in url:
            params = _parse_querystring(url.split("?", 1)[1])
            pid = (params.get("PROLIFIC_PID") or params.get("prolific_pid") or "").strip()
            study_id = (params.get("STUDY_ID") or params.get("study_id") or "").strip()
            sess_id = (params.get("SESSION_ID") or params.get("session_id") or "").strip()
    except Exception:
        pass
    return pid, study_id, sess_id


def _store_prolific_on_participant(player, pid: str, study_id: str = "", sess_id: str = "") -> None:
    """
    Store Prolific identifiers robustly WITHOUT adding DB fields.
    Stores into:
      - participant.vars["prolific_id"]
      - participant.label  (handy for admin lookup & exports)
      - participant.vars["study_id"]
      - participant.vars["prolific_session_id"]
    """
    p = player.participant

    pid = clean_str(pid)
    study_id = clean_str(study_id)
    sess_id = clean_str(sess_id)

    if pid:
        p.vars["prolific_id"] = pid
        try:
            p.label = pid
        except Exception:
            pass

    if study_id:
        p.vars["study_id"] = study_id

    if sess_id:
        p.vars["prolific_session_id"] = sess_id

    try:
        p.save()
    except Exception:
        pass

    logger.info(
        "Captured Prolific params: pid=%s study_id=%s session_id=%s participant_code=%s",
        pid or "",
        study_id or "",
        sess_id or "",
        getattr(p, "code", ""),
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
            pid, study_id, sess_id = _extract_prolific_params(player)
            if pid:
                _store_prolific_on_participant(player, pid, study_id, sess_id)

    @staticmethod
    def vars_for_template(player):
        # runs on GET, so it captures immediately without needing any button/POST
        _ProlificCaptureMixin._capture(player)
        return {}

    @staticmethod
    def before_next_page(player, timeout_happened):
        # safety retry
        if player.session.config.get("for_prolific") and not player.participant.vars.get("prolific_id"):
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
