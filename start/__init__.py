from otree.api import *
import logging
import json

from reading_xls.get_data import get_data

logger = logging.getLogger(__name__)

def _truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

def clean_str(x):
    if x is None:
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none"} else s

def build_image_url(player, filename: str) -> str:
    """
    Practice images live under: <s3_base>/practice/<filename>.<ext>
    BUT practice 6/7 can have blank image.
    """
    filename = clean_str(filename)
    if not filename:
        return ""

    s3_base = clean_str(player.session.vars.get("s3path_base"))
    ext = clean_str(player.session.vars.get("extension") or "png")

    # ensure extension
    if not filename.lower().endswith(f".{ext}"):
        filename = f"{filename}.{ext}"

    base = s3_base.rstrip("/")
    if not base:
        return filename

    # Practice images must be in /practice
    return f"{base}/practice/{filename}"


class C(BaseConstants):
    NAME_IN_URL = "start"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    survey_data = models.LongStringField(blank=True)
    practice_response = models.LongStringField(blank=True)


def creating_session(subsession: BaseSubsession):
    session = subsession.session
    if subsession.round_number != 1:
        return

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Session config must include 'filename'")

    payload = get_data(filename)
    settings = payload.get("settings", {}) or {}

    # Store full settings
    session.vars["sheet_settings"] = settings

    # Extract practice dicts that were loaded from practice_1..practice_7 tabs
    practice_settings = {}
    for k, v in settings.items():
        if isinstance(k, str) and k.startswith("Practice") and isinstance(v, dict):
            practice_settings[k] = v
    session.vars["practice_settings"] = practice_settings

    # General session vars
    session.vars["allowed_values"] = settings.get("allowed_values", []) or []
    session.vars["allowed_regex"] = settings.get("allowed_regex", []) or []
    session.vars["suffixes"] = settings.get("suffixes", ["solve/d", "exercises"]) or ["solve/d", "exercises"]
    session.vars["interpreter_choices"] = settings.get("interpreter_choices", []) or []
    session.vars["interpreter_title"] = settings.get("interpreter_title", "Buy medals:")
    session.vars["EndOfIntroText"] = settings.get("EndOfIntroText", "")

    # Fix S3 base from console URL
    raw_s3 = clean_str(settings.get("s3path") or settings.get("s3path_base") or "")
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            bucket = raw_s3.split("buckets/")[1].split("?")[0].strip("/")
            raw_s3 = f"https://{bucket}.s3.eu-central-1.amazonaws.com"
        except Exception:
            pass
    session.vars["s3path_base"] = raw_s3
    session.vars["extension"] = clean_str(settings.get("extension") or "png")


class _BasePage(Page):
    pass


class _PracticePage(_BasePage):
    practice_id = None
    template_name = None
    form_model = "player"
    form_fields = ["practice_response"]

    @classmethod
    def is_displayed(cls, player: Player):
        # ✅ THIS prevents the “skip all practice pages” issue.
        # Toggle comes from global settings sheet: Practice1..Practice7 = 1/0
        settings = player.session.vars.get("sheet_settings", {}) or {}
        flag = settings.get(f"Practice{cls.practice_id}", "1")
        return _truthy(flag)

    @classmethod
    def _settings(cls, player: Player):
        key = f"Practice{cls.practice_id}"
        s = (player.session.vars.get("practice_settings", {}) or {}).get(key, {}) or {}
        s = dict(s)  # copy
        s["full_image_path"] = build_image_url(player, s.get("image", ""))
        # useful for templates (dynamic row requirement)
        if "required_rows" not in s:
            # Practice4 expects 3 rows; Practice6 expects 3; Practice7 expects 5
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
            allowed_values=player.session.vars.get("allowed_values", []),
            suffixes=player.session.vars.get("suffixes", []),
        )


class Consent(_BasePage):
    @staticmethod
    def before_next_page(player, timeout_happened=False):
        if player.session.config.get("for_prolific"):
            p = player.participant
            p.vars["prolific_id"] = p.label


class Demographics(_BasePage):
    form_model = "player"
    form_fields = ["survey_data"]


class Instructions(_BasePage):
    pass


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
    def vars_for_template(self):
        return dict(end_of_intro_text=self.session.vars.get("EndOfIntroText", ""))


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
