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
    # Avoid double /practice if base already has it (rare)
    if "/practice" not in base:
        return f"{base}/practice/{filename}"
    return f"{base}/{filename}"


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
    
    print(f"Start App: Loaded {len(practice_settings)} practice pages.")


class _BasePage(Page):
    pass


class _PracticePage(_BasePage):
    practice_id = None
    template_name = None
    form_model = "player"
    form_fields = ["practice_response"]

    @classmethod
    def is_displayed(cls, player: Player):
        # FIX: Check if the practice content exists in our loaded dictionary.
        # DO NOT check 'settings.get(key)' with _truthy because that dictionary is not "True" string.
        key = f"Practice{cls.practice_id}"
        practices = player.session.vars.get("practice_settings", {})
        
        # If the key exists (e.g. "Practice1"), we show the page.
        return key in practices

    @classmethod
    def _settings(cls, player: Player):
        key = f"Practice{cls.practice_id}"
        s = (player.session.vars.get("practice_settings", {}) or {}).get(key, {}) or {}
        s = dict(s)  # copy
        s["full_image_path"] = build_image_url(player, s.get("image", ""))
        
        # Pass required rows for template logic
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
            # Pass simple JSON strings for JS validation
            js_right_answers=json.dumps(s.get("right_answers", [])),
            js_regex=json.dumps(player.session.vars.get("allowed_regex", [])),
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
    
    @staticmethod
    def vars_for_template(player: Player):
        s = _PracticePage._settings(player)
        allowed = player.session.vars.get("allowed_values", [])
        return dict(
            title=s.get("title", "Practice 5"),
            main_text=s.get("main_text", ""),
            image_path=s.get("full_image_path", ""),
            js_right_answers=json.dumps(s.get("right_answers", [])),
            vocab1=allowed[0] if len(allowed) > 0 else [],
            vocab2=allowed[1] if len(allowed) > 1 else [],
            suffixes=player.session.vars.get("suffixes", [])
        )


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
