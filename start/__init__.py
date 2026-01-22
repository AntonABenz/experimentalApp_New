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
    return "" if s.lower() in {"nan", "none"} else s


def fix_s3_url(raw_s3: str) -> str:
    raw_s3 = clean_str(raw_s3)
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            bucket = raw_s3.split("buckets/")[1].split("?")[0].strip("/")
            return f"https://{bucket}.s3.eu-central-1.amazonaws.com"
        except Exception:
            return raw_s3
    return raw_s3


def build_practice_image_url(session, image_name: str) -> str:
    """
    Practice images live under: {s3path_base}/practice/{image}.{ext}
    """
    name = clean_str(image_name)
    if not name:
        return ""

    base = (session.vars.get("s3path_base") or "").rstrip("/")
    ext = session.vars.get("extension") or "png"

    if not name.lower().endswith(f".{ext}"):
        name = f"{name}.{ext}"

    if not base:
        # fallback: return filename only (might work if served locally)
        return name

    return f"{base}/practice/{name}"


def _get_right_answers_list(practice_data: dict) -> list:
    """
    You said Practice5 won't validate, but other practices might.
    Keep this for Practice1-4/6/7 if needed.
    """
    answers = []
    keys = [k for k in practice_data.keys() if str(k).lower().startswith("right_answer_")]

    def extract_num(k):
        nums = re.findall(r"\d+", str(k))
        return int(nums[0]) if nums else 999

    keys.sort(key=extract_num)

    for k in keys:
        val = clean_str(practice_data.get(k))
        if val:
            answers.append(val)
    return answers


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

    # global vars
    raw_s3 = str(settings.get("s3path") or settings.get("s3path_base") or "")
    session.vars["s3path_base"] = fix_s3_url(raw_s3)
    session.vars["extension"] = settings.get("extension", "png")

    session.vars["allowed_values"] = settings.get("allowed_values", []) or []
    session.vars["allowed_regex"] = settings.get("allowed_regex", []) or []
    session.vars["suffixes"] = settings.get("suffixes", ["solve/d", "exercises"])
    session.vars["interpreter_choices"] = settings.get("interpreter_choices", []) or []
    session.vars["interpreter_title"] = settings.get("interpreter_title", "Interpretation")
    session.vars["EndOfIntroText"] = settings.get("EndOfIntroText", "")

    # practice settings
    practice_settings = {}
    for k, v in settings.items():
        if k.startswith("Practice") and isinstance(v, dict):
            p = v.copy()
            p["right_answer"] = _get_right_answers_list(p)  # kept for other practice pages
            practice_settings[k] = p

    session.vars["practice_settings"] = practice_settings


# -------------------------------------------------------------------
# Pages
# -------------------------------------------------------------------

class _BasePage(Page):
    pass


class _PracticePage(_BasePage):
    practice_id = None
    template_name = None
    form_model = "player"
    form_fields = ["practice_response"]

    @classmethod
    def is_displayed(cls, player: Player):
        key = f"Practice{cls.practice_id}"
        return key in player.session.vars.get("practice_settings", {})

    @classmethod
    def _settings(cls, player: Player):
        key = f"Practice{cls.practice_id}"
        s = player.session.vars["practice_settings"].get(key, {}).copy()

        # robust image lookup
        img = s.get("image")
        if not img:
            for kk, vv in s.items():
                if str(kk).strip().lower() == "image":
                    img = vv
                    break

        s["full_image_path"] = build_practice_image_url(player.session, img)
        return s

    @classmethod
    def vars_for_template(cls, player: Player):
        s = cls._settings(player)
        return dict(
            settings=s,
            allowed_values=player.session.vars.get("allowed_values", []),
            suffixes=player.session.vars.get("suffixes", []),
            js_regex=json.dumps(player.session.vars.get("allowed_regex", [])),
        )


class Consent(_BasePage):
    @staticmethod
    def before_next_page(player, timeout_happened=False):
        if player.session.config.get("for_prolific"):
            player.participant.vars["prolific_id"] = player.participant.label


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
    """
    Display-only: image + vocab box + next
    """
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
            vocab1=allowed[0] if len(allowed) > 0 else [],
            vocab2=allowed[1] if len(allowed) > 1 else [],
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
