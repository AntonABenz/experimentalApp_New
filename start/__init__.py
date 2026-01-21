from otree.api import *
import logging
import json
import re

from reading_xls.get_data import get_data

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# CONSTANTS / MODELS
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def clean_str(x):
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return s


def _truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def _normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", clean_str(s)).strip()


def build_practice_image_url(session, filename: str) -> str:
    """
    Practice images live in: <s3_base>/practice/<filename>.<ext>
    """
    filename = clean_str(filename)
    if not filename:
        return ""

    s3_base = clean_str(session.vars.get("s3path_base", "")).rstrip("/")
    ext = clean_str(session.vars.get("extension", "png"))

    if not filename.lower().endswith(f".{ext}"):
        filename = f"{filename}.{ext}"

    if not s3_base:
        return filename

    return f"{s3_base}/practice/{filename}"


def extract_right_answer_list(practice_dict: dict) -> list[str]:
    """
    Converts right_answer_1..N into a simple list stored as practice_dict["right_answer"].
    Keeps original strings exactly (we normalize later when comparing).
    """
    out = []
    for i in range(1, 100):
        k = f"right_answer_{i}"
        v = clean_str(practice_dict.get(k, ""))
        if v:
            out.append(v)
    return out


def right_answers_as_rows(raw_list: list[str]) -> list[list[str]]:
    """
    For text-field practices: "3; the A" -> ["3", "the A"]
    For radio practices: "1" -> ["1"]
    """
    rows = []
    for raw in raw_list:
        parts = [p.strip() for p in str(raw).split(";")]
        rows.append(parts)
    return rows


# ----------------------------------------------------------------------------
# SESSION CREATION
# ----------------------------------------------------------------------------
def creating_session(subsession: BaseSubsession):
    session = subsession.session
    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Session config must include 'filename'")

    payload = get_data(filename)
    settings = payload.get("settings", {}) or {}

    # Store full settings for debugging if needed
    session.vars["sheet_settings"] = settings

    # Core globals
    session.vars["allowed_values"] = settings.get("allowed_values", []) or []
    session.vars["allowed_regex"] = settings.get("allowed_regex", []) or []
    session.vars["suffixes"] = settings.get("suffixes", ["solve/d", "exercises"])
    session.vars["interpreter_choices"] = settings.get("interpreter_choices", []) or []
    session.vars["interpreter_title"] = settings.get("interpreter_title", "Buy medals:")
    session.vars["EndOfIntroText"] = settings.get("EndOfIntroText", "")

    # s3path_base cleanup (console URL -> bucket endpoint)
    raw_s3 = clean_str(settings.get("s3path") or settings.get("s3path_base") or "")
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            bucket = raw_s3.split("buckets/")[1].split("?")[0].strip("/")
            raw_s3 = f"https://{bucket}.s3.eu-central-1.amazonaws.com"
        except Exception:
            pass
    session.vars["s3path_base"] = raw_s3
    session.vars["extension"] = clean_str(settings.get("extension", "png"))

    # Practice enabled flags (Practice1..Practice7 = 1/0)
    session.vars["practice_enabled"] = {
        f"Practice{i}": _truthy(settings.get(f"Practice{i}", 1))
        for i in range(1, 8)
    }

    # Practice configs (Practice1..Practice7 dicts)
    practice_settings = {}
    for k, v in settings.items():
        if k.startswith("Practice") and isinstance(v, dict):
            d = v.copy()

            # ✅ Build right_answer list from right_answer_1..N
            d["right_answer"] = extract_right_answer_list(d)
            d["right_answers_rows"] = right_answers_as_rows(d["right_answer"])

            # ✅ Precompute full image path (practice folder)
            d["full_image_path"] = build_practice_image_url(session, d.get("image", ""))

            # ✅ Useful default: required_rows
            # - If right_answer exists for text rows, required rows = number of right answers
            # - For practice6/7 (no right answers): enforce 3 / 5 as per your spec
            if k == "Practice6":
                d["required_rows"] = 3
            elif k == "Practice7":
                d["required_rows"] = 5
            elif d["right_answer"]:
                d["required_rows"] = len(d["right_answer"])
            else:
                d["required_rows"] = 0

            practice_settings[k] = d

    session.vars["practice_settings"] = practice_settings


# ----------------------------------------------------------------------------
# BASE PRACTICE PAGE (SERVER-SIDE VALIDATION)
# ----------------------------------------------------------------------------
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
        enabled = player.session.vars.get("practice_enabled", {}).get(key, True)
        return bool(enabled)

    @classmethod
    def _settings(cls, player: Player):
        key = f"Practice{cls.practice_id}"
        return (player.session.vars.get("practice_settings", {}).get(key, {}) or {}).copy()

    @classmethod
    def vars_for_template(cls, player: Player):
        s = cls._settings(player)
        return dict(
            settings=s,
            allowed_values=player.session.vars.get("allowed_values", []),
            suffixes=player.session.vars.get("suffixes", []),
            js_right_answers=json.dumps(s.get("right_answer", [])),
            js_regex=json.dumps(player.session.vars.get("allowed_regex", [])),
        )

    @classmethod
    def error_message(cls, player: Player, values):
        """
        Blocks Next if practice_response doesn't match right answers.
        - Practice1-3: expects JSON list of booleans [true,false,...]
        - Practice4-5: expects JSON list-of-lists [["3","the A"], ...]
        - Practice6-7: no right answers; only row-count constraints handled in JS
        """
        s = cls._settings(player)
        gold_raw = s.get("right_answer", []) or []
        if not gold_raw:
            return None  # nothing to validate

        raw = clean_str(values.get("practice_response", ""))
        if not raw:
            return "Please answer the practice task."

        try:
            resp = json.loads(raw)
        except Exception:
            return "Could not read your answer. Please try again."

        # Practice 1-3: list[bool]
        if cls.practice_id in {1, 2, 3}:
            gold = [bool(int(x)) for x in gold_raw if str(x).strip() != ""]
            if not isinstance(resp, list):
                return "Please answer all questions."
            if len(resp) != len(gold):
                return "Please answer all questions."
            if any(bool(resp[i]) != gold[i] for i in range(len(gold))):
                return "That was not correct. Please try again."
            return None

        # Practice 4-5: list[list[str]] (order-insensitive; normalize spaces)
        gold_rows = right_answers_as_rows(gold_raw)
        if not isinstance(resp, list):
            return "Please enter your sentences."

        # normalize user rows into same representation
        user_rows = []
        for row in resp:
            if isinstance(row, list):
                user_rows.append([_normalize_spaces(x).lower() for x in row])
            else:
                user_rows.append([_normalize_spaces(row).lower()])

        gold_norm = [
            [_normalize_spaces(x).lower() for x in row]
            for row in gold_rows
        ]

        # compare sets (order independent)
        if sorted(user_rows) != sorted(gold_norm):
            return "Some answers are incorrect. Please check spelling and values."

        return None


# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
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
    template_name = "start/Practice5.html"  # text-fields template


class Practice5(_PracticePage):
    practice_id = 5
    template_name = "start/Practice5.html"  # text-fields template


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
