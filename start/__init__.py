from otree.api import *
import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ====================================================================
# HELPERS
# ====================================================================

def load_excel(path: str):
    """Load Excel safely and return dict of sheets."""
    excel_path = Path(path)

    if not excel_path.exists():
        logger.error(f"Excel not found at: {excel_path.resolve()}")
        return None

    logger.info(f"Loading Excel: {excel_path.resolve()}")
    return pd.read_excel(excel_path, sheet_name=None, dtype=str)


def normalize_kv(df):
    """Convert 2-column sheet into a {name:value} dict."""
    df = df.rename(columns=str.lower)
    if not {"name", "value"}.issubset(df.columns):
        return {}

    out = {}
    for _, row in df.iterrows():
        name = str(row["name"]).strip()
        val = "" if pd.isna(row["value"]) else str(row["value"]).strip()
        if name:
            out[name] = val
    return out


def extract_right_answers(kv_dict):
    """Reads right_answer_1, right_answer_2 ..."""
    answers = []
    for key in sorted(kv_dict.keys()):
        if key.startswith("right_answer_"):
            answers.append(kv_dict[key])
    return answers


# ====================================================================
# LOAD PRACTICE SHEETS
# ====================================================================

def load_practices(excel):
    """Reads practice_1 ... practice_n tabs."""
    practices = {}

    for sheet_name, df in excel.items():
        name = sheet_name.lower().strip()

        if not name.startswith("practice_"):
            continue

        kv = normalize_kv(df)
        if not kv:
            continue

        practice = {
            "title": kv.get("title", sheet_name),
            "main_text": kv.get("main_text", ""),
            # IMPORTANT: read S3 column (named s3path in your Excel)
            "image": kv.get("s3path", ""),       # <<<<<<<<<<<<<<<<<<<<<< FIX
            "right_answer": extract_right_answers(kv),
        }

        practices[name] = practice

    return practices


# ====================================================================
# MODEL + SESSION SETUP
# ====================================================================

class C(BaseConstants):
    NAME_IN_URL = "start"
    NUM_ROUNDS = 1
    PLAYERS_PER_GROUP = None


class Subsession(BaseSubsession):

    def creating_session(self):
        config = self.session.config
        xlsx = config.get("practice_xlsx")

        excel = load_excel(xlsx)
        if excel is None:
            self.session.vars["practice_settings"] = {}
            return

        # -------- Settings (meta) sheet --------
        settings_sheet = excel.get("Settings") or excel.get("settings")
        sheet_settings = normalize_kv(settings_sheet) if settings_sheet is not None else {}

        interpreter_choices = sheet_settings.get("interpreter_choices", "")
        interpreter_choices = [c.strip() for c in interpreter_choices.split(";") if c.strip()]

        # -------- Practice pages --------
        practice_settings = load_practices(excel)

        # -------- Store into session.vars --------
        self.session.vars["practice_settings"] = practice_settings
        self.session.vars["sheet_settings"] = sheet_settings
        self.session.vars["interpreter_title"] = sheet_settings.get("interpreter_title", "Interpretation")
        self.session.vars["interpreter_choices"] = interpreter_choices

        logger.info(f"Loaded practices: {list(practice_settings.keys())}")
        logger.info(f"Interpreter choices = {interpreter_choices}")


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    pass


# ====================================================================
# BASE + PRACTICE PAGES
# ====================================================================

class _BasePage(Page):
    instructions = False

    def get_context_data(self, **context):
        d = super().get_context_data(**context)
        d["instructions"] = self.instructions
        return d


class Consent(_BasePage):
    pass


class Demographics(_BasePage):
    form_model = 'player'
    form_fields = []


class Instructions(_BasePage):
    instructions = True


class _PracticePage(_BasePage):
    template_name = "start/Practice1.html"
    practice_id = None

    @classmethod
    def _settings(cls, player):
        p = player.session.vars["practice_settings"]

        key = f"practice_{cls.practice_id}"
        s = p.get(key, {}).copy()

        # Use S3 path directly
        s["full_image_path"] = s.get("image") or f"https://picsum.photos/200/300?text=Practice+{cls.practice_id}"

        s.setdefault("title", f"Practice {cls.practice_id}")
        s.setdefault("main_text", "")
        s.setdefault("right_answer", [])

        return s

    @classmethod
    def vars_for_template(cls, player):
        return dict(settings=cls._settings(player))

    @classmethod
    def js_vars(cls, player):
        return dict(settings=cls._settings(player))


class Practice1(_PracticePage): practice_id = 1
class Practice2(_PracticePage): practice_id = 2
class Practice3(_PracticePage): practice_id = 3
class Practice4(_PracticePage): practice_id = 4
class Practice5(_PracticePage): practice_id = 5
class Practice6(_PracticePage): practice_id = 6
class Practice7(_PracticePage): practice_id = 7


class EndOfIntro(_BasePage):
    pass


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
