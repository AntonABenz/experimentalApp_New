from otree.api import *
import logging, re
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
#  IMAGE URL BUILDER — FIXED FOR S3
# -------------------------------------------------------------------

def build_s3_url(player, filename: str) -> str:
    """
    Build final S3 URL for images.

    Rules:
    - read base from Excel settings sheet
    - if sheet gives only bucket root, append /practice
    - fallback always points to .../practice/<filename>
    """
    settings = player.session.vars.get('sheet_settings', {})
    base = settings.get('s3path_base')

    default_base = (
        "https://disjunction-experiment-pictures-zas2025."
        "s3.eu-central-1.amazonaws.com/practice"
    )

    # No base provided → fallback to default (previous behavior)
    if not base:
        base = default_base
    else:
        base = base.rstrip("/")

        # If user gives bucket root, automatically append /practice
        if base.endswith(".amazonaws.com"):
            base = base + "/practice"

    return f"{base}/{filename}"


# -------------------------------------------------------------------
#  KEY/VALUE SHEET PARSER
# -------------------------------------------------------------------

def _kv_sheet_to_dict(df) -> dict:
    """Parses sheets with columns: name | value | comment (optional)."""
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

    if not {"name", "value"}.issubset(df.columns):
        return {}

    out = {}
    for _, r in df.iterrows():
        key = str(r.get("name") or "").strip()
        if key:
            v = r.get("value")
            out[key] = "" if pd.isna(v) else str(v).strip()

    return out


# -------------------------------------------------------------------
#  LOAD PRACTICE SHEETS + SETTINGS FROM EXCEL
# -------------------------------------------------------------------

def _load_practices(xlsx_filename: str):
    """Loads settings sheet + all practice_# sheets."""
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / xlsx_filename,
        root / "data" / xlsx_filename,
        root / "start" / "data" / xlsx_filename,
    ]

    xlsx_path = next((p for p in candidates if p.exists()), None)
    if not xlsx_path:
        logger.error(f"Excel not found: {xlsx_filename}")
        return {}, {}

    # Load all sheets
    book = pd.read_excel(xlsx_path, sheet_name=None, dtype=str)

    # ---- SETTINGS SHEET ----
    meta = {}
    if "settings" in book:
        # Settings sheet does NOT have headers → read manually
        settings_df = pd.read_excel(
            xlsx_path,
            sheet_name="settings",
            header=None,
            dtype=str,
        )
        # Force columns into name/value
        settings_df = settings_df.rename(columns={0: "name", 1: "value"})
        meta = _kv_sheet_to_dict(settings_df)

    # ---- PRACTICE SHEET LOADING ----
    practice_settings = {}
    for tab, df in book.items():
        name = tab.lower().strip()
        if not name.startswith("practice_"):
            continue

        # practices DO have name/value/comment header row
        kv = _kv_sheet_to_dict(df)
        if not kv:
            continue

        img = kv.get("image", "")
        ext = meta.get("extension", "png")

        # append extension if needed
        if img and not img.lower().endswith(f".{ext}"):
            img = f"{img}.{ext}"

        # collect right answers in numeric order
        answers = [kv[k] for k in sorted(kv.keys()) if k.startswith("right_answer_")]

        practice_settings[name] = {
            "title": kv.get("title", name.title()),
            "main_text": kv.get("main_text", ""),
            "image": img,
            "right_answer": answers,
        }

    return practice_settings, meta


# -------------------------------------------------------------------
#  OTree MODELS
# -------------------------------------------------------------------

class C(BaseConstants):
    NAME_IN_URL = "start"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


def creating_session(subsession):
    session = subsession.session
    cfg = session.config

    xlsx = cfg.get("filename")
    if not xlsx:
        raise RuntimeError(
            "Session config must include 'filename' pointing to the Excel file."
        )

    # Load everything
    ps, meta = _load_practices(xlsx)

    # Save for use in templates
    session.vars["practice_settings"] = ps
    session.vars["sheet_settings"] = meta

    # INTERPRETER CHOICES
    ic = meta.get("interpreter_choices")
    if ic:
        session.vars["interpreter_choices"] = [x.strip() for x in ic.split(";")]
    else:
        # fallback: number of answers in first practice sheet
        first = list(ps.values())[0]
        n = len(first.get("right_answer", []))
        session.vars["interpreter_choices"] = [f"Choice {i}" for i in range(1, n + 1)]

    # interpreter title
    session.vars.setdefault(
        "interpreter_title", meta.get("interpreter_title", "Interpretation")
    )


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    survey_data = models.LongStringField(blank=True)


# -------------------------------------------------------------------
#  BASE PAGE + PRACTICE PAGE LOADER
# -------------------------------------------------------------------

class _BasePage(Page):
    pass


class _PracticePage(_BasePage):
    practice_id = None
    template_name = None  # subclasses must set this

    @classmethod
    def _settings(cls, player):
        key = f"practice_{cls.practice_id}"
        s = player.session.vars["practice_settings"][key].copy()

        filename = s.get("image", "")
        s["full_image_path"] = build_s3_url(player, filename)

        return s

    @classmethod
    def vars_for_template(cls, player):
        return dict(settings=cls._settings(player))

    @classmethod
    def js_vars(cls, player):
        return dict(settings=cls._settings(player))


# Practice 1–3: medals, Yes/No layout (Practice1.html)
class Practice1(_PracticePage):
    practice_id = 1
    template_name = "start/Practice1.html"


class Practice2(_PracticePage):
    practice_id = 2
    template_name = "start/Practice1.html"


class Practice3(_PracticePage):
    practice_id = 3
    template_name = "start/Practice1.html"


# Practice 4–7: describing results with text fields (Practice4.html)
class Practice4(_PracticePage):
    practice_id = 4
    template_name = "start/Practice4.html"


class Practice5(_PracticePage):
    practice_id = 5
    template_name = "start/Practice4.html"


class Practice6(_PracticePage):
    practice_id = 6
    template_name = "start/Practice4.html"


class Practice7(_PracticePage):
    practice_id = 7
    template_name = "start/Practice4.html"


class Consent(_BasePage):
    pass


class Demographics(_BasePage):
    form_model = "player"
    form_fields = ["survey_data"]


class Instructions(_BasePage):
    pass


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
