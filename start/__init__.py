from otree.api import *
import logging, re
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
#  IMAGE URL BUILDER — ALWAYS USES S3, NEVER STATIC
# -------------------------------------------------------------------

def build_s3_url(player, filename: str) -> str:
    """Builds an S3 URL using the settings sheet base."""
    settings = player.session.vars.get('sheet_settings', {})
    base = settings.get('s3path_base')

    if not base:
        # fallback: still return a plausible S3 url
        return f"https://disjunction-experiment-pictures-zas2025.s3.eu-central-1.amazonaws.com/practice/{filename}"

    if base.endswith('/'):
        return base + filename
    return base + '/' + filename


# -------------------------------------------------------------------
#  EXCEL LOADER (loads practice_1, practice_2, ...)
# -------------------------------------------------------------------

def _kv_sheet_to_dict(df) -> dict:
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if not {'name', 'value'}.issubset(df.columns):
        return {}

    out = {}
    for _, r in df.iterrows():
        k = str(r.get('name') or '').strip()
        if k:
            v = r.get('value')
            out[k] = '' if pd.isna(v) else str(v).strip()
    return out


def _load_practices(xlsx_filename: str):
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / xlsx_filename,
        root / 'data' / xlsx_filename,
        root / 'start' / 'data' / xlsx_filename,
    ]
    xlsx_path = next((p for p in candidates if p.exists()), None)

    if not xlsx_path:
        logger.error(f"Excel not found: {xlsx_filename}")
        return {}, {}

    book = pd.read_excel(xlsx_path, sheet_name=None, dtype=str)

    # ---- METADATA (s3path_base, interpreter choices, etc.) ----
    meta = {}
    if "settings" in book:
        meta = _kv_sheet_to_dict(book["settings"])

    # ---- PRACTICE PAGES ----
    practice_settings = {}
    for tab, df in book.items():
        name = tab.lower().strip()
        if not name.startswith("practice_"):
            continue

        kv = _kv_sheet_to_dict(df)
        if not kv:
            continue

        img = kv.get('image', '')
        ext = meta.get('extension', 'png')

        if img and not img.lower().endswith(f".{ext}"):
            img = f"{img}.{ext}"

        # collect right answers in numeric order
        answers = [kv[k] for k in sorted(kv.keys()) if k.startswith('right_answer_')]

        practice_settings[name] = {
            "title": kv.get("title", name.title()),
            "main_text": kv.get("main_text", ""),
            "image": img,
            "right_answer": answers,
        }

    return practice_settings, meta


# -------------------------------------------------------------------
# OTree models
# -------------------------------------------------------------------

class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


def creating_session(subsession):
    session = subsession.session
    cfg = session.config

    xlsx = cfg.get("filename")
    if not xlsx:
        raise RuntimeError("Session config must include a valid 'filename' pointing to the Excel file.")

    ps, meta = _load_practices(xlsx)

    session.vars["practice_settings"] = ps
    session.vars["sheet_settings"] = meta

    ic = meta.get('interpreter_choices')
    if ic:
        session.vars["interpreter_choices"] = [x.strip() for x in ic.split(';')]
    else:
        # fallback: number of answers
        first = list(ps.values())[0]
        n = len(first.get('right_answer', []))
        session.vars["interpreter_choices"] = [f"Choice {i}" for i in range(1, n + 1)]

    session.vars.setdefault("interpreter_title", meta.get("interpreter_title", "Interpretation"))


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    survey_data = models.LongStringField(blank=True)


# -------------------------------------------------------------------
# Base Page
# -------------------------------------------------------------------

class _BasePage(Page):
    pass


# -------------------------------------------------------------------
# Practice Page Template Loader
# -------------------------------------------------------------------

class _PracticePage(_BasePage):
    template_name = 'start/Practice1.html'
    practice_id = None

    @classmethod
    def _settings(cls, player):
        key = f"practice_{cls.practice_id}"
        s = player.session.vars["practice_settings"][key].copy()

        # Build S3 URL
        filename = s.get("image", "")
        s["full_image_path"] = build_s3_url(player, filename)

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


class Consent(_BasePage): pass
class Demographics(_BasePage):
    form_model = 'player'
    form_fields = ['survey_data']
class Instructions(_BasePage): pass
class EndOfIntro(_BasePage): pass


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
