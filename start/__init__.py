from otree.api import *
import logging
from pathlib import Path
import pandas as pd
import re
import json

logger = logging.getLogger(__name__)

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

# -------------------------------------------------------------------
#  HELPER FUNCTIONS
# -------------------------------------------------------------------

def normalize_key(key):
    """
    Smart Normalizer:
    1. Lowercase everything.
    2. Turn any spaces or underscores into a single underscore.
    """
    if not key: return ""
    k = str(key).lower().strip()
    return re.sub(r'[\s_]+', '_', k)

def _kv_sheet_to_dict(df) -> dict:
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if not {"name", "value"}.issubset(df.columns):
        return {}

    out = {}
    for _, r in df.iterrows():
        raw_key = r.get("name")
        if pd.isna(raw_key) or raw_key == "": continue
        
        key = normalize_key(raw_key)
        v = r.get("value")
        out[key] = "" if pd.isna(v) else str(v).strip()
    return out

def build_s3_url(player, filename: str) -> str:
    if not filename or pd.isna(filename) or str(filename).lower() == "nan": 
        return ""
        
    settings = player.session.vars.get('sheet_settings', {})
    base = settings.get('s3path_base', "")
    ext = settings.get('extension', "png")

    if not filename.lower().endswith(f".{ext}"):
        filename = f"{filename}.{ext}"

    if not base:
        base = "https://disjunction-experiment-pictures-zas2025.s3.eu-central-1.amazonaws.com/practice"
    else:
        base = base.rstrip("/")
        if "amazonaws.com" in base and not base.endswith("/practice"):
            base += "/practice"

    return f"{base}/{filename}"

def _load_practices(xlsx_filename: str):
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

    book = pd.read_excel(xlsx_path, sheet_name=None, dtype=str)

    # 1. Load Settings
    meta = {}
    if "settings" in book:
        settings_df = pd.read_excel(xlsx_path, sheet_name="settings", header=None, dtype=str)
        settings_df = settings_df.rename(columns={0: "name", 1: "value"})
        meta = _kv_sheet_to_dict(settings_df)

    # 2. Load Practices
    practice_settings = {}
    for tab, df in book.items():
        clean_name = normalize_key(tab)
        if not clean_name.startswith("practice"): continue

        try:
            p_id = int(re.search(r'\d+', clean_name).group())
            key_name = f"practice_{p_id}"
        except: continue

        kv = _kv_sheet_to_dict(df)
        if not kv: continue

        img = kv.get("image", "") 
        answers = [kv[k] for k in sorted(kv.keys()) if k.startswith("right_answer")]

        practice_settings[key_name] = {
            "title": kv.get("title", tab),
            "main_text": kv.get("main_text", ""),
            "image": img,
            "right_answer": answers,
        }

    return practice_settings, meta


# -------------------------------------------------------------------
#  SESSION CREATION
# -------------------------------------------------------------------

def creating_session(subsession: BaseSubsession):
    session = subsession.session
    cfg = session.config

    xlsx = cfg.get("filename")
    if not xlsx: raise RuntimeError("Session config must include 'filename'.")

    ps, meta = _load_practices(xlsx)

    session.vars["practice_settings"] = ps
    session.vars["sheet_settings"] = meta

    print(f"DEBUG: Keys found in settings: {list(meta.keys())}")

    # Suffixes
    suffixes = []
    i = 1
    while True:
        key = f"suffix_{i}"
        val = meta.get(key)
        if val: suffixes.append(val)
        else:
            if i > 5: break 
        i += 1
    
    if not suffixes:
        print("WARNING: Suffixes missing. Using defaults.")
        suffixes = ["solve/d", "exercises"]
        
    session.vars["suffixes"] = suffixes

    # Allowed Values
    allowed_values = []
    i = 1
    while True:
        key = f"allowed_values_{i}"
        raw = meta.get(key)
        
        if raw:
            print(f"DEBUG: Found {key}: {raw}")
            allowed_values.append([x.strip() for x in raw.split(";") if x.strip()])
        else:
            if i > 5: break 
            allowed_values.append([])
        i += 1
    
    # Fallback
    if not allowed_values or all(len(x)==0 for x in allowed_values):
        print("WARNING: Allowed Values missing in Excel. Using fallback.")
        av1 = "All; Some; None; Many; Most; Some, but not all; Many, but not most; Not all; Not any; Some did not; Most didn't; etc.".split(";")
        av2 = "the A; the B; the C; the A and B; the A or B; the A and the C; the A, the B, and the C; etc.; all; most; etc.; not any of the; etc.".split(";")
        allowed_values = [
            [x.strip() for x in av1 if x.strip()],
            [x.strip() for x in av2 if x.strip()]
        ]

    session.vars["allowed_values"] = allowed_values
    
    session.vars["EndOfIntroText"] = meta.get("endofintrotext", "")
    
    ic = meta.get("interpreter_choices")
    if ic:
        session.vars["interpreter_choices"] = [x.strip() for x in ic.split(";")]
    else:
        session.vars["interpreter_choices"] = []


# -------------------------------------------------------------------
#  PAGES
# -------------------------------------------------------------------

class _BasePage(Page):
    pass

class _PracticePage(_BasePage):
    practice_id = None
    template_name = None 

    @classmethod
    def _settings(cls, player: Player):
        key = f"practice_{cls.practice_id}"
        s = player.session.vars["practice_settings"].get(key, {}).copy()
        s["full_image_path"] = build_s3_url(player, s.get("image", ""))
        if "right_answer" not in s: s["right_answer"] = []
        return s

    @classmethod
    def vars_for_template(cls, player: Player):
        # DIRECT PASSING TO TEMPLATE
        return dict(
            settings=cls._settings(player),
            allowed_values=player.session.vars.get("allowed_values", []),
            suffixes=player.session.vars.get("suffixes", [])
        )

# --- Page Definitions ---

class Consent(_BasePage):
    @staticmethod
    def before_next_page(player, timeout_happened=False):
        if player.session.config.get("for_prolific"):
            p = player.participant
            p.vars["prolific_id"] = p.label
            p.vars.setdefault("study_id", None)
            p.vars.setdefault("prolific_session_id", None)

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
