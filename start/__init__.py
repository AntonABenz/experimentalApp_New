from otree.api import *
import logging
from pathlib import Path
import pandas as pd
import re

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
    Ultra-robust key normalizer.
    Removes ALL spaces, underscores, and converts to lowercase.
    Example: "Allowed_Values_1" -> "allowedvalues1"
    """
    if not key: return ""
    return str(key).lower().replace("_", "").replace(" ", "").strip()

def _kv_sheet_to_dict(df) -> dict:
    # Standardize column headers to lower case
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

    # Basic check for 'name' and 'value' columns
    if not {"name", "value"}.issubset(df.columns):
        return {}

    out = {}
    for _, r in df.iterrows():
        raw_key = r.get("name")
        if pd.isna(raw_key) or raw_key == "":
            continue
            
        # Apply ultra-robust normalization
        key = normalize_key(raw_key)
        
        v = r.get("value")
        out[key] = "" if pd.isna(v) else str(v).strip()

    return out

def build_s3_url(player, filename: str) -> str:
    """Robust S3 URL builder."""
    if not filename or pd.isna(filename) or str(filename).lower() == "nan": 
        return ""
        
    settings = player.session.vars.get('sheet_settings', {})
    # Look for normalized key 's3pathbase' (from s3path_base)
    base = settings.get('s3pathbase', "")
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
        # Load without header first to inspect
        settings_df = pd.read_excel(xlsx_path, sheet_name="settings", header=None, dtype=str)
        # Assume col 0 is name, col 1 is value
        settings_df = settings_df.rename(columns={0: "name", 1: "value"})
        meta = _kv_sheet_to_dict(settings_df)

    # 2. Load Practices
    practice_settings = {}
    for tab, df in book.items():
        # Normalize sheet name: "Practice 1" -> "practice1"
        clean_name = normalize_key(tab)
        
        if not clean_name.startswith("practice"):
            continue

        try:
            p_id = int(re.search(r'\d+', clean_name).group())
            key_name = f"practice_{p_id}"
        except:
            continue

        kv = _kv_sheet_to_dict(df)
        if not kv: continue

        img = kv.get("image", "") # will be normalized key 'image'
        # Collect right answers (normalized keys like rightanswer1)
        answers = [kv[k] for k in sorted(kv.keys()) if k.startswith("rightanswer")]

        practice_settings[key_name] = {
            "title": kv.get("title", tab),
            "main_text": kv.get("maintext", ""), # normalized key
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
    if not xlsx:
        raise RuntimeError("Session config must include 'filename'.")

    ps, meta = _load_practices(xlsx)

    session.vars["practice_settings"] = ps
    session.vars["sheet_settings"] = meta

    # --- DEBUGGING PRINTS ---
    print("--- LOADING SESSION DATA ---")
    print(f"Meta keys found: {list(meta.keys())}")

    # Parse Suffixes (normalized key: suffix1)
    suffixes = []
    i = 1
    while True:
        key = f"suffix{i}" # matches normalize_key("suffix_1")
        val = meta.get(key)
        if val: suffixes.append(val)
        else:
            if i > 5: break 
        i += 1
    session.vars["suffixes"] = suffixes

    # Parse Allowed Values (normalized key: allowedvalues1)
    allowed_values = []
    i = 1
    while True:
        key = f"allowedvalues{i}" # matches normalize_key("Allowed_Values_1")
        raw = meta.get(key)
        
        # DEBUG PRINT
        if raw: 
            print(f"Found {key}: {raw}")
            allowed_values.append([x.strip() for x in raw.split(";") if x.strip()])
        else:
            print(f"Missing {key}")
            allowed_values.append([])
            if i > 20: break 
        i += 1
    
    session.vars["allowed_values"] = allowed_values
    print(f"Final Allowed Values: {allowed_values}")
    print("--------------------------")

    session.vars["EndOfIntroText"] = meta.get("endofintrotext", "")
    
    ic = meta.get("interpreterchoices")
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

        # Build secure URL
        filename = s.get("image", "")
        s["full_image_path"] = build_s3_url(player, filename)
        
        if "right_answer" not in s:
            s["right_answer"] = []
        return s

    @classmethod
    def vars_for_template(cls, player: Player):
        return dict(settings=cls._settings(player))

    @classmethod
    def js_vars(cls, player: Player):
        # THIS IS THE CRITICAL FIX: Passing vars to the template
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
    template_name = "start/Practice4.html"

class Practice6(_PracticePage):
    practice_id = 6
    template_name = "start/Practice6.html"

class Practice7(_PracticePage):
    practice_id = 7
    template_name = "start/Practice7.html"

class EndOfIntro(_BasePage):
    def vars_for_template(self):
        return dict(
            end_of_intro_text=self.session.vars.get("EndOfIntroText", "")
        )

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
