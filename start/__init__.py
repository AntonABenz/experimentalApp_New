from otree.api import *
import logging
import re
import json

# IMPORT THE ROBUST LOADER
from reading_xls.get_data import get_data

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

def clean_str(x):
    if x is None: return ""
    return str(x).strip()

def build_image_url(player, filename: str) -> str:
    if not filename or filename.lower() in ["nan", "none", ""]: 
        return ""
        
    s3_base = player.session.vars.get('s3path_base', "")
    ext = player.session.vars.get('extension', "png")

    # Clean filename
    filename = filename.strip()
    if not filename.lower().endswith(f".{ext}"):
        filename = f"{filename}.{ext}"

    # Handle S3 base vs Practice subfolder
    # Images in "start" app usually live in /practice folder
    if s3_base:
        base = s3_base.rstrip("/")
        if "amazonaws.com" in base:
            # Ensure we point to the practice folder
            return f"{base}/practice/{filename}"
        return f"{base}/{filename}"
    
    return filename

# -------------------------------------------------------------------
#  SESSION CREATION
# -------------------------------------------------------------------

def creating_session(subsession: BaseSubsession):
    session = subsession.session
    
    # 1. Load Data using the shared robust loader
    #    (This handles Google Sheet download automatically)
    filename = session.config.get("filename")
    if not filename: 
        raise RuntimeError("Session config must include 'filename'")

    payload = get_data(filename)
    settings = payload.get("settings", {})

    # 2. Store Settings in Session
    #    (Practice sheets are already dicts inside 'settings' from get_data)
    session.vars['sheet_settings'] = settings
    
    # Extract Practice Configs specifically for easy access
    practice_settings = {}
    for k, v in settings.items():
        if k.startswith("Practice") and isinstance(v, dict):
            practice_settings[k] = v
    session.vars["practice_settings"] = practice_settings

    # 3. Process Config Lists (Suffixes, etc)
    #    (get_data already processed allowed_values/regex into lists in settings)
    session.vars["allowed_values"] = settings.get("allowed_values", [])
    session.vars["suffixes"] = settings.get("suffixes", ["solve/d", "exercises"])
    session.vars["interpreter_choices"] = settings.get("interpreter_choices", [])
    
    # Misc
    session.vars["EndOfIntroText"] = settings.get("EndOfIntroText", "")
    
    # Global S3 Base
    raw_s3 = str(settings.get("s3path") or settings.get("s3path_base") or "")
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            bucket = raw_s3.split("buckets/")[1].split("?")[0].strip("/")
            raw_s3 = f"https://{bucket}.s3.eu-central-1.amazonaws.com"
        except: pass
    session.vars['s3path_base'] = raw_s3
    session.vars['extension'] = settings.get("extension", "png")

    print(f"Start App: Loaded {len(practice_settings)} practice pages.")


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
        # Look for "Practice1", "Practice2" (Capitalized, no underscore)
        # matching the normalization in get_data.py
        key = f"Practice{cls.practice_id}"
        
        # Fallback for old naming styles just in case
        if key not in player.session.vars["practice_settings"]:
            # Try "Practice 1" or "practice_1" logic if needed
            pass

        s = player.session.vars["practice_settings"].get(key, {}).copy()
        
        # Build image URL dynamically
        img_name = s.get("image", "")
        s["full_image_path"] = build_image_url(player, img_name)
        
        # Ensure right_answer is a list
        # (get_data might supply a string, we might need to parse it if get_data didn't)
        ra = s.get("right_answer")
        if not isinstance(ra, list):
             s["right_answer"] = [] # handled in template or JS
             
        return s

    @classmethod
    def vars_for_template(cls, player: Player):
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

class Practice5(_BasePage):
    template_name = "start/Practice5.html"

    @staticmethod
    def vars_for_template(player: Player):
        # reuse the same settings structure
        key = "Practice5" # Note capitalization to match get_data
        s = player.session.vars.get("practice_settings", {}).get(key, {}).copy()
        
        img = s.get("image", "")
        full_image_path = build_image_url(player, img)

        allowed = player.session.vars.get("allowed_values", [])
        vocab1 = allowed[0] if len(allowed) > 0 else []
        vocab2 = allowed[1] if len(allowed) > 1 else []

        return dict(
            title=s.get("title", "Practice 5"),
            main_text=s.get("main_text", ""),
            image_path=full_image_path,
            vocab1=vocab1,
            vocab2=vocab2,
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
