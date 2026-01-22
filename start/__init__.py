from otree.api import *
import logging
import json
import re

from reading_xls.get_data import get_data

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
#  HELPER FUNCTIONS
# -------------------------------------------------------------------

def clean_str(x):
    if x is None: return ""
    return str(x).strip()

def build_image_url(player, filename: str) -> str:
    """Constructs the full S3 URL for an image."""
    if not filename or filename.lower() in ["nan", "none", ""]: 
        return ""
    
    s3_base = player.session.vars.get('s3path_base', "")
    ext = player.session.vars.get('extension', "png")
    filename = filename.strip()
    
    # Ensure extension
    if not filename.lower().endswith(f".{ext}"):
        filename = f"{filename}.{ext}"

    if s3_base:
        base = s3_base.rstrip("/")
        # Practice images are usually in /practice subfolder
        if "amazonaws.com" in base:
            return f"{base}/practice/{filename}"
        return f"{base}/{filename}"
    return filename

def _get_right_answers_list(practice_data: dict) -> list:
    """Parses right_answer_1..N into a list."""
    answers = []
    keys = [k for k in practice_data.keys() if str(k).lower().startswith("right_answer_")]
    
    def extract_num(k):
        nums = re.findall(r'\d+', str(k))
        return int(nums[0]) if nums else 999
    keys.sort(key=extract_num)
    
    for k in keys:
        val = clean_str(practice_data.get(k))
        if val:
            answers.append(val)
    return answers

# -------------------------------------------------------------------
#  MODELS
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
#  SESSION CREATION
# -------------------------------------------------------------------

def creating_session(subsession: BaseSubsession):
    session = subsession.session
    filename = session.config.get("filename")
    if not filename: raise RuntimeError("Session config must include 'filename'")

    payload = get_data(filename)
    settings = payload.get("settings", {})

    session.vars['sheet_settings'] = settings
    
    # Process Practice Tabs
    practice_settings = {}
    for k, v in settings.items():
        if k.startswith("Practice") and isinstance(v, dict):
            p_data = v.copy()
            p_data['right_answer'] = _get_right_answers_list(p_data)
            practice_settings[k] = p_data
            
    session.vars["practice_settings"] = practice_settings

    # Global Lists
    session.vars["allowed_values"] = settings.get("allowed_values", [])
    session.vars["allowed_regex"] = settings.get("allowed_regex", [])
    session.vars["suffixes"] = settings.get("suffixes", ["solve/d", "exercises"])
    session.vars["interpreter_choices"] = settings.get("interpreter_choices", [])
    session.vars["interpreter_title"] = settings.get("interpreter_title", "Interpretation")
    session.vars["EndOfIntroText"] = settings.get("EndOfIntroText", "")
    
    # S3
    raw_s3 = str(settings.get("s3path") or settings.get("s3path_base") or "")
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            bucket = raw_s3.split("buckets/")[1].split("?")[0].strip("/")
            raw_s3 = f"https://{bucket}.s3.eu-central-1.amazonaws.com"
        except: pass
    session.vars['s3path_base'] = raw_s3
    session.vars['extension'] = settings.get("extension", "png")

# -------------------------------------------------------------------
#  PAGES
# -------------------------------------------------------------------

class _BasePage(Page):
    pass

class _PracticePage(_BasePage):
    practice_id = None
    template_name = None 
    form_model = 'player'
    form_fields = ['practice_response']

    @classmethod
    def is_displayed(cls, player: Player):
        key = f"Practice{cls.practice_id}"
        # Check toggle from sheet settings (default 1)
        sheet_settings = player.session.vars.get("sheet_settings", {})
        flag = str(sheet_settings.get(key, "1")).strip().lower()
        enabled = flag in ["1", "true", "yes"]
        # Check if data exists
        has_data = key in player.session.vars.get("practice_settings", {})
        return enabled and has_data

    @classmethod
    def _settings(cls, player: Player):
        key = f"Practice{cls.practice_id}"
        s = player.session.vars["practice_settings"].get(key, {}).copy()
        s["full_image_path"] = build_image_url(player, s.get("image", ""))
        
        # Set required rows based on Practice ID
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
            js_right_answers=json.dumps(s.get("right_answer", [])),
            js_regex=json.dumps(player.session.vars.get("allowed_regex", [])),
            allowed_values=player.session.vars.get("allowed_values", []),
            suffixes=player.session.vars.get("suffixes", [])
        )

# --- Page Definitions ---

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
    practice_id = 5
    template_name = "start/Practice5.html"
    
    @staticmethod
    def vars_for_template(player: Player):
        s = _PracticePage._settings(player)
        allowed = player.session.vars.get("allowed_values", [])
        return dict(
            settings=s,
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
