from otree.api import *
import logging
import json
import re

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
    # Field to store practice answers (JSON) for validation
    practice_response = models.LongStringField(blank=True)

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

    filename = filename.strip()
    if not filename.lower().endswith(f".{ext}"):
        filename = f"{filename}.{ext}"

    if s3_base:
        base = s3_base.rstrip("/")
        if "amazonaws.com" in base:
            return f"{base}/practice/{filename}"
        return f"{base}/{filename}"
    
    return filename

# --- VALIDATION HELPERS ---

def validate_medals(player, value, practice_id):
    """
    For P1, P2, P3: Checks if user medals match right_answer_X in Excel.
    Expected Excel format: right_answer_1=1, right_answer_2=0...
    """
    if not value: return "Please make a selection."
    
    try:
        user_answers = json.loads(value) # e.g. [1, 0, 1, 0]
    except:
        return "Invalid data format."

    key = f"Practice{practice_id}"
    settings = player.session.vars["practice_settings"].get(key, {})
    correct_answers = settings.get("right_answers", []) # e.g. [['1'], ['0'], ['1'], ['0']]

    # Compare
    for i, correct_set in enumerate(correct_answers):
        if i >= len(user_answers): break
        
        # correct_set is a list like ['1'] or ['the A']
        # user_answers[i] is raw int/string
        correct_val = correct_set[0] if correct_set else ""
        user_val = str(user_answers[i])
        
        if clean_str(user_val) != clean_str(correct_val):
            return "Incorrect solution. Please try again."

    return None

def validate_sentences_exact(player, value, practice_id):
    """
    For P4: Checks if sentences match exact strings in Excel.
    Expected Excel format: right_answer_1="3; the A"
    """
    if not value: return "Please complete the description."
    
    try:
        user_sentences = json.loads(value) # e.g. [["3", "the A"], ["2", "the B"]]
    except:
        return "Invalid data format."

    key = f"Practice{practice_id}"
    settings = player.session.vars["practice_settings"].get(key, {})
    correct_rows = settings.get("right_answers", []) # e.g. [['3', 'the A'], ['2', 'the B']]

    if len(user_sentences) != len(correct_rows):
        return f"Please produce exactly {len(correct_rows)} sentences."

    for i, row in enumerate(correct_rows):
        # row is e.g. ['3', 'the A']
        user_row = user_sentences[i]
        
        # Check part 1 (Number/Quantifier)
        if len(row) > 0 and clean_str(user_row[0]) != clean_str(row[0]):
            return f"Sentence {i+1}: First part is incorrect."
        
        # Check part 2 (Object)
        if len(row) > 1 and clean_str(user_row[1]) != clean_str(row[1]):
            return f"Sentence {i+1}: Second part is incorrect."

    return None

def validate_sentences_regex(player, value):
    """
    For P6, P7: Checks if sentences match the Main Experiment Regex.
    """
    if not value: return "Please complete the description."
    
    try:
        user_sentences = json.loads(value)
    except:
        return "Invalid data format."

    # Fetch Regex from Session (loaded from Settings sheet)
    regexes = player.session.vars.get("allowed_regex", [])
    if len(regexes) < 2:
        return None # No regex defined, skip validation

    reg1 = re.compile(regexes[0], re.IGNORECASE)
    reg2 = re.compile(regexes[1], re.IGNORECASE)

    for i, row in enumerate(user_sentences):
        # row is [part1, part2]
        if not row or len(row) < 2: continue
        
        val1 = str(row[0]).strip()
        val2 = str(row[1]).strip()

        if not reg1.fullmatch(val1):
            return f"Sentence {i+1}: '{val1}' is not a valid start."
        
        if not reg2.fullmatch(val2):
            return f"Sentence {i+1}: '{val2}' is not a valid ending."

    return None

# -------------------------------------------------------------------
#  SESSION CREATION
# -------------------------------------------------------------------

def creating_session(subsession: BaseSubsession):
    session = subsession.session
    filename = session.config.get("filename")
    if not filename: raise RuntimeError("Session config must include 'filename'")

    payload = get_data(filename)
    settings = payload.get("settings", {})

    # Store Settings
    session.vars['sheet_settings'] = settings
    
    practice_settings = {}
    for k, v in settings.items():
        if k.startswith("Practice") and isinstance(v, dict):
            practice_settings[k] = v
    session.vars["practice_settings"] = practice_settings

    session.vars["allowed_values"] = settings.get("allowed_values", [])
    session.vars["allowed_regex"] = settings.get("allowed_regex", []) # Crucial for P6/7
    session.vars["suffixes"] = settings.get("suffixes", ["solve/d", "exercises"])
    session.vars["interpreter_choices"] = settings.get("interpreter_choices", [])
    session.vars["EndOfIntroText"] = settings.get("EndOfIntroText", "")
    
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
    form_fields = ['practice_response'] # Required for validation

    @classmethod
    def _settings(cls, player: Player):
        key = f"Practice{cls.practice_id}"
        s = player.session.vars["practice_settings"].get(key, {}).copy()
        s["full_image_path"] = build_image_url(player, s.get("image", ""))
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

class Demographics(_BasePage):
    form_model = "player"
    form_fields = ["survey_data"]

class Instructions(_BasePage):
    pass

class Practice1(_PracticePage):
    practice_id = 1
    template_name = "start/Practice1.html"
    @staticmethod
    def error_message(player, values):
        return validate_medals(player, values['practice_response'], 1)

class Practice2(_PracticePage):
    practice_id = 2
    template_name = "start/Practice1.html"
    @staticmethod
    def error_message(player, values):
        return validate_medals(player, values['practice_response'], 2)

class Practice3(_PracticePage):
    practice_id = 3
    template_name = "start/Practice1.html"
    @staticmethod
    def error_message(player, values):
        return validate_medals(player, values['practice_response'], 3)

class Practice4(_PracticePage):
    practice_id = 4
    template_name = "start/Practice4.html"
    @staticmethod
    def error_message(player, values):
        return validate_sentences_exact(player, values['practice_response'], 4)

class Practice5(_PracticePage):
    practice_id = 5 # Used logic from PracticePage for loading
    template_name = "start/Practice5.html"
    
    @staticmethod
    def vars_for_template(player: Player):
        # Practice 5 is special (custom template vars) but we can use helper
        s = _PracticePage._settings(player)
        allowed = player.session.vars.get("allowed_values", [])
        return dict(
            title=s.get("title", "Practice 5"),
            main_text=s.get("main_text", ""),
            image_path=s.get("full_image_path", ""),
            vocab1=allowed[0] if len(allowed) > 0 else [],
            vocab2=allowed[1] if len(allowed) > 1 else [],
        )
    
    @staticmethod
    def error_message(player, values):
        return validate_sentences_exact(player, values['practice_response'], 5)

class Practice6(_PracticePage):
    practice_id = 6
    template_name = "start/Practice6.html"
    @staticmethod
    def error_message(player, values):
        return validate_sentences_regex(player, values['practice_response'])

class Practice7(_PracticePage):
    practice_id = 7
    template_name = "start/Practice7.html"
    @staticmethod
    def error_message(player, values):
        return validate_sentences_regex(player, values['practice_response'])

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
