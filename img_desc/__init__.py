from otree.api import *
import json
import logging
import time
from django.shortcuts import redirect

logger = logging.getLogger("benzapp.img_desc")

PRODUCER = "P"
INTERPRETER = "I"
STUBURL = "https://app.prolific.co/submissions/complete?cc="

class Constants(BaseConstants):
    name_in_url = "img_desc"
    players_per_group = None
    num_rounds = 80  # This fixes the "End of Intro" skipping issue

    PLACEMENT_ERR = "ERROR_BATCH_PLACEMENT"
    API_ERR = "API_ERROR"
    FALLBACK_URL = STUBURL + PLACEMENT_ERR
    API_ERR_URL = STUBURL + API_ERR

# ----------------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------------
class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    # We store the entire 80-round schedule here as JSON
    # This prevents the "Database Column Missing" errors entirely.
    batch_history = models.LongStringField(initial="[]")
    
    inner_role = models.StringField()
    faulty = models.BooleanField(initial=False)
    feedback = models.LongStringField(label="")
    
    # Decisions
    producer_decision = models.LongStringField()
    interpreter_decision = models.LongStringField()
    
    # Timing
    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)
    
    full_return_url = models.StringField(blank=True)

    def get_current_batch_data(self):
        """Helper to get the dictionary for the current round from JSON."""
        if not self.batch_history: return {}
        try:
            history = json.loads(self.batch_history)
            # Find the row where round_number matches the current round
            for item in history:
                if int(item.get('round_number', 0)) == self.round_number:
                    return item
        except:
            pass
        return {}

    def update_current_batch_data(self, updates: dict):
        """Saves player decisions back into the JSON history."""
        try:
            history = json.loads(self.batch_history)
            found = False
            for item in history:
                if int(item.get('round_number', 0)) == self.round_number:
                    item.update(updates)
                    found = True
            if found:
                self.batch_history = json.dumps(history)
        except:
            pass

    def get_image_url(self):
        data = self.get_current_batch_data()
        img = data.get('image', '').strip()
        
        if not img or img.lower() in ['nan', 'na', 'na_x', 'none', '']:
            return ""
            
        base = self.session.vars.get('s3path_base', '').rstrip('/')
        ext = self.session.vars.get('extension', 'png')
        
        # Clean up path if needed
        if "amazonaws" in base:
            base = base.replace("/practice", "")
            
        clean_name = img.replace(" ", "_")
        if not clean_name.lower().endswith(f".{ext}"):
            clean_name = f"{clean_name}.{ext}"
            
        return f"{base}/{clean_name}"

    def get_sentences_data(self):
        """
        Retrieves sentences.
        If Producer: shows previous sentences for this item.
        If Interpreter: finds the Partner's data and shows it.
        """
        data = self.get_current_batch_data()
        my_role = data.get('role', '')
        
        # 1. If Producer, just show what's in the data (or empty)
        if my_role == PRODUCER:
            raw = data.get('sentences', '[]')
            return json.loads(raw) if raw else []

        # 2. If Interpreter, we need to find the Partner
        partner_id_in_group = int(data.get('partner_id', 0))
        if partner_id_in_group == 0:
            return []

        # Find the specific player object for the partner
        partner = None
        for p in self.subsession.get_players():
            if p.id_in_subsession == partner_id_in_group:
                partner = p
                break
        
        if not partner:
            return []

        # Look at the Partner's history for this specific item
        # (Assuming partner played this item in the same round or previously)
        p_data = partner.get_current_batch_data()
        
        # Security check: Ensure partner was actually a Producer
        if p_data.get('role') == PRODUCER:
            raw = p_data.get('sentences', '[]')
            return json.loads(raw) if raw else []
            
        return []

# ----------------------------------------------------------------------------
# SESSION CREATION
# ----------------------------------------------------------------------------
def _truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

def creating_session(subsession: Subsession):
    session = subsession.session
    
    # Run only once at start of session
    if subsession.round_number != 1:
        return

    filename = session.config.get("filename")
    if not filename: 
        raise RuntimeError("No filename in session config")

    # 1. Load Data (Using your fixed get_data.py)
    from reading_xls.get_data import get_data
    excel_payload = get_data(filename)
    
    records = excel_payload['data'] # This is a list of dicts
    settings = excel_payload['settings']
    
    # 2. Store Settings in Session
    session.vars["s3path_base"] = settings.get("s3path_base", "")
    session.vars["extension"] = settings.get("extension", "png")
    session.vars["prefix"] = settings.get("prefix", "")
    session.vars["interpreter_title"] = settings.get("interpreter_title", "Buy medals:")
    session.vars["caseflag"] = _truthy(settings.get("caseflag"))
    session.vars["suffixes"] = settings.get("suffixes", [])
    session.vars["interpreter_choices"] = settings.get("interpreter_choices", [])
    session.vars["allowed_values"] = settings.get("allowed_values", [])
    session.vars["allowed_regexes"] = settings.get("allowed_regex", [])
    session.vars["instructions_url"] = settings.get("instructions_url", "https://google.com")
    if session.config.get("completion_code"):
        session.vars["completion_code"] = str(session.config["completion_code"])

    # 3. LOGIC: Assign Rows to Players
    # We group rows by 'id' from Excel, sort them, and assign to Players 1..N
    players = subsession.get_players()
    from collections import defaultdict
    
    data_by_id = defaultdict(list)
    
    for r in records:
        # Resolve ID (handle various column names)
        try:
            raw_id = float(r.get('id') or r.get('ID') or r.get('id_in_group') or 0)
            # CRITICAL FIX: Excel 0 -> oTree 1
            pid = int(raw_id) + 1 
        except: continue

        # Resolve Partner ID
        try:
            raw_partner = float(r.get('partner_id') or r.get('Partner_ID') or 0)
            # CRITICAL FIX: Excel 0 -> oTree 1
            partner_pid = int(raw_partner) + 1
        except: partner_pid = 0
        
        # Sort Keys (Exp number, then Round number)
        try: exp_num = int(float(r.get('Exp', 0)))
        except: exp_num = 0
        try: rnd_num = int(float(r.get('group_enumeration') or r.get('Round') or 0))
        except: rnd_num = 0
        
        data_by_id[pid].append({
            'sort_key': (exp_num, rnd_num),
            'batch': exp_num,
            'partner_id': partner_pid,
            'role': str(r.get('role', '')).strip(),
            'condition': str(r.get('Condition', '')).strip(),
            'item_nr': str(r.get('Item.Nr', '')).strip(),
            'image': str(r.get('Item', '')).strip(),
            'sentences': str(r.get('sentences', '')).strip() or "[]",
            'rewards': ''
        })

    # Assign history to each player
    for p in players:
        my_items = data_by_id.get(p.id_in_subsession, [])
        
        # Sort: Exp 0 (Practice) -> Exp 1 (Main)
        my_items.sort(key=lambda x: x['sort_key'])
        
        final_history = []
        for i, item in enumerate(my_items):
            # Renumber rounds to be 1, 2, 3 ... 80
            item['round_number'] = i + 1 
            final_history.append(item)
            
        # Store as JSON string (No database table needed!)
        p.participant.vars['batch_history'] = json.dumps(final_history)

# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
class FaultyCatcher(Page):
    @staticmethod
    def is_displayed(player):
        return player.faulty

    def get(self):
        return redirect(Constants.FALLBACK_URL)

class Q(Page):
    form_model = "player"

    @staticmethod
    def is_displayed(player):
        if player.round_number > Constants.num_rounds: return False
        
        # Load data from participant vars if needed
        if player.batch_history == "[]" and 'batch_history' in player.participant.vars:
            player.batch_history = player.participant.vars['batch_history']

        data = player.get_current_batch_data()
        
        # If no data found for this round:
        if not data:
            # Only mark faulty if it's round 1 (prevents crashes at end of game)
            if player.round_number == 1: player.faulty = True
            return False

        player.inner_role = data.get('role', '')
        
        # Start Timer
        if player.start_decision_time == 0:
            player.start_decision_time = time.time()
            
        return True

    @staticmethod
    def get_form_fields(player):
        if player.inner_role == PRODUCER: return ["producer_decision"]
        if player.inner_role == INTERPRETER: return ["interpreter_decision"]
        return []

    @staticmethod
    def vars_for_template(player):
        return dict(
            d=player.get_current_batch_data(),
            allowed_values=player.session.vars.get("allowed_values"),
            allowed_regexes=player.session.vars.get("allowed_regexes"),
            suffixes=player.session.vars.get("suffixes"),
            prefix=player.session.vars.get("prefix"),
            interpreter_choices=player.session.vars.get("interpreter_choices"),
            interpreter_title=player.session.vars.get("interpreter_title"),
            instructions_url=player.session.vars.get("instructions_url"),
            server_image_url=player.get_image_url(),
            caseflag=player.session.vars.get("caseflag"),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.end_decision_time = time.time()
        if player.start_decision_time:
            player.decision_seconds = player.end_decision_time - player.start_decision_time
        
        # Save decisions to local history
        updates = {}
        if player.inner_role == PRODUCER:
            updates['sentences'] = player.producer_decision
        elif player.inner_role == INTERPRETER:
            updates['rewards'] = player.interpreter_decision
            
        player.update_current_batch_data(updates)
        
        # Save back to participant so it persists
        player.participant.vars['batch_history'] = player.batch_history

class Feedback(Page):
    form_model = "player"
    form_fields = ["feedback"]

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.num_rounds

class FinalForProlific(Page):
    @staticmethod
    def is_displayed(player):
        return player.session.config.get("for_prolific") and player.round_number == Constants.num_rounds

    def get(self):
        cc = (player.session.vars.get("completion_code") or 
              player.session.config.get("completion_code"))
        if not cc: return redirect(Constants.API_ERR_URL)
        return redirect(STUBURL + str(cc))

# ----------------------------------------------------------------------------
# EXPORT (Simplified & Robust)
# ----------------------------------------------------------------------------
def custom_export(players):
    # Header
    yield [
        "session_code", "participant_code", "round_number", "role",
        "condition", "item_nr", "image", "producer_sentences",
        "interpreter_rewards", "decision_seconds", "feedback"
    ]

    processed_participants = set()
    
    for p in players:
        if p.participant.code in processed_participants:
            continue
        processed_participants.add(p.participant.code)
        
        # Get the full 80-round history from the participant
        history_json = p.participant.vars.get('batch_history', '[]')
        try:
            history = json.loads(history_json)
        except:
            history = []
            
        # Get timing from the actual played rounds
        timing_map = {}
        feedback_str = ""
        for sub_p in p.participant.get_players():
            timing_map[sub_p.round_number] = sub_p.decision_seconds
            if sub_p.round_number == Constants.num_rounds:
                feedback_str = sub_p.feedback or ""

        # Sort by round
        history.sort(key=lambda x: int(x.get('round_number', 0)))
        
        for item in history:
            rnd = int(item.get('round_number', 0))
            if rnd < 1 or rnd > Constants.num_rounds: continue
            
            yield [
                p.session.code,
                p.participant.code,
                rnd,
                item.get('role', ''),
                item.get('condition', ''),
                item.get('item_nr', ''),
                item.get('image', ''),
                item.get('sentences', ''),
                item.get('rewards', ''),
                timing_map.get(rnd, 0),
                feedback_str if rnd == Constants.num_rounds else ""
            ]

page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
