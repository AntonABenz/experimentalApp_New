from otree.api import *
import json
import logging
import os
import re
import time

# We use pandas for robust Excel reading
import pandas as pd
from django.shortcuts import redirect

logger = logging.getLogger("benzapp.img_desc")

PRODUCER = "P"
INTERPRETER = "I"

STUBURL = "https://app.prolific.co/submissions/complete?cc="


class Constants(BaseConstants):
    name_in_url = "img_desc"
    players_per_group = None
    num_rounds = 80

    PLACEMENT_ERR = "ERROR_BATCH_PLACEMENT"
    API_ERR = "API_ERROR"
    FALLBACK_URL = STUBURL + PLACEMENT_ERR
    API_ERR_URL = STUBURL + API_ERR


def normalize_key(key):
    if not key:
        return ""
    return re.sub(r"[\s_]+", "_", str(key).lower().strip())


def _truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_excel_strict(filename):
    """
    Robust Excel reader that looks in multiple paths and handles data/settings sheets.
    """
    from pathlib import Path
    
    # 1. Resolve Path
    candidates = [
        Path(filename),
        Path("start/data") / filename,
        Path("/app/start/data") / filename,
        Path("data") / filename,
    ]
    
    xlsx_path = None
    for p in candidates:
        if p.exists():
            xlsx_path = p
            break
            
    if not xlsx_path:
        raise FileNotFoundError(f"Excel file '{filename}' not found. Searched: {[str(c) for c in candidates]}")

    logger.info(f"LOADING EXCEL FROM: {xlsx_path}")

    # 2. Read Sheets
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    sheet_names = [s.lower().strip() for s in xls.sheet_names]
    
    data_sheet = xls.sheet_names[0] # Default to first
    settings_sheet = None
    
    for i, s in enumerate(sheet_names):
        if "setting" in s:
            settings_sheet = xls.sheet_names[i]
        if s in {"data", "trials", "items"}:
            data_sheet = xls.sheet_names[i]

    # 3. Parse Data
    df = pd.read_excel(xlsx_path, sheet_name=data_sheet, dtype=str, keep_default_na=False, engine="openpyxl")
    
    # 4. Parse Settings
    settings = {}
    if settings_sheet:
        raw = pd.read_excel(xlsx_path, sheet_name=settings_sheet, dtype=str, keep_default_na=False, header=None, engine="openpyxl")
        # Assume Col A is Key, Col B is Value
        if len(raw.columns) >= 2:
            for _, row in raw.iterrows():
                k, v = str(row[0]).strip(), str(row[1]).strip()
                if k: settings[k] = v

    return df, settings


# ============================================================================
# OTREE MODELS
# ============================================================================

class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # --- State ---
    # We store the ENTIRE batch history for this player as a JSON string.
    # This replaces the database table.
    batch_history = models.LongStringField(initial="[]")
    
    # Current round pointers
    inner_role = models.StringField()
    active_batch_id = models.IntegerField(initial=0) # Index in batch_history

    # --- Metrics ---
    faulty = models.BooleanField(initial=False)
    feedback = models.LongStringField(label="")
    
    # --- Timing & Decisions ---
    producer_decision = models.LongStringField()
    interpreter_decision = models.LongStringField()
    
    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)
    
    full_return_url = models.StringField(blank=True)

    def get_current_batch_data(self):
        """Helper to get the dict for the current round from JSON."""
        if not self.batch_history: return {}
        try:
            history = json.loads(self.batch_history)
            # Find entry for this round
            for item in history:
                if int(item.get('round_number', 0)) == self.round_number:
                    return item
        except:
            pass
        return {}

    def update_current_batch_data(self, updates: dict):
        """Updates the JSON history with new decisions."""
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
        
        # Ghost file trap
        if "ABC-3" in img:
            logger.warning(f"⚠️ GHOST ITEM DETECTED for {self.participant.code}: {img}")

        if not img or img.lower() in ['nan', 'na', 'na_x', 'none']:
            return ""
            
        base = self.session.vars.get('s3path_base', '').rstrip('/')
        ext = self.session.vars.get('extension', 'png')
        
        if "amazonaws" in base:
            base = base.replace("/practice", "")
            
        clean_name = img.replace(" ", "_")
        if not clean_name.lower().endswith(f".{ext}"):
            clean_name = f"{clean_name}.{ext}"
            
        return f"{base}/{clean_name}"

    def get_sentences_data(self):
        """
        Logic:
        - If Producer: Show my own previous sentences (from Excel or previous save).
        - If Interpreter: Find my PARTNER from the previous batch and show THEIR sentences.
        """
        data = self.get_current_batch_data()
        my_role = data.get('role', '')
        
        # 1. Producer sees what is in the data (or empty)
        if my_role == PRODUCER:
            raw = data.get('sentences', '[]')
            return json.loads(raw) if raw else []

        # 2. Interpreter needs to find partner
        # Partner ID in Excel refers to the partner's 'id' (id_in_group)
        partner_id_in_group = int(data.get('partner_id', 0))
        if partner_id_in_group == 0:
            return []

        # Find the player object who has that id_in_group
        # (Note: id_in_subsession is usually aligned with excel id)
        # We search specifically:
        partner = None
        for p in self.subsession.get_players():
            # We assume we stored 'id' from Excel in participant.vars or calculated it
            # Simple assumption: player.id_in_subsession maps to Excel 'id'
            if p.id_in_subsession == partner_id_in_group:
                partner = p
                break
        
        if not partner:
            return []

        # Now get the Partner's history for the PREVIOUS batch (round - 1?)
        # Actually, standard design is: Prod Round 1 -> Interp Round 1.
        # But usually Interpreter describes the Producer's result from the SAME round or PREV?
        # Based on your logic: "prev = get_previous_batch()".
        # We will look for the partner's data in the SAME round logic or matching batch logic.
        
        # Let's assume standard: Partner produced sentences in THIS round's data
        # Check partner's history for this round
        p_data = partner.get_current_batch_data()
        
        # Safety check: Partner must be Producer
        if p_data.get('role') == PRODUCER:
            raw = p_data.get('sentences', '[]')
            return json.loads(raw) if raw else []
            
        return []


# ============================================================================
# SESSION CREATION
# ============================================================================

def creating_session(subsession: Subsession):
    session = subsession.session
    
    # Only run once per session
    if subsession.round_number != 1:
        return

    filename = session.config.get("filename")
    if not filename: raise RuntimeError("No filename in session config")

    # 1. Load Excel
    df, settings_raw = _read_excel_strict(filename)
    
    # 2. Parse Settings
    clean_settings = {normalize_key(k): v for k, v in settings_raw.items()}
    
    session.vars["s3path_base"] = clean_settings.get("s3path_base", "")
    session.vars["extension"] = clean_settings.get("extension", "png")
    session.vars["prefix"] = clean_settings.get("prefix", "")
    session.vars["interpreter_title"] = clean_settings.get("interpreter_title", "Buy medals:")
    session.vars["caseflag"] = _truthy(clean_settings.get("caseflag"))
    
    # Suffixes & Choices
    session.vars["suffixes"] = [
        str(clean_settings.get(f"suffix_{i}", "")).strip() 
        for i in range(1, 11) 
        if clean_settings.get(f"suffix_{i}")
    ]
    
    raw_choices = clean_settings.get("interpreter_choices", "")
    session.vars["interpreter_choices"] = [x.strip() for x in raw_choices.split(";")] if raw_choices else []

    # Regex/Values
    allowed_vals = []
    allowed_reg = []
    for i in range(1, 21):
        v = clean_settings.get(f"allowed_values_{i}")
        r = clean_settings.get(f"allowed_regex_{i}")
        if v or r:
            allowed_vals.append([x.strip() for x in str(v).split(";")] if v else [])
            allowed_reg.append(str(r).strip() if r else "")
    session.vars["allowed_values"] = allowed_vals
    session.vars["allowed_regexes"] = allowed_reg
    
    # Instructions & Prolific
    session.vars["instructions_url"] = clean_settings.get("instructions_url", "https://google.com")
    if session.config.get("completion_code"):
        session.vars["completion_code"] = str(session.config["completion_code"])

    # 3. DISTRIBUTE DATA TO PLAYERS
    # We assign rows based on 'id' in Excel matching 'id_in_subsession' of players.
    # This creates a static, crash-proof assignment.
    
    records = df.to_dict(orient="records")
    players = subsession.get_players()
    
    # Group records by 'id' (Excel column 'id')
    from collections import defaultdict
    data_by_id = defaultdict(list)
    
    for r in records:
        pid = int(float(r.get('id', 0))) # Handle "1.0" strings
        data_by_id[pid].append({
            'round_number': int(float(r.get('group_enumeration', 0))),
            'batch': int(float(r.get('Exp', 0))),
            'role': str(r.get('role', '')).strip(),
            'partner_id': int(float(r.get('partner_id', 0))),
            'condition': str(r.get('Condition', '')).strip(),
            'item_nr': str(r.get('Item.Nr', '')).strip(),
            'image': str(r.get('Item', '')).strip(),
            'sentences': str(r.get('sentences', '')).strip() or "[]",
            'rewards': ''
        })

    # Assign to players
    for p in players:
        # Get data for this player ID
        my_data = data_by_id.get(p.id_in_subsession, [])
        if not my_data:
            # Fallback: if we have more players than IDs, cycle? or leave empty
            logger.warning(f"No data found for Player {p.id_in_subsession} in Excel!")
        
        # Save as JSON on the Player object
        # IMPORTANT: We save this to Round 1 player, but oTree doesn't auto-copy to future rounds.
        # We must copy it in 'creating_session' for all rounds OR use participant.vars.
        # Using participant.vars is safer for multi-round access.
        p.participant.vars['batch_history'] = json.dumps(my_data)


# ============================================================================
# PAGES
# ============================================================================

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
        
        # 1. Load Data from Participant vars if not present
        if player.batch_history == "[]" and 'batch_history' in player.participant.vars:
            player.batch_history = player.participant.vars['batch_history']

        # 2. Get Current Round Data
        data = player.get_current_batch_data()
        if not data:
            # If no data for this round, skip or faulty
            # For robustness, we mark faulty only if really empty
            if player.round_number == 1: player.faulty = True
            return False

        # 3. Set Role for Template
        player.inner_role = data.get('role', '')
        
        # 4. Start Timer
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
        
        # Save decision to local JSON history
        updates = {}
        if player.inner_role == PRODUCER:
            updates['sentences'] = player.producer_decision
        elif player.inner_role == INTERPRETER:
            updates['rewards'] = player.interpreter_decision
            
        player.update_current_batch_data(updates)
        
        # Sync back to participant vars so future rounds see it!
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


# ============================================================================
# EXPORT
# ============================================================================

def custom_export(players):
    """
    Export 80 rows per participant (from JSON history) + Time/Feedback.
    """
    yield [
        "session_code", "participant_code", "round_number", "role",
        "condition", "item_nr", "image", "producer_sentences",
        "interpreter_rewards", "decision_seconds", "feedback"
    ]

    # Iterate over players (only look at Round 80 to get full history, OR iterate all)
    # Better: iterate all players, get their batch_history from participant vars
    
    # We grab unique participants
    processed_participants = set()
    
    for p in players:
        if p.participant.code in processed_participants:
            continue
        processed_participants.add(p.participant.code)
        
        # Get history
        history_json = p.participant.vars.get('batch_history', '[]')
        try:
            history = json.loads(history_json)
        except:
            history = []
            
        # Also need timing data per round. This is stored on the Player objects.
        # We need to map round_number -> decision_seconds
        timing_map = {}
        feedback_str = ""
        
        # Get all rounds for this participant
        my_rounds = p.participant.get_players()
        for sub_p in my_rounds:
            timing_map[sub_p.round_number] = sub_p.decision_seconds
            if sub_p.round_number == Constants.num_rounds:
                feedback_str = sub_p.feedback or ""

        # Now yield 80 rows
        # Sort history by round number just in case
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
