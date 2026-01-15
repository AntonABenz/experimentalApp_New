from otree.api import *
import json
import logging
import time
import re
import random
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

# ----------------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------------
class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    batch_history = models.LongStringField(initial="[]")
    
    inner_role = models.StringField()
    faulty = models.BooleanField(initial=False)
    feedback = models.LongStringField(label="")
    
    producer_decision = models.LongStringField()
    interpreter_decision = models.LongStringField()
    
    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)
    
    full_return_url = models.StringField(blank=True)

    def get_current_batch_data(self):
        if not self.batch_history: return {}
        try:
            history = json.loads(self.batch_history)
            for item in history:
                if int(item.get('round_number', 0)) == self.round_number:
                    return item
        except:
            pass
        return {}

    def update_current_batch_data(self, updates: dict):
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
        img = str(data.get('image', '')).strip()
        
        if not img or img.lower() in ['nan', 'na', 'na_x', 'none', '', 'x']:
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
        data = self.get_current_batch_data()
        my_role = data.get('role', '')
        
        if my_role == PRODUCER:
            raw = data.get('sentences', '[]')
            return json.loads(raw) if raw else []

        partner_id_in_group = int(data.get('partner_id', 0))
        if partner_id_in_group == 0:
            return []

        partner = None
        for p in self.subsession.get_players():
            if p.id_in_subsession == partner_id_in_group:
                partner = p
                break
        
        if not partner:
            return []

        p_data = partner.get_current_batch_data()
        raw = p_data.get('sentences', '[]')
        return json.loads(raw) if raw else []

    def get_full_sentences(self):
        prefix = self.session.vars.get("prefix", "") or ""
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []
        
        sentences = [s for s in sentences if isinstance(s, list)]

        res = []
        for sentence in sentences:
            expansion = []
            if prefix: expansion.append(prefix)
            for val, suffix in zip(sentence, suffixes):
                expansion.append(str(val))
                expansion.append(str(suffix))
            res.append(" ".join(expansion))
        return res

# ----------------------------------------------------------------------------
# SESSION CREATION
# ----------------------------------------------------------------------------
def _truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

def creating_session(subsession: Subsession):
    session = subsession.session
    if subsession.round_number != 1:
        return

    filename = session.config.get("filename")
    if not filename: raise RuntimeError("No filename in session config")

    from reading_xls.get_data import get_data
    excel_payload = get_data(filename)
    
    raw_records = excel_payload['data']
    settings = excel_payload['settings']
    
    # --- FIX S3 PATH ---
    # 1. Try 's3path' (from your excel) OR 's3path_base'
    raw_s3 = settings.get("s3path") or settings.get("s3path_base") or ""
    
    # 2. Fix AWS Console URLs (convert to public link)
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            # Extract bucket name from console URL
            # Format: .../buckets/BUCKET_NAME?region=...
            parts = raw_s3.split("buckets/")[1].split("?")[0]
            raw_s3 = f"https://{parts}.s3.eu-central-1.amazonaws.com"
            logger.info(f"Fixed S3 URL to: {raw_s3}")
        except:
            pass # Use as-is if parsing fails

    session.vars["s3path_base"] = raw_s3
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

    # Build Valid Image Pool
    valid_pool = []
    for r in raw_records:
        img = str(r.get('Item', '')).strip()
        if img.startswith('d-') and 'NA' not in img and 'ABC' not in img:
            valid_pool.append(img)
    if not valid_pool: valid_pool = ["d-A-B-BC-3"]

    # Assign Players
    players = subsession.get_players()
    from collections import defaultdict
    data_by_id = defaultdict(list)
    
    for r in raw_records:
        try: exp_num = int(float(r.get('Exp', 0)))
        except: exp_num = 0
        try: rnd_num = int(float(r.get('group_enumeration') or r.get('Round') or 0))
        except: rnd_num = 0
        
        condition = str(r.get('Condition', '')).strip()
        item_nr = str(r.get('Item.Nr', '')).strip()
        image = str(r.get('Item', '')).strip()

        # REPAIR: If Producer 0 or broken image, swap
        prod_val = str(r.get('Producer', ''))
        is_prod0 = (prod_val == '0' or prod_val == '0.0')
        is_broken = (image == 'NA_x' or image == 'nan' or image == '' or 'D_' in image)
        
        if is_prod0 or is_broken:
            image = random.choice(valid_pool)

        # Parse Sentences
        sentences = []
        for i in range(1, 6):
            p1 = str(r.get(f'Sentence_{i}_1', '')).strip()
            p2 = str(r.get(f'Sentence_{i}_2', '')).strip()
            if (p1 and p1.lower() != 'nan') or (p2 and p2.lower() != 'nan'):
                sentences.append([p1, p2])
        sentences_json = json.dumps(sentences)

        if 'Producer' in r and 'Interpreter' in r:
            try:
                prod_id = int(float(r['Producer'])) + 1
                interp_id = int(float(r['Interpreter'])) + 1
                
                data_by_id[prod_id].append({
                    'sort_key': (exp_num, rnd_num),
                    'batch': exp_num,
                    'partner_id': interp_id,
                    'role': PRODUCER,
                    'condition': condition,
                    'item_nr': item_nr,
                    'image': image,
                    'sentences': sentences_json,
                    'rewards': ''
                })
                
                data_by_id[interp_id].append({
                    'sort_key': (exp_num, rnd_num),
                    'batch': exp_num,
                    'partner_id': prod_id,
                    'role': INTERPRETER,
                    'condition': condition,
                    'item_nr': item_nr,
                    'image': image,
                    'sentences': '[]',
                    'rewards': ''
                })
            except: continue
        elif 'id' in r or 'ID' in r:
            try:
                pid = int(float(r.get('id') or r.get('ID') or 0)) + 1
                partner_pid = int(float(r.get('partner_id', 0))) + 1
                data_by_id[pid].append({
                    'sort_key': (exp_num, rnd_num),
                    'batch': exp_num,
                    'partner_id': partner_pid,
                    'role': str(r.get('role', '')).strip(),
                    'condition': condition,
                    'item_nr': item_nr,
                    'image': image,
                    'sentences': r.get('sentences', '[]'),
                    'rewards': ''
                })
            except: continue

    for p in players:
        my_items = data_by_id.get(p.id_in_subsession, [])
        my_items.sort(key=lambda x: x['sort_key'])
        final_history = []
        for i, item in enumerate(my_items):
            item['round_number'] = i + 1 
            final_history.append(item)
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
        if player.batch_history == "[]" and 'batch_history' in player.participant.vars:
            player.batch_history = player.participant.vars['batch_history']
        data = player.get_current_batch_data()
        if not data:
            if player.round_number == 1: player.faulty = True
            return False
        player.inner_role = data.get('role', '')
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
        updates = {}
        if player.inner_role == PRODUCER:
            updates['sentences'] = player.producer_decision
        elif player.inner_role == INTERPRETER:
            updates['rewards'] = player.interpreter_decision
        player.update_current_batch_data(updates)
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
        cc = (player.session.vars.get("completion_code") or player.session.config.get("completion_code"))
        if not cc: return redirect(Constants.API_ERR_URL)
        return redirect(STUBURL + str(cc))

# ----------------------------------------------------------------------------
# EXPORT
# ----------------------------------------------------------------------------
def custom_export(players):
    yield ["session_code", "participant_code", "round_number", "role", "condition", "item_nr", "image", "producer_sentences", "interpreter_rewards", "decision_seconds", "feedback"]
    processed_participants = set()
    for p in players:
        if p.participant.code in processed_participants: continue
        processed_participants.add(p.participant.code)
        history_json = p.participant.vars.get('batch_history', '[]')
        try: history = json.loads(history_json)
        except: history = []
        timing_map = {}
        feedback_str = ""
        for sub_p in p.participant.get_players():
            timing_map[sub_p.round_number] = sub_p.decision_seconds
            if sub_p.round_number == Constants.num_rounds: feedback_str = sub_p.feedback or ""
        history.sort(key=lambda x: int(x.get('round_number', 0)))
        for item in history:
            rnd = int(item.get('round_number', 0))
            if rnd < 1 or rnd > Constants.num_rounds: continue
            yield [p.session.code, p.participant.code, rnd, item.get('role', ''), item.get('condition', ''), item.get('item_nr', ''), item.get('image', ''), item.get('sentences', ''), item.get('rewards', ''), timing_map.get(rnd, 0), feedback_str if rnd == Constants.num_rounds else ""]

page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
