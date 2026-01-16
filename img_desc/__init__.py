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

# We removed the ChainData model to simplify.
# We will rely on Pre-Loaded Excel Data (Bot) for now to ensure stability.

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
        if not img or img.lower() in ['nan', 'na', 'na_x', 'none', '', 'x']: return ""
        
        base = self.session.vars.get('s3path_base', '').rstrip('/')
        ext = self.session.vars.get('extension', 'png')
        if "amazonaws" in base: base = base.replace("/practice", "")
        clean_name = img.replace(" ", "_")
        if not clean_name.lower().endswith(f".{ext}"): clean_name = f"{clean_name}.{ext}"
        return f"{base}/{clean_name}"

    def get_sentences_data(self):
        """
        Retrieves sentences for the Interpreter.
        It uses the 'bot_fallback' field which contains the Excel sentences.
        """
        data = self.get_current_batch_data()
        my_role = data.get('role', '')

        # Producers see nothing
        if my_role == PRODUCER:
            raw = data.get('sentences', '[]')
            # If they previously entered something, show it (e.g. back button)
            return json.loads(raw) if raw and raw != '[]' else []

        # INTERPRETERS
        # 1. Check if we already have final sentences (e.g. from human partner)
        raw_final = data.get('sentences', '[]')
        if raw_final and raw_final != '[]':
            return json.loads(raw_final)

        # 2. Use Bot Fallback (Excel Data)
        # This guarantees sentences always appear
        bot_sentences = data.get('bot_fallback', '[]')
        if bot_sentences and bot_sentences != '[]':
             # Auto-fill 'sentences' with bot data so it's recorded
             self.update_current_batch_data({'sentences': bot_sentences})
             return json.loads(bot_sentences)

        return []

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
    if subsession.round_number != 1: return

    filename = session.config.get("filename")
    if not filename: raise RuntimeError("No filename in session config")

    from reading_xls.get_data import get_data
    excel_payload = get_data(filename)
    raw_records = excel_payload['data']
    settings = excel_payload['settings']
    
    # Settings Setup
    raw_s3 = settings.get("s3path") or settings.get("s3path_base") or ""
    if "console.aws.amazon.com" in raw_s3:
        try:
            parts = raw_s3.split("buckets/")[1].split("?")[0]
            raw_s3 = f"https://{parts}.s3.eu-central-1.amazonaws.com"
        except: pass

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

    # 1. Build Image Pool
    valid_pool = []
    for r in raw_records:
        img = str(r.get('Item', '')).strip()
        if img.startswith('d-') and 'NA' not in img and 'ABC' not in img:
            valid_pool.append(img)
    if not valid_pool: valid_pool = ["d-A-B-BC-3"]

    # 2. Build Data
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

        # REPAIR LOGIC
        prod_val = str(r.get('Producer', ''))
        is_bot = (prod_val == '0' or prod_val == '0.0')
        is_broken = (image == 'NA_x' or image == 'nan' or image == '' or 'D_' in image)
        if is_bot or is_broken:
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
                prod_id = int(float(r['Producer']))
                interp_id = int(float(r['Interpreter']))
                
                if prod_id > 0:
                    data_by_id[prod_id].append({
                        'sort_key': (exp_num, rnd_num),
                        'batch': exp_num,
                        'role': PRODUCER,
                        'condition': condition,
                        'item_nr': item_nr,
                        'image': image,
                        'sentences': sentences_json, 
                        'bot_fallback': sentences_json, 
                        'partner_id': interp_id,
                        'rewards': ''
                    })
                
                if interp_id > 0:
                    # ALWAYS copy bot sentences to fallback so we have them
                    fallback = sentences_json
                    
                    data_by_id[interp_id].append({
                        'sort_key': (exp_num, rnd_num),
                        'batch': exp_num,
                        'role': INTERPRETER,
                        'condition': condition,
                        'item_nr': item_nr,
                        'image': image,
                        'sentences': '[]', 
                        'bot_fallback': fallback, 
                        'partner_id': prod_id,
                        'rewards': ''
                    })
            except: continue

    # 3. Assign + Force Structure (3P / 5I)
    for p in players:
        my_items = data_by_id.get(p.id_in_subsession, [])
        producers = [x for x in my_items if x['role'] == PRODUCER]
        interpreters = [x for x in my_items if x['role'] == INTERPRETER]
        
        producers.sort(key=lambda x: x['sort_key'])
        interpreters.sort(key=lambda x: x['sort_key'])
        
        # SAFEGUARD: If 0 producers, force 3 I->P
        if len(producers) == 0 and len(interpreters) >= 3:
            for _ in range(3):
                item = interpreters.pop(0) 
                item['role'] = PRODUCER
                item['producer_sentences'] = ""
                producers.append(item)
        
        # Build 3P / 5I
        final_history = []
        p_idx = 0
        i_idx = 0
        current_round = 1
        
        while current_round <= Constants.num_rounds:
            for _ in range(3):
                if current_round > Constants.num_rounds: break
                if p_idx < len(producers):
                    item = producers[p_idx].copy()
                    item.pop('sort_key', None)
                    item['round_number'] = current_round
                    final_history.append(item)
                    p_idx += 1
                    current_round += 1
                else: break 
            
            for _ in range(5):
                if current_round > Constants.num_rounds: break
                if i_idx < len(interpreters):
                    item = interpreters[i_idx].copy()
                    item.pop('sort_key', None)
                    item['round_number'] = current_round
                    final_history.append(item)
                    i_idx += 1
                    current_round += 1
                else: break 
            
            if p_idx >= len(producers) and i_idx >= len(interpreters): break
            
        p.participant.vars['batch_history'] = json.dumps(final_history)

# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
class FaultyCatcher(Page):
    @staticmethod
    def is_displayed(player): return player.faulty
    def get(self): return redirect(Constants.FALLBACK_URL)

class Q(Page):
    form_model = "player"

    @staticmethod
    def is_displayed(player):
        if player.round_number > Constants.num_rounds: return False
        if player.batch_history == "[]" and 'batch_history' in player.participant.vars:
            player.batch_history = player.participant.vars['batch_history']
        data = player.get_current_batch_data()
        if not data: return False
        player.inner_role = data.get('role', '')
        if player.start_decision_time == 0: player.start_decision_time = time.time()
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
        try:
            hist = json.loads(player.batch_history)
            return player.round_number == len(hist)
        except: return False

class FinalForProlific(Page):
    @staticmethod
    def is_displayed(player):
        try:
            hist = json.loads(player.batch_history)
            return player.session.config.get("for_prolific") and player.round_number == len(hist)
        except: return False
    def get(self):
        cc = (player.session.vars.get("completion_code") or player.session.config.get("completion_code"))
        if not cc: return redirect(Constants.API_ERR_URL)
        return redirect(STUBURL + str(cc))

def custom_export(players):
    yield ["session", "participant", "round", "role", "condition", "item_nr", "image", "sentences", "rewards", "seconds"]
    for p in players:
        history = json.loads(p.participant.vars.get('batch_history', '[]'))
        history.sort(key=lambda x: int(x.get('round_number', 0)))
        for item in history:
            rnd = int(item.get('round_number', 0))
            if rnd < 1 or rnd > Constants.num_rounds: continue
            yield [p.session.code, p.participant.code, rnd, item.get('role'), item.get('condition'), item.get('item_nr'), item.get('image'), item.get('sentences'), item.get('rewards'), 0]

page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
