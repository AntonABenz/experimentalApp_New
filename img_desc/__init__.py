from otree.api import *
import json
import logging
import time
import re
import random
from starlette.responses import RedirectResponse 

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

    def _history(self):
        try:
            return json.loads(self.batch_history or "[]")
        except Exception:
            return []

    def get_current_batch_data(self):
        rnd = int(self.round_number or 0)
        for item in self._history():
            if int(item.get("round_number", 0)) == rnd:
                return item
        return {}

    def update_current_batch_data(self, updates: dict):
        try:
            hist = self._history()
            rnd = int(self.round_number or 0)
            changed = False
            for item in hist:
                if int(item.get("round_number", 0)) == rnd:
                    item.update(updates)
                    changed = True
                    break
            if changed:
                self.batch_history = json.dumps(hist)
        except Exception:
            pass

    def get_image_url(self):
        data = self.get_current_batch_data()
        img = clean_str(data.get("image", ""))
        if not img or img.lower() in {"nan", "na", "na_x", "none", "", "x"}:
            return ""
        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        ext = self.session.vars.get("extension") or "png"
        if "amazonaws" in base:
            base = base.replace("/practice", "")
        clean_name = img.replace(" ", "_")
        if not clean_name.lower().endswith(f".{ext}"):
            clean_name = f"{clean_name}.{ext}"
        return f"{base}/{clean_name}"

    def get_sentences_data(self):
        data = self.get_current_batch_data()
        if not data: return []
        role = data.get("role", "")
        if role == PRODUCER: return []
        raw = data.get("producer_sentences") or data.get("sentences") or "[]"
        try: return json.loads(raw) if raw else []
        except Exception: return []

    def get_full_sentences(self):
        try:
            prefix = self.session.vars.get("prefix") or ""
            suffixes = self.session.vars.get("suffixes") or []
            sentences = self.get_sentences_data() or []
            sentences = [s for s in sentences if isinstance(s, list)]
            res = []
            for sentence in sentences:
                parts = []
                if prefix: parts.append(str(prefix))
                for val, suf in zip(sentence, suffixes):
                    val_str = str(val).strip() if val else ""
                    if not val_str or val_str.lower() in ['nan', 'none']:
                        val_str = "None"
                    parts.append(val_str)
                    if suf: parts.append(str(suf))
                if len(sentence) > len(suffixes):
                    for extra in sentence[len(suffixes):]:
                        extra_str = str(extra).strip() if extra else ""
                        if not extra_str or extra_str.lower() in ['nan', 'none']:
                            extra_str = "None"
                        parts.append(extra_str)
                res.append(" ".join([p for p in parts if p is not None and str(p).strip() != ""]))
            return res
        except Exception as e:
            logger.error(f"Error in get_full_sentences: {e}")
            return []

    def save_to_chain(self):
        pass

# ----------------------------------------------------------------------------
# UTIL
# ----------------------------------------------------------------------------
def safe_int(x, default=0) -> int:
    try: return int(float(x))
    except Exception: return default

def clean_str(x) -> str:
    if x is None: return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none"}: return ""
    return s

def normalize_key(key):
    if not key: return ""
    return re.sub(r"[\s_]+", "_", str(key).lower().strip())

def _truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}

def is_valid_real_image(img: str) -> bool:
    img = clean_str(img)
    if not img: return False
    low = img.lower()
    if low in {"na_x", "na", "nan", "none", "x"}: return False
    if img.startswith("D_") or (img.startswith("d_") and not img.startswith("d-")): return False
    if img.startswith("d-") or img.startswith("D-"): return True
    return False

def extract_sentences_from_row(r: dict) -> str:
    pairs = []
    for i in range(1, 6):
        a = clean_str(r.get(f"Sentence_{i}_1"))
        b = clean_str(r.get(f"Sentence_{i}_2"))
        if a or b: pairs.append([a, b])
    return json.dumps(pairs)

def fix_s3_url(raw_s3: str) -> str:
    raw_s3 = clean_str(raw_s3)
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            bucket = raw_s3.split("buckets/")[1].split("?")[0].strip("/")
            return f"https://{bucket}.s3.eu-central-1.amazonaws.com"
        except Exception: return raw_s3
    return raw_s3

# ----------------------------------------------------------------------------
# SESSION CREATION
# ----------------------------------------------------------------------------
def creating_session(subsession: Subsession):
    session = subsession.session
    if subsession.round_number != 1: return

    try:
        filename = session.config.get("filename")
        if not filename: raise RuntimeError("No filename in session config")

        logger.info(f"Starting session creation with filename: {filename}")

        # Dynamic import
        try:
            from reading_xls.get_data import get_data
        except ImportError:
            import sys
            from pathlib import Path
            reading_xls_path = Path(__file__).parent.parent / "reading_xls"
            sys.path.insert(0, str(reading_xls_path))
            from get_data import get_data
        
        excel_payload = get_data(filename)
        raw_data = excel_payload.get("data")
        settings = excel_payload.get("settings") or {}

        if hasattr(raw_data, "to_dict"): rows = raw_data.to_dict(orient="records")
        else: rows = list(raw_data or [])

        # --- 1. Process Global Settings ---
        clean_settings = {}
        for k, v in settings.items():
            if isinstance(v, str): # Only process string settings here
                clean_settings[normalize_key(k)] = clean_str(v)
        
        s3_raw = clean_settings.get("s3path") or clean_settings.get("s3path_base") or ""
        session.vars["s3path_base"] = fix_s3_url(s3_raw)
        session.vars["extension"] = clean_settings.get("extension") or "png"
        session.vars["prefix"] = clean_settings.get("prefix") or ""
        session.vars["interpreter_title"] = clean_settings.get("interpreter_title") or "Buy medals:"
        session.vars["caseflag"] = _truthy(clean_settings.get("caseflag"))
        session.vars["instructions_url"] = clean_settings.get("instructions_url") or "https://google.com"

        # --- 2. Process Practice Configurations ---
        # The 'start' app needs these in session.vars to render practice pages
        for k, v in settings.items():
            if k.startswith("Practice") and isinstance(v, dict):
                # We found a practice sheet config!
                practice_data = v.copy()
                
                # Pre-calculate the full image URL for the template
                img_name = practice_data.get('image', '')
                if img_name:
                    img_name = img_name.strip()
                    # Ensure extension
                    ext = session.vars["extension"]
                    if not img_name.lower().endswith(f".{ext}"):
                        img_name = f"{img_name}.{ext}"
                    # Prepend S3 Base
                    base = session.vars["s3path_base"].rstrip("/")
                    if "amazonaws" in base:
                        base = base.replace("/practice", "") # standard cleanup
                    
                    practice_data['image'] = f"{base}/{img_name}"
                
                # Store in session.vars (e.g. session.vars['Practice1'])
                session.vars[k] = practice_data
                logger.info(f"Loaded {k} into session.vars with image {practice_data.get('image')}")

        # --- 3. Lists and Other Configs ---
        suffixes = []
        for i in range(1, 11):
            v = clean_settings.get(f"suffix_{i}")
            if v: suffixes.append(v)
        session.vars["suffixes"] = suffixes

        ic = settings.get("interpreter_choices")
        if isinstance(ic, str): session.vars["interpreter_choices"] = [x.strip() for x in ic.split(";") if x.strip()]
        elif isinstance(ic, list): session.vars["interpreter_choices"] = ic
        else: session.vars["interpreter_choices"] = []

        session.vars["allowed_values"] = settings.get("allowed_values", []) or []
        session.vars["allowed_regexes"] = settings.get("allowed_regex", []) or []

        if session.config.get("completion_code"):
            session.vars["completion_code"] = str(session.config["completion_code"])

        # --- 4. Determine Slots and Schedule ---
        slot_ids = set()
        for r in rows:
            p = safe_int(r.get("Producer"), 0)
            i = safe_int(r.get("Interpreter"), 0)
            if p != 0 and p != 9: slot_ids.add(p)
            if i != 0: slot_ids.add(i)

        if not slot_ids: raise RuntimeError("No IDs found in Excel.")
        K = max(slot_ids)
        players = subsession.get_players()
        slot_to_pid = {}
        pid_to_slot = {}
        for idx, pl in enumerate(players, start=1):
            if idx <= K:
                slot_to_pid[idx] = pl.id_in_subsession
                pid_to_slot[pl.id_in_subsession] = idx

        # Valid Image Pool
        valid_pool = []
        for r in rows:
            producer_slot = safe_int(r.get("Producer"), 0)
            img = clean_str(r.get("Item"))
            if producer_slot != 0 and producer_slot != 9 and is_valid_real_image(img):
                valid_pool.append(img)
        if not valid_pool: valid_pool = ["d-A-B-BC-3"]

        # Build Schedule
        from collections import defaultdict
        data_by_pid = defaultdict(list)

        for idx, r in enumerate(rows):
            exp_num = 0
            for key in ["d111", "Exp", "Experiment", "Exp.", "experiment"]:
                val = r.get(key)
                if val:
                    exp_num = safe_int(val, 0)
                    break
            
            round_in_excel = safe_int(r.get("Round"), 0)
            trial = safe_int(r.get("Trial"), 0)
            condition = clean_str(r.get("Condition"))
            item_nr = clean_str(r.get("Item.Nr"))
            image_raw = clean_str(r.get("Item"))
            producer_slot = safe_int(r.get("Producer"), 0)
            interpreter_slot = safe_int(r.get("Interpreter"), 0)

            if interpreter_slot == 0: continue
            sentences_json = extract_sentences_from_row(r)
            interp_pid = slot_to_pid.get(interpreter_slot)
            prod_pid = slot_to_pid.get(producer_slot) if (producer_slot != 0 and producer_slot != 9) else None
            if not interp_pid: continue

            sort_key = (exp_num, round_in_excel, trial, idx)

            if producer_slot == 9:
                picked_image = random.choice(valid_pool)
                data_by_pid[interp_pid].append({
                    "sort_key": sort_key, "role": INTERPRETER, "partner_id": 0,
                    "exp": exp_num, "round_in_excel": round_in_excel, "trial": trial,
                    "condition": condition, "item_nr": item_nr, "image": picked_image,
                    "producer_sentences": sentences_json, "interpreter_rewards": "",
                })
            elif producer_slot == 0:
                picked = random.choice(valid_pool)
                data_by_pid[interp_pid].append({
                    "sort_key": sort_key, "role": INTERPRETER, "partner_id": 0,
                    "exp": exp_num, "round_in_excel": round_in_excel, "trial": trial,
                    "condition": condition, "item_nr": item_nr, "image": picked,
                    "producer_sentences": sentences_json, "interpreter_rewards": "",
                })
            elif prod_pid:
                prod_image = image_raw if is_valid_real_image(image_raw) else random.choice(valid_pool)
                data_by_pid[prod_pid].append({
                    "sort_key": sort_key, "role": PRODUCER, "partner_id": interp_pid,
                    "exp": exp_num, "round_in_excel": round_in_excel, "trial": trial,
                    "condition": condition, "item_nr": item_nr, "image": prod_image,
                    "producer_sentences": "", "interpreter_rewards": "",
                })
                data_by_pid[interp_pid].append({
                    "sort_key": sort_key, "role": INTERPRETER, "partner_id": prod_pid,
                    "exp": exp_num, "round_in_excel": round_in_excel, "trial": trial,
                    "condition": condition, "item_nr": item_nr, "image": prod_image,
                    "producer_sentences": sentences_json, "interpreter_rewards": "",
                })

        empty = []
        for p in players:
            my_items = data_by_pid.get(p.id_in_subsession, [])
            my_items.sort(key=lambda x: x["sort_key"])

            producer_items = [it for it in my_items if it.get("role") == PRODUCER]
            interpreter_items = [it for it in my_items if it.get("role") == INTERPRETER]

            if len(producer_items) == 0 and len(interpreter_items) >= 3:
                for _ in range(3):
                    item = interpreter_items.pop(0) 
                    item["role"] = PRODUCER
                    item["producer_sentences"] = ""
                    producer_items.append(item)

            final_history = []
            round_counter = 1
            p_idx = 0
            i_idx = 0

            while round_counter <= Constants.num_rounds:
                for _ in range(3):
                    if round_counter > Constants.num_rounds: break
                    if p_idx < len(producer_items):
                        item = producer_items[p_idx].copy()
                        item.pop("sort_key", None)
                        item["round_number"] = round_counter
                        final_history.append(item)
                        p_idx += 1
                        round_counter += 1
                    else: break
                
                for _ in range(5):
                    if round_counter > Constants.num_rounds: break
                    if i_idx < len(interpreter_items):
                        item = interpreter_items[i_idx].copy()
                        item.pop("sort_key", None)
                        item["round_number"] = round_counter
                        final_history.append(item)
                        i_idx += 1
                        round_counter += 1
                    else: break
                
                if p_idx >= len(producer_items) and i_idx >= len(interpreter_items): break

            p.batch_history = json.dumps(final_history)
            p.participant.vars["batch_history"] = p.batch_history
            if not final_history: empty.append(p.id_in_subsession)

        if empty: logger.warning(f"EMPTY schedules for: {empty}")

    except Exception as e:
        logger.error(f"ERROR in creating_session: {e}", exc_info=True)
        raise

# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
class FaultyCatcher(Page):
    @staticmethod
    def is_displayed(player): return player.faulty
    def get(self): 
        return RedirectResponse(Constants.FALLBACK_URL, status_code=302)

class Q(Page):
    form_model = "player"
    @staticmethod
    def is_displayed(player):
        if player.round_number > Constants.num_rounds: return False
        if (player.batch_history == "[]" or not player.batch_history) and "batch_history" in player.participant.vars:
            player.batch_history = player.participant.vars["batch_history"]
        data = player.get_current_batch_data()
        if not data:
            if player.round_number == 1: player.faulty = True
            return False
        player.inner_role = data.get("role", "")
        if player.start_decision_time == 0: player.start_decision_time = time.time()
        return True

    @staticmethod
    def get_form_fields(player):
        if player.inner_role == PRODUCER: return ["producer_decision"]
        if player.inner_role == INTERPRETER: return ["interpreter_decision"]
        return []

    @staticmethod
    def vars_for_template(player):
        raw_choices = player.session.vars.get("interpreter_choices") or ""
        if isinstance(raw_choices, str): interpreter_choices = [x.strip() for x in raw_choices.split(";") if x.strip()]
        elif isinstance(raw_choices, list): interpreter_choices = raw_choices
        else: interpreter_choices = []
        return dict(
            d=player.get_current_batch_data(),
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regexes=player.session.vars.get("allowed_regexes", []),
            suffixes=player.session.vars.get("suffixes", []),
            prefix=player.session.vars.get("prefix", ""),
            interpreter_choices=interpreter_choices,
            interpreter_title=player.session.vars.get("interpreter_title") or "Buy medals:",
            instructions_url=player.session.vars.get("instructions_url"),
            server_image_url=player.get_image_url(),
            caseflag=player.session.vars.get("caseflag"),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.end_decision_time = time.time()
        if player.start_decision_time: player.decision_seconds = player.end_decision_time - player.start_decision_time
        updates = {}
        if player.inner_role == PRODUCER: updates["producer_sentences"] = player.producer_decision
        elif player.inner_role == INTERPRETER: updates["interpreter_rewards"] = player.interpreter_decision
        if updates:
            player.update_current_batch_data(updates)
            player.participant.vars["batch_history"] = player.batch_history

class Feedback(Page):
    form_model = "player"
    form_fields = ["feedback"]
    @staticmethod
    def is_displayed(player): return player.round_number == Constants.num_rounds

class FinalForProlific(Page):
    @staticmethod
    def is_displayed(player): return player.session.config.get("for_prolific") and player.round_number == Constants.num_rounds
    def get(self):
        cc = self.player.session.vars.get("completion_code") or self.player.session.config.get("completion_code")
        if not cc: return RedirectResponse(Constants.API_ERR_URL, status_code=302)
        return RedirectResponse(STUBURL + str(cc), status_code=302)

# ----------------------------------------------------------------------------
# EXPORT
# ----------------------------------------------------------------------------
def custom_export(players):
    """
    Export all 80 rounds with extended columns:
    demographics, exp_num, producer_id, interpreter_id.
    """
    yield [
        "session", "participant", "prolific_id", "demographics", 
        "exp_num", "round", "role", "producer_id", "interpreter_id", 
        "condition", "item_nr", "image", "sentences_formatted", "sentences_raw", 
        "rewards", "seconds", "feedback"
    ]
    
    from collections import defaultdict
    players_by_participant = defaultdict(list)
    for p in players:
        players_by_participant[p.participant.code].append(p)
    
    for participant_code, participant_players in players_by_participant.items():
        try:
            first_player = participant_players[0]
            
            # Metadata
            prolific_id = first_player.participant.vars.get('prolific_id', '')
            excel_slot = first_player.id_in_subsession
            
            # Demographics
            demographics = ""
            try:
                if 'demographics' in first_player.participant.vars:
                    demographics = json.dumps(first_player.participant.vars['demographics'])
                else:
                    start_players = [p for p in first_player.participant.get_players() if hasattr(p, 'survey_data')]
                    if start_players: demographics = start_players[0].survey_data or ""
            except: pass

            # History
            history_json = first_player.participant.vars.get('batch_history', '[]')
            history = json.loads(history_json)
            
            # Timing
            timing_map = {}
            feedback_str = ""
            for p in participant_players:
                if p.round_number:
                    timing_map[p.round_number] = p.decision_seconds or 0
                    if p.round_number == Constants.num_rounds and p.feedback:
                        feedback_str = p.feedback
            
            # Sort
            history.sort(key=lambda x: int(x.get('round_number', 0)))
            
            for item in history:
                rnd = int(item.get('round_number', 0))
                if rnd < 1 or rnd > Constants.num_rounds: continue
                
                # Format Sentences
                raw_sentences = item.get('producer_sentences') or item.get('sentences') or ""
                formatted_sentences = raw_sentences
                try:
                    if isinstance(raw_sentences, str) and raw_sentences.startswith('['):
                        data = json.loads(raw_sentences)
                        parts = []
                        for pair in data:
                            if isinstance(pair, list) and len(pair) >= 2:
                                parts.append(f"{pair[0]} {pair[1]}")
                        if parts:
                            formatted_sentences = "; ".join(parts)
                except: pass

                # IDs
                my_role = item.get('role', '')
                partner_id = item.get('partner_id', 0)
                if my_role == PRODUCER:
                    prod_id = excel_slot
                    interp_id = partner_id
                else:
                    prod_id = partner_id
                    interp_id = excel_slot

                # Fix: Look for "d111" OR "Exp"
                exp_num = item.get('exp', '') or item.get('d111', '')

                yield [
                    first_player.session.code, participant_code, prolific_id, demographics,
                    exp_num, rnd, my_role, prod_id, interp_id,
                    item.get('condition', ''), item.get('item_nr', ''), item.get('image', ''), 
                    formatted_sentences, raw_sentences,
                    item.get('interpreter_rewards', '') or item.get('rewards', ''), 
                    timing_map.get(rnd, 0), feedback_str if rnd == Constants.num_rounds else ""
                ]
        except Exception: continue

page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
