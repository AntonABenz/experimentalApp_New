from otree.api import *
import json
import logging
import os
import re
import time
import random
from sqlalchemy import create_engine, text
from django.shortcuts import redirect

logger = logging.getLogger("benzapp.img_desc")

PRODUCER = "P"
INTERPRETER = "I"

STUBURL = "https://app.prolific.co/submissions/complete?cc="

class Constants(BaseConstants):
    name_in_url = "img_desc"
    players_per_group = None
    num_rounds = 80

    STUBURL = STUBURL
    PLACEMENT_ERR = "ERROR_BATCH_PLACEMENT"
    API_ERR = "API_ERROR"
    FALLBACK_URL = STUBURL + PLACEMENT_ERR
    API_ERR_URL = STUBURL + API_ERR
    INTERPRETER = INTERPRETER
    PRODUCER = PRODUCER

class Batch(ExtraModel):
    session_code = models.StringField()
    owner_code = models.StringField(blank=True)
    sentences = models.LongStringField()
    rewards = models.LongStringField(blank=True)
    condition = models.StringField()
    item_nr = models.StringField()
    image = models.StringField()
    round_number = models.IntegerField()
    role = models.StringField()
    batch = models.IntegerField()
    id_in_group = models.IntegerField()
    partner_id = models.IntegerField()
    busy = models.BooleanField(initial=False)
    processed = models.BooleanField(initial=False)

_custom_engine = None

def get_engine():
    global _custom_engine
    if _custom_engine:
        return _custom_engine
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        db_url = "sqlite:///db.sqlite3"
    else:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
    _custom_engine = create_engine(db_url)
    return _custom_engine

# --- OPTIMIZED SQL HELPER (Prevents Memory Crashes) ---
def get_batches_filtered(session_code, batch_val=None, round_val=None, owner_code=None):
    """
    Fetches only the specific rows needed, instead of loading the whole database.
    """
    engine = get_engine()
    clauses = ["session_code = :session_code"]
    params = {"session_code": session_code}

    if batch_val is not None:
        clauses.append("batch = :batch")
        params["batch"] = batch_val
    
    if round_val is not None:
        clauses.append("round_number = :rnd")
        params["rnd"] = round_val
        
    if owner_code is not None:
        clauses.append("owner_code = :owner")
        params["owner"] = owner_code

    where_str = " AND ".join(clauses)
    sql = text(f"SELECT * FROM img_desc_batch WHERE {where_str}")

    with engine.connect() as conn:
        result = conn.execute(sql, params)
        return [dict(zip(result.keys(), row)) for row in result]

def get_single_batch_by_id(batch_id):
    engine = get_engine()
    sql = text("SELECT * FROM img_desc_batch WHERE id = :bid")
    with engine.connect() as conn:
        result = conn.execute(sql, {"bid": batch_id}).fetchone()
        if result:
            return dict(zip(result.keys(), result))
    return None

def sql_update_batch(batch_id, **kwargs):
    if not kwargs:
        return
    engine = get_engine()
    set_clauses = []
    params = {"id": batch_id}
    for k, v in kwargs.items():
        set_clauses.append(f"{k} = :{k}")
        params[k] = v
    sql_str = f"UPDATE img_desc_batch SET {', '.join(set_clauses)} WHERE id = :id"
    sql = text(sql_str)
    with engine.connect() as conn:
        conn.execute(sql, params)
        if hasattr(conn, "commit"):
            conn.commit()

def normalize_key(key):
    if not key:
        return ""
    return re.sub(r"[\s_]+", "_", str(key).lower().strip())

class Subsession(BaseSubsession):
    active_batch = models.IntegerField(initial=1)
    study_id = models.StringField()
    completion_code = models.StringField()
    full_return_url = models.StringField()

    @property
    def get_active_batch(self):
        # Optimization: Filter by batch immediately
        return get_batches_filtered(self.session.code, batch_val=self.active_batch)

    def check_for_batch_completion(self):
        session = self.session
        active_batch = self.active_batch or 1
        
        # Optimization: Only load current batch rows
        current_rows = get_batches_filtered(session.code, batch_val=active_batch)
        
        remaining = [
            b for b in current_rows
            if not b["processed"]
        ]
        if remaining:
            return
        session.vars["active_batch"] = active_batch + 1
        Subsession.objects.filter(session=session).update(active_batch=active_batch + 1)

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    inner_role = models.StringField()
    inner_sentences = models.LongStringField()
    batch = models.IntegerField()
    faulty = models.BooleanField(initial=False)
    feedback = models.LongStringField(label="")

    prolific_id = models.StringField()
    prol_study_id = models.StringField()
    prol_session_id = models.StringField()
    completion_code = models.StringField()
    full_return_url = models.StringField(blank=True)

    vars_dump = models.LongStringField()
    producer_decision = models.LongStringField()
    interpreter_decision = models.LongStringField()

    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)

    link_id = models.IntegerField(initial=0)

    def role(self):
        return self.inner_role

    def get_linked_batch(self):
        if not self.link_id:
            return None
        # Optimization: Fetch single row by ID
        return get_single_batch_by_id(self.link_id)

    def get_previous_batch(self):
        if self.inner_role != INTERPRETER:
            return dict(sentences="[]")
        
        l = self.get_linked_batch()
        if not l or l.get("partner_id") in (0, "0", None):
            return dict(sentences="[]")
        
        active_batch = self.subsession.field_maybe_none("active_batch") or 1
        target_batch_idx = active_batch - 1
        if target_batch_idx < 0:
            return dict(sentences="[]")

        # Optimization: Only fetch the previous batch rows (reduce load by 98%)
        candidates = get_batches_filtered(self.session.code, batch_val=target_batch_idx)

        for obj in candidates:
            if (
                obj["role"] == PRODUCER
                and obj["partner_id"] == l["id_in_group"]
                and obj["id_in_group"] == l["partner_id"]
                and obj["condition"] == l["condition"]
            ):
                return obj
        return dict(sentences="[]")

    def get_sentences_data(self):
        l = self.get_linked_batch()
        if not l:
            return []
        
        if self.inner_role == PRODUCER:
            try:
                return json.loads(l.get("sentences") or "[]")
            except Exception:
                return []

        try:
            if l.get("partner_id") in (0, "0", None):
                return json.loads(l.get("sentences") or "[]")
            prev = self.get_previous_batch()
            return json.loads(prev.get("sentences") or "[]")
        except Exception:
            return []

    def update_batch(self):
        if not self.link_id:
            return
        updates = {}
        if self.inner_role == PRODUCER:
            updates["sentences"] = self.producer_decision
        if self.inner_role == INTERPRETER:
            updates["rewards"] = self.interpreter_decision
        if updates:
            sql_update_batch(self.link_id, **updates)

    def mark_data_processed(self):
        self.participant.vars["full_study_completed"] = True
        
        # Optimization: Only load MY batches
        my_batches = get_batches_filtered(self.session.code, owner_code=self.participant.code)
        
        for b in my_batches:
            sql_update_batch(b["id"], processed=True)
        self.subsession.check_for_batch_completion()

    def get_full_sentences(self):
        prefix = self.session.vars.get("prefix", "") or ""
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []
        sentences = [sub for sub in sentences if isinstance(sub, list) and "" not in sub]

        res = []
        for sentence in sentences:
            expansion = [str(item) for pair in zip(sentence, suffixes) for item in pair]
            if prefix:
                expansion.insert(0, prefix)
            res.append(" ".join(expansion))
        return res

    def get_image_url(self):
        l = self.get_linked_batch()
        if not l:
            return ""
        
        image_name = str(l.get("image") or "").strip()
        
        if image_name == "NA_x":
            image_name = "d-A-B-BC-3" 

        if not image_name or image_name.lower() in ["none", "nan", "na", "x"]:
            return ""

        image_name = image_name.replace(" ", "_")

        ext = self.session.vars.get("extension", "png") or "png"
        if not image_name.lower().endswith(f".{ext}"):
            image_name = f"{image_name}.{ext}"

        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        if "amazonaws.com" in base:
            base = base.replace("/practice", "")
            
        return f"{base}/{image_name}"

    def start(self):
        session = self.session
        subsession = self.subsession

        if subsession.field_maybe_none("active_batch") is None:
            Subsession.objects.filter(session=session, active_batch__isnull=True).update(active_batch=1)
            subsession.active_batch = 1

        # Optimization: Fetch ONLY rows for current batch
        active_candidates = get_batches_filtered(session.code, batch_val=subsession.active_batch)
        
        # Filter logic in Python
        candidates = [
            b for b in active_candidates
            if not b["busy"]
            and (b.get("owner_code") or "") == ""
        ]

        if self.round_number == 1:
            if not candidates:
                self.faulty = True
                return

            candidates.sort(key=lambda b: b["id_in_group"])
            free = candidates[0]
            chosen_batch = free["batch"]
            chosen_id = free["id_in_group"]

            self.batch = chosen_batch

            sql_update_batch(
                free["id"],
                busy=True,
                owner_code=self.participant.code
            )

        # Optimization: Fetch ONLY the single row needed for this player/round
        my_rows = get_batches_filtered(
            session.code, 
            round_val=self.round_number, 
            owner_code=self.participant.code
        )

        if not my_rows:
            self.faulty = True
            return

        my_row = my_rows[0]

        self.link_id = my_row["id"]
        self.inner_role = my_row["role"]
        self.inner_sentences = json.dumps(self.get_sentences_data())

        if self.round_number == 1 and session.config.get("for_prolific"):
            p = self.participant
            vars_ = p.vars
            prolific_id = vars_.get("prolific_id") or vars_.get("prolific_pid")
            if vars_.get("prolific_session_id"):
                p.label = vars_.get("prolific_session_id")


def creating_session(subsession: Subsession):
    session = subsession.session

    if subsession.field_maybe_none("active_batch") is None:
        subsession.active_batch = 1

    if subsession.round_number != 1:
        return

    subsession.active_batch = 1
    session.vars["active_batch"] = 1

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Missing filename")

    from reading_xls.get_data import get_data
    excel_data = get_data(filename)
    df = excel_data.get("data")
    session.vars["user_data"] = df

    # LOG THE COLUMNS TO SEE WHAT WE HAVE
    logger.info(f"Excel columns found: {list(df.columns)}")
    if len(df) > 0:
        logger.info(f"First row sample: {df.iloc[0].to_dict()}")

    def clean_str(val):
        if val is None:
            return ""
        s = str(val).strip()
        if s.lower() == "nan":
            return ""
        return s
    
    def safe_int(val, default=0):
        if val is None:
            return default
        s = str(val).strip()
        if s == "" or s.lower() == "nan":
            return default
        try:
            return int(float(s))
        except:
            return default
    
    if "Condition" in df.columns: 
        df["Condition"] = df["Condition"].apply(clean_str)
    if "Item.Nr" in df.columns: 
        df["Item.Nr"] = df["Item.Nr"].apply(clean_str)
    if "Item" in df.columns: 
        df["Item"] = df["Item"].apply(clean_str)

    records = df.to_dict(orient="records")
    
    for i, r in enumerate(records):
        # Try multiple possible column names for round_number
        round_num = safe_int(
            r.get("group_enumeration") or 
            r.get("round") or 
            r.get("Round") or 
            r.get("trial") or
            r.get("Trial")
        )
        
        # Try multiple possible column names for id_in_group
        id_in_grp = safe_int(
            r.get("id") or 
            r.get("ID") or 
            r.get("player_id") or 
            r.get("participant_id") or
            r.get("id_in_group")
        )
        
        batch_dict = {
            'session_code': session.code,
            'owner_code': "",
            'batch': safe_int(r.get("Exp")),
            'item_nr': clean_str(r.get("Item.Nr")),
            'condition': clean_str(r.get("Condition")),
            'image': clean_str(r.get("Item")),
            'round_number': round_num,
            'role': clean_str(r.get("role") or r.get("Role")),
            'id_in_group': id_in_grp,
            'partner_id': safe_int(r.get("partner_id") or r.get("Partner_ID")),
            'sentences': clean_str(r.get("sentences") or "[]"),
        }
        
        # Log first batch for debugging
        if i == 0:
            logger.info(f"First batch being created: {batch_dict}")
        
        Batch.create(**batch_dict)

    logger.info(f"Created {len(records)} batch records")

    # Rest of your settings code stays the same
    settings = excel_data.get("settings") or {}
    clean_settings = {}
    for k, v in settings.items():
        clean_settings[normalize_key(k)] = v
    session.vars["user_settings"] = clean_settings

    for k in ["s3path_base", "extension", "prefix", "interpreter_choices", "interpreter_title"]:
        session.vars[k] = clean_settings.get(normalize_key(k))

    session.vars["suffixes"] = clean_settings.get("suffixes") or []

    allowed_values = []
    allowed_regexes = []
    i = 1
    while True:
        v_key = f"allowed_values_{i}"
        r_key = f"allowed_regex_{i}"
        v_val = clean_settings.get(normalize_key(v_key))
        r_val = clean_settings.get(normalize_key(r_key))
        if v_val or r_val:
            if v_val:
                allowed_values.append([x.strip() for x in str(v_val).split(";") if x.strip()])
            else:
                allowed_values.append([])
            allowed_regexes.append(str(r_val).strip() if r_val else "")
            i += 1
            continue
        if i > 10:
            break
        i += 1

    session.vars["allowed_values"] = allowed_values
    session.vars["allowed_regexes"] = allowed_regexes

    default_url = "https://docs.google.com/document/d/e/2PACX-1vTg_Hd8hXK-TZS77rC6W_BlY2NtWhQqCLzlgW0LeomoEUdhoDNYPNVOO7Pt6g0-JksykUrgRdtcVL3u/pub?embedded=true"
    url_from_settings = clean_settings.get(normalize_key("instructions_url"))
    session.vars["instructions_url"] = url_from_settings if url_from_settings else default_url

class FaultyCatcher(Page):
    @staticmethod
    def is_displayed(player):
        return player.faulty

    def get(self):
        if self.player.faulty:
            return redirect(Constants.FALLBACK_URL)
        return super().get()

class Q(Page):
    instructions = True
    form_model = "player"

    @staticmethod
    def is_displayed(player):
        if player.faulty:
            return False
        
        if player.start_decision_time == 0:
            player.start_decision_time = time.time()
        
        player.start()
        if player.faulty:
            return False
        return player.round_number <= Constants.num_rounds

    @staticmethod
    def get_form_fields(player):
        role = player.field_maybe_none("inner_role")
        if role == PRODUCER:
            return ["producer_decision"]
        elif role == INTERPRETER:
            return ["interpreter_decision"]
        return []

    @staticmethod
    def vars_for_template(player):
        raw_choices = player.session.vars.get("interpreter_choices") or ""
        if isinstance(raw_choices, str):
            interpreter_choices = [x.strip() for x in raw_choices.split(";") if x.strip()]
        elif isinstance(raw_choices, list):
            interpreter_choices = raw_choices
        else:
            interpreter_choices = []
    
        interpreter_title = player.session.vars.get("interpreter_title") or "Buy medals:"
    
        return dict(
            d=player.get_linked_batch(),
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regexes=player.session.vars.get("allowed_regexes", []),
            suffixes=player.session.vars.get("suffixes", []),
            prefix=player.session.vars.get("prefix", ""),
            interpreter_choices=interpreter_choices,
            interpreter_title=interpreter_title,
            instructions_url=player.session.vars.get("instructions_url"),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.end_decision_time = time.time()
        if player.start_decision_time > 0:
            player.decision_seconds = player.end_decision_time - player.start_decision_time
        
        player.update_batch()
        if player.round_number == Constants.num_rounds:
            player.mark_data_processed()


class Feedback(Page):
    form_model = "player"
    form_fields = ["feedback"]

    def is_displayed(self):
        return self.round_number == Constants.num_rounds


class FinalForProlific(Page):
    @staticmethod
    def is_displayed(player):
        return player.session.config.get("for_prolific") and (player.round_number == Constants.num_rounds)

    def get(self):
        url = self.player.field_maybe_none("full_return_url")
        if url:
            return redirect(url)
        return redirect("https://cnn.com")

def custom_export(players):
    """
    Optimized Export using Django ORM to prevent memory spikes and SQL errors.
    Works on both SQLite (Local) and Postgres (Heroku).
    """
    # 1. YIELD HEADER IMMEDIATELY
    yield [
        "session", "participant_code", "round", "role", "condition", 
        "item_nr", "image", "producer_sentences", "interpreter_rewards", 
        "decision_seconds"
    ]

    # 2. Get unique session codes efficiently
    if hasattr(players, 'values_list'):
        session_codes = list(players.values_list('session__code', flat=True).distinct())
    else:
        session_codes = list(set(p.session.code for p in players))

    if not session_codes:
        return

    # 3. Build timing map efficiently
    timing_map = {}
    # Fetch only necessary fields to save memory
    timing_data = Player.objects.filter(
        session__code__in=session_codes
    ).values_list('session__code', 'participant__code', 'round_number', 'decision_seconds').iterator()

    for s_code, p_code, r_num, seconds in timing_data:
        timing_map[(s_code, p_code, r_num)] = seconds

    # 4. Stream Batch data using ORM iterator (Database Agnostic)
    columns = [
        'session_code', 'owner_code', 'round_number', 'role', 
        'condition', 'item_nr', 'image', 'sentences', 'rewards'
    ]
    
    batches = Batch.objects.filter(
        session_code__in=session_codes
    ).order_by('session_code', 'batch', 'id_in_group').values_list(*columns).iterator(chunk_size=2000)

    for row in batches:
        s_code, o_code, r_num, role, cond, item, img, sent, rew = row
        
        ds = ""
        if o_code:
            ds = timing_map.get((s_code, o_code, r_num), "")

        yield [
            s_code, o_code, r_num, role, cond, item, img, sent, rew, ds
        ]


page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
