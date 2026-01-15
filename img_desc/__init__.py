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
        result = conn.execute(sql, {"bid": batch_id})
        row = result.fetchone()
        if not row:
            return None
        return dict(zip(result.keys(), row))


def sql_update_batch(batch_id, **kwargs):
    if not kwargs:
        return
    engine = get_engine()
    set_clauses = []
    params = {"id": batch_id}
    for k, v in kwargs.items():
        set_clauses.append(f"{k} = :{k}")
        params[k] = v

    sql = text(f"UPDATE img_desc_batch SET {', '.join(set_clauses)} WHERE id = :id")
    with engine.begin() as conn:     # ✅ commits on exit
        conn.execute(sql, params)

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
    
        current_rows = get_batches_filtered(session.code, batch_val=active_batch)
        remaining = [b for b in current_rows if not b["processed"]]
    
        if remaining:
            return
    
        # bump in memory only
        self.active_batch = active_batch + 1
        session.vars["active_batch"] = active_batch + 1


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
    
        # ensure default
        if subsession.field_maybe_none("active_batch") is None:
            subsession.active_batch = 1
    
        # round 1: allocate participant
        if self.round_number == 1:
            active_batch = subsession.active_batch or 1
    
            active_rows = get_batches_filtered(session.code, batch_val=active_batch)
    
            candidates = [
                b for b in active_rows
                if not b["busy"] and (b.get("owner_code") or "") == ""
            ]
            if not candidates:
                self.faulty = True
                return
    
            candidates.sort(key=lambda b: b["id_in_group"])
            chosen = candidates[0]
            chosen_id = chosen["id_in_group"]
            chosen_batch = chosen["batch"]
    
            self.batch = chosen_batch
    
            # IMPORTANT: mark ALL rows for this id_in_group as owned/busy
            for b in active_rows:
                if b["id_in_group"] == chosen_id:
                    sql_update_batch(b["id"], busy=True, owner_code=self.participant.code)
    
        # now fetch exactly this participant+round row
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

def delete_batches_for_session(session_code: str):
    engine = get_engine()
    sql = text("DELETE FROM img_desc_batch WHERE session_code = :sc")
    with engine.begin() as conn:     # ✅ commits on exit
        conn.execute(sql, {"sc": session_code})



def _to_int(v, default=0):
    try:
        if v is None or str(v).strip() == "":
            return default
        return int(float(v))
    except Exception:
        return default

def _to_str(v):
    return "" if v is None else str(v).strip()

def creating_session(subsession: Subsession):
    session = subsession.session
    if subsession.round_number != 1:
        return

    # ✅ wipe stale rows
    delete_batches_for_session(session.code)

    subsession.active_batch = 1
    session.vars["active_batch"] = 1

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Missing filename")

    from reading_xls.get_data import get_data
    excel_data = get_data(filename)
    df = excel_data.get("data")
    settings = excel_data.get("settings") or {}

    # ✅ load settings into session.vars (you were missing this)
    clean_settings = {normalize_key(k): v for k, v in settings.items()}
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
            allowed_values.append([x.strip() for x in str(v_val).split(";") if x.strip()] if v_val else [])
            allowed_regexes.append(str(r_val).strip() if r_val else "")
        else:
            if i > 10:
                break
        i += 1

    session.vars["allowed_values"] = allowed_values
    session.vars["allowed_regexes"] = allowed_regexes

    default_url = "https://docs.google.com/document/d/e/2PACX-1vTg_Hd8hXK-TZS77rC6W_BlY2NtWhQqCLzlgW0LeomoEUdhoDNYPNVOO7Pt6g0-JksykUrgRdtcVL3u/pub?embedded=true"
    session.vars["instructions_url"] = clean_settings.get(normalize_key("instructions_url")) or default_url

    # ✅ insert rows
    records = df.to_dict(orient="records")
    for r in records:
        Batch.create(
            session_code=session.code,
            owner_code="",
            batch=_to_int(r.get("Exp"), 0),
            item_nr=_to_str(r.get("Item.Nr")),
            condition=_to_str(r.get("Condition")),
            image=_to_str(r.get("Item")),
            round_number=_to_int(r.get("group_enumeration"), 0),
            role=_to_str(r.get("role")),
            id_in_group=_to_int(r.get("id"), 0),
            partner_id=_to_int(r.get("partner_id"), 0),
            sentences=_to_str(r.get("sentences")) or "[]",
        )

    # ✅ logs INSIDE function
    try:
        items = set(_to_str(x) for x in df["Item"].tolist()) if "Item" in df.columns else set()
        logger.info(f"Loaded {len(df)} rows from Excel. Unique Item count={len(items)}")
        logger.info(f"Old image present? {'d-A-B-AB-BC-ABC-3' in items}")
    except Exception:
        logger.info(f"Loaded {len(df)} rows from Excel.")

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
        if player.round_number > Constants.num_rounds:
            return False
        if player.faulty:
            return False
    
        # run start() once per round (guard by link_id)
        if not player.link_id:
            player.start()
    
        if player.faulty:
            return False
    
        # start timing only after start() succeeded
        if player.start_decision_time == 0:
            player.start_decision_time = time.time()
    
        return True


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
    yield [
        "session_code", "participant_code", "round_number", "role", "condition",
        "item_nr", "image", "producer_sentences", "interpreter_rewards", "decision_seconds"
    ]

    # If players is a queryset, stream minimal fields
    if hasattr(players, "values_list"):
        pq = players.values_list("session__code", "participant__code", "round_number", "decision_seconds").iterator(chunk_size=2000)
        session_codes = set(players.values_list("session__code", flat=True).distinct())
    else:
        pq = [(p.session.code, p.participant.code, p.round_number, p.decision_seconds) for p in players]
        session_codes = set(p.session.code for p in players)

    # Build timing map (small)
    timing_map = {}
    for s_code, p_code, rnd, secs in pq:
        timing_map[(s_code, p_code, int(rnd))] = secs

    # Pull batches per session via SQL (stream-ish)
    # NOTE: still loads per session; if huge, fetch per session_code
    for s_code in session_codes:
        rows = get_batches_filtered(s_code)  # all rows for that session
        for b in rows:
            p_code = b.get("owner_code") or ""
            rnd = int(b.get("round_number") or 0)
            if rnd < 1 or rnd > Constants.num_rounds:
                continue
            ds = timing_map.get((s_code, p_code, rnd), "") if p_code else ""
            yield [
                s_code,
                p_code,
                rnd,
                b.get("role") or "",
                b.get("condition") or "",
                b.get("item_nr") or "",
                b.get("image") or "",
                b.get("sentences") or "",
                b.get("rewards") or "",
                "" if ds is None else ds,
            ]

page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
