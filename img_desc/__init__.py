from otree.api import *
import json
import logging
import os
import re
import time
from sqlalchemy import create_engine, text
from django.shortcuts import redirect
from django.db import models as django_models  # <--- REQUIRED for Batch.objects

logger = logging.getLogger("benzapp.img_desc")

PRODUCER = "P"
INTERPRETER = "I"

STUBURL = "https://app.prolific.co/submissions/complete?cc="

# ----------------------------
# Helpers
# ----------------------------
def normalize_key(key):
    if not key:
        return ""
    return re.sub(r"[\s_]+", "_", str(key).lower().strip())

def _is_nan_like(x) -> bool:
    try:
        return isinstance(x, float) and x != x
    except Exception:
        return False

def _force_str_keep_none_token(x) -> str:
    if x is None or _is_nan_like(x):
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    if s.lower() == "none":
        return "None"
    return s

def _ensure_json_list(s: str) -> str:
    if not s:
        return "[]"
    s = str(s).strip()
    if not s:
        return "[]"
    try:
        v = json.loads(s)
        return json.dumps(v if isinstance(v, list) else [])
    except Exception:
        return "[]"

def _parse_tokens_list(raw):
    if raw is None:
        return []
    try:
        v = json.loads(raw) if isinstance(raw, str) else raw
    except Exception:
        return []
    if not isinstance(v, list):
        return []
    out = []
    for row in v:
        if not isinstance(row, list):
            continue
        cleaned = []
        for tok in row:
            tok_s = _force_str_keep_none_token(tok)
            if tok_s == "":
                continue
            cleaned.append(tok_s)
        if cleaned:
            out.append(cleaned)
    return out

# ----------------------------
# Constants
# ----------------------------
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

# ----------------------------
# Batch Model (Standard Django)
# ----------------------------
class Batch(ExtraModel):
    # This Manager line allows Batch.objects.filter(...) to work
    objects = django_models.Manager()

    # Standard Django fields to prevent "Cannot resolve keyword" errors
    session_code = django_models.CharField(max_length=255)
    owner_code = django_models.CharField(max_length=255, blank=True, default="")
    
    # TextFields for longer content
    sentences = django_models.TextField(blank=True, default="")
    rewards = django_models.TextField(blank=True, default="")
    
    condition = django_models.CharField(max_length=255, blank=True, default="")
    item_nr = django_models.CharField(max_length=255, blank=True, default="")
    image = django_models.CharField(max_length=255, blank=True, default="")
    
    round_number = django_models.IntegerField(default=0)
    role = django_models.CharField(max_length=255, blank=True, default="")
    
    batch = django_models.IntegerField(default=0)
    id_in_group = django_models.IntegerField(default=0)
    partner_id = django_models.IntegerField(default=0)
    
    busy = django_models.BooleanField(default=False)
    processed = django_models.BooleanField(default=False)

# ----------------------------
# SQL Helpers (Low Memory)
# ----------------------------
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

def get_batches_filtered(session_code, batch_val=None, round_val=None, owner_code=None, role=None, partner_id=None, id_in_group=None):
    """
    Optimized Fetch: Loads only necessary rows to prevent Memory Crashes.
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
    if role is not None:
        clauses.append("role = :role")
        params["role"] = role
    if partner_id is not None:
        clauses.append("partner_id = :partner_id")
        params["partner_id"] = partner_id
    if id_in_group is not None:
        clauses.append("id_in_group = :id_in_group")
        params["id_in_group"] = id_in_group

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

class Subsession(BaseSubsession):
    active_batch = models.IntegerField(initial=1)
    study_id = models.StringField(blank=True)
    completion_code = models.StringField(blank=True)
    full_return_url = models.StringField(blank=True)

    def check_for_batch_completion(self):
        active_batch = self.field_maybe_none("active_batch") or 1
        rows = get_batches_filtered(self.session.code, batch_val=active_batch)
        remaining = any(not r['processed'] for r in rows)
        
        if remaining:
            return
        self.active_batch = active_batch + 1
        self.session.vars["active_batch"] = self.active_batch
        self.save()

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    inner_role = models.StringField(blank=True)
    inner_sentences = models.LongStringField(blank=True)
    batch = models.IntegerField(initial=1)
    faulty = models.BooleanField(initial=False)
    feedback = models.LongStringField(blank=True)
    prolific_id = models.StringField(blank=True)
    prol_study_id = models.StringField(blank=True)
    prol_session_id = models.StringField(blank=True)
    completion_code = models.StringField(blank=True)
    full_return_url = models.StringField(blank=True)
    vars_dump = models.LongStringField(blank=True)
    producer_decision = models.LongStringField(blank=True)
    interpreter_decision = models.LongStringField(blank=True)
    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)
    link_id = models.IntegerField(initial=0)

    def role(self):
        return self.inner_role

    def get_linked_batch(self):
        if not self.link_id:
            return None
        return get_single_batch_by_id(self.link_id)

    def get_previous_batch(self):
        if self.inner_role != INTERPRETER:
            return {"sentences": "[]"}
        
        l = self.get_linked_batch()
        if not l:
            return {"sentences": "[]"}
        if l.get("partner_id") in (0, None):
            return {"sentences": "[]"}

        active_batch = self.subsession.field_maybe_none("active_batch") or 1
        target_batch = active_batch - 1
        if target_batch < 1:
            return {"sentences": "[]"}

        matches = get_batches_filtered(
            self.session.code,
            batch_val=target_batch,
            role=PRODUCER,
            partner_id=l["id_in_group"],
            id_in_group=l["partner_id"]
        )
        
        for m in matches:
            if m.get("condition") == l.get("condition"):
                return m
        
        return {"sentences": "[]"}

    def get_sentences_data(self):
        l = self.get_linked_batch()
        if not l:
            return []
        if self.inner_role == PRODUCER:
            return _parse_tokens_list(l.get("sentences") or "[]")
        
        if l.get("partner_id") in (0, None):
            return _parse_tokens_list(l.get("sentences") or "[]")

        prev = self.get_previous_batch()
        return _parse_tokens_list(prev.get("sentences") or "[]")

    def update_batch(self):
        if not self.link_id:
            return
        updates = {}
        if self.inner_role == PRODUCER:
            updates["sentences"] = _ensure_json_list(self.producer_decision)
        elif self.inner_role == INTERPRETER:
            updates["rewards"] = self.interpreter_decision
        
        if updates:
            sql_update_batch(self.link_id, **updates)

    def mark_data_processed(self):
        my_code = self.participant.code
        my_batches = get_batches_filtered(self.session.code, owner_code=my_code)
        for b in my_batches:
            sql_update_batch(b['id'], processed=True)
        self.subsession.check_for_batch_completion()

    def get_full_sentences(self):
        prefix = self.session.vars.get("prefix") or ""
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []
        res = []
        for sentence_tokens in sentences:
            expansion = [str(item) for pair in zip(sentence_tokens, suffixes) for item in pair]
            if prefix:
                expansion.insert(0, prefix)
            res.append(" ".join(expansion))
        return res

    def get_image_url(self):
        l = self.get_linked_batch()
        if not l:
            return ""
        image_name = _force_str_keep_none_token(l.get("image"))
        if not image_name or image_name.lower() in {"na", "na_x", "nan", "x"}:
            return ""
        ext = self.session.vars.get("extension") or "png"
        if not image_name.lower().endswith(f".{ext}"):
            image_name = f"{image_name}.{ext}"
        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        if not base:
            return ""
        if "amazonaws.com" in base:
            base = base.replace("/practice", "")
        return f"{base}/{image_name}"

    def start(self):
        session = self.session
        subsession = self.subsession

        # 1. Try to find an EXISTING assignment for this round
        my_rows = get_batches_filtered(
            session.code,
            round_val=self.round_number,
            owner_code=self.participant.code
        )

        # 2. If no assignment, we need to CLAIM a new batch
        if not my_rows:
            if subsession.field_maybe_none("active_batch") is None:
                subsession.active_batch = 1
                subsession.save()
            
            # Fetch candidates using SQL optimization
            active_candidates = get_batches_filtered(session.code, batch_val=subsession.active_batch)
            candidates = [
                b for b in active_candidates
                if not b["busy"] and (b.get("owner_code") or "") == ""
            ]
            
            if not candidates:
                self.faulty = True
                return

            candidates.sort(key=lambda b: b["id_in_group"])
            free = candidates[0]
            chosen_batch = free["batch"]
            chosen_id = free["id_in_group"]
            self.batch = chosen_batch

            # Claim the batch
            sql_update_batch(
                free["id"],
                busy=True,
                owner_code=self.participant.code
            )

            # Refetch MY row
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
            vars_ = self.participant.vars
            if vars_.get("prolific_session_id"):
                self.participant.label = vars_.get("prolific_session_id")

# ----------------------------
# Session Creation
# ----------------------------
def creating_session(subsession: Subsession):
    session = subsession.session
    if subsession.field_maybe_none("active_batch") is None:
        subsession.active_batch = 1
        subsession.save()

    if subsession.round_number != 1:
        return

    session.vars["active_batch"] = 1
    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Missing filename in session config")

    from reading_xls.get_data import get_data
    excel_data = get_data(filename)
    df = excel_data.get("data")
    if df is None:
        raise RuntimeError("Excel loader returned no 'data' sheet")

    for col in ["Condition", "Item.Nr", "Item", "role", "sentences"]:
        if col in df.columns:
            df[col] = df[col].apply(_force_str_keep_none_token)

    for i in range(1, 6):
        c = f"Sentence_{i}"
        if c in df.columns:
            df[c] = df[c].apply(_force_str_keep_none_token)

    # Use Batch.objects.filter().delete() because we are now using Standard Django
    Batch.objects.filter(session_code=session.code).delete()

    records = df.to_dict(orient="records")
    for r in records:
        Batch.objects.create(
            session_code=session.code,
            owner_code="",
            batch=int(r.get("Exp") or 0),
            item_nr=_force_str_keep_none_token(r.get("Item.Nr")),
            condition=_force_str_keep_none_token(r.get("Condition")),
            image=_force_str_keep_none_token(r.get("Item")),
            round_number=int(r.get("group_enumeration") or 0),
            role=_force_str_keep_none_token(r.get("role")),
            id_in_group=int(r.get("id") or 0),
            partner_id=int(r.get("partner_id") or 0),
            sentences=_ensure_json_list(_force_str_keep_none_token(r.get("sentences"))),
        )

    settings = excel_data.get("settings") or {}
    clean_settings = {normalize_key(k): v for k, v in settings.items()}

    session.vars["s3path_base"] = _force_str_keep_none_token(clean_settings.get("s3path_base"))
    session.vars["extension"] = _force_str_keep_none_token(clean_settings.get("extension")) or "png"
    session.vars["prefix"] = _force_str_keep_none_token(clean_settings.get("prefix"))
    session.vars["interpreter_choices"] = clean_settings.get("interpreter_choices") or ""
    session.vars["interpreter_title"] = _force_str_keep_none_token(clean_settings.get("interpreter_title")) or "Buy medals:"

    suffixes = []
    if isinstance(clean_settings.get("suffixes"), list):
        suffixes = [str(x) for x in clean_settings.get("suffixes") if str(x).strip()]
    else:
        for i in range(1, 11):
            s = _force_str_keep_none_token(clean_settings.get(f"suffix_{i}"))
            if s:
                suffixes.append(s)
    session.vars["suffixes"] = suffixes

    allowed_values = []
    allowed_regexes = []
    for i in range(1, 11):
        v_val = clean_settings.get(f"allowed_values_{i}")
        r_val = clean_settings.get(f"allowed_regex_{i}")
        if v_val is None and r_val is None:
            continue
        if v_val:
            allowed_values.append([x.strip() for x in str(v_val).split(";") if x.strip()])
        else:
            allowed_values.append([])
        allowed_regexes.append(str(r_val).strip() if r_val else "")

    session.vars["allowed_values"] = allowed_values
    session.vars["allowed_regexes"] = allowed_regexes

    caseflag_raw = _force_str_keep_none_token(clean_settings.get("caseflag"))
    session.vars["caseflag"] = caseflag_raw.strip().lower() in {"true", "1", "t", "yes", "y"}

    default_url = (
        "https://docs.google.com/document/d/e/"
        "2PACX-1vTg_Hd8hXK-TZS77rC6W_BlY2NtWhQqCLzlgW0LeomoEUdhoDNYPNVOO7Pt6g0-JksykUrgRdtcVL3u/"
        "pub?embedded=true"
    )
    url_from_settings = _force_str_keep_none_token(clean_settings.get("instructions_url"))
    session.vars["instructions_url"] = url_from_settings or default_url

    cc = session.config.get("completion_code") or clean_settings.get("completion_code")
    if cc:
        session.vars["completion_code"] = _force_str_keep_none_token(cc)

# ----------------------------
# Pages
# ----------------------------
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
        if role == INTERPRETER:
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

        return dict(
            d=player.get_linked_batch(),
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regexes=player.session.vars.get("allowed_regexes", []),
            caseflag=player.session.vars.get("caseflag", False),
            suffixes=player.session.vars.get("suffixes", []),
            prefix=player.session.vars.get("prefix", ""),
            interpreter_choices=interpreter_choices,
            interpreter_title=player.session.vars.get("interpreter_title", "Buy medals:"),
            instructions_url=player.session.vars.get("instructions_url"),
            server_image_url=player.get_image_url(),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.end_decision_time = time.time()
        player.decision_seconds = max(0.0, player.end_decision_time - (player.start_decision_time or player.end_decision_time))
        player.update_batch()
        if player.round_number == Constants.num_rounds:
            player.mark_data_processed()

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
        url = self.player.field_maybe_none("full_return_url")
        if url:
            return redirect(url)
        completion_code = (
            self.player.session.config.get("completion_code")
            or self.player.session.vars.get("completion_code")
            or self.player.subsession.field_maybe_none("completion_code")
        )
        if not completion_code:
            return redirect(Constants.API_ERR_URL)
        return redirect(Constants.STUBURL + str(completion_code))

def custom_export(players):
    """
    Export logic using Django ORM and Iterators.
    Includes FEEDBACK column.
    """
    yield [
        "session_code", "participant_code", "round_number", "role",
        "condition", "item_nr", "image", "producer_sentences",
        "interpreter_rewards", "decision_seconds", "feedback"
    ]

    from itertools import groupby
    pq = (
        players
        .values_list(
            "session__code", "participant__code", "round_number",
            "inner_role", "producer_decision", "interpreter_decision", "decision_seconds", "feedback"
        )
        .order_by("session__code", "participant__code", "round_number")
        .iterator(chunk_size=2000)
    )

    def keyfunc(row):
        return (row[0], row[1])

    # Since Batch uses Standard Django Models now, we can use objects.filter safely
    for (session_code, participant_code), rows in groupby(pq, key=keyfunc):
        bq = (
            Batch.objects
            .filter(session_code=session_code, owner_code=participant_code)
            .values_list("round_number", "role", "condition", "item_nr", "image", "sentences", "rewards")
            .iterator(chunk_size=500)
        )
        batch_map = {int(rn): (role, cond, item_nr, img, sent, rew) for rn, role, cond, item_nr, img, sent, rew in bq}

        for row in rows:
            _, _, rnd, inner_role, prod_dec, interp_dec, ds, fb = row
            rnd = int(rnd)
            
            # Filter rounds 1-80
            if rnd < 1 or rnd > Constants.num_rounds:
                continue

            b = batch_map.get(rnd)
            if b:
                role, cond, item_nr, img, sent, rew = b
                producer_sentences = sent or ""
                interpreter_rewards = rew or ""
            else:
                role = inner_role or ""
                cond = ""
                item_nr = ""
                img = ""
                producer_sentences = prod_dec or ""
                interpreter_rewards = interp_dec or ""

            if ds is None: ds = ""
            if fb is None: fb = ""

            yield [
                session_code, participant_code, rnd, role, cond, item_nr, img,
                producer_sentences, interpreter_rewards, ds, fb
            ]

page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
