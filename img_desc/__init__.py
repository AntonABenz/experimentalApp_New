# img_desc/__init__.py

from otree.api import *
import json
import logging
from pprint import pprint
import os

# Use independent SQLAlchemy engine to bypass oTree wrappers
from sqlalchemy import create_engine, text

from django.db import models as djmodels
from django.forms.models import model_to_dict
from django.shortcuts import redirect

from .utils import get_url_for_image, get_completion_info, increase_space
from reading_xls.get_data import get_data

logger = logging.getLogger("benzapp.img_desc")

PRODUCER = "P"
INTERPRETER = "I"

STUBURL = "https://app.prolific.co/submissions/complete?cc="


# =====================================================================
# CONSTANTS
# =====================================================================

class Constants(BaseConstants):
    name_in_url = "img_desc"
    players_per_group = None
    num_rounds = 85

    STUBURL = STUBURL
    PLACEMENT_ERR = "ERROR_BATCH_PLACEMENT"
    API_ERR = "API_ERROR"
    FALLBACK_URL = STUBURL + PLACEMENT_ERR
    API_ERR_URL = STUBURL + API_ERR

    INTERPRETER = INTERPRETER
    PRODUCER = PRODUCER


# =====================================================================
# BATCH MODEL
# =====================================================================

class Batch(ExtraModel):
    """
    Table definition. We access this via Raw SQL helpers below.
    """
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


# =====================================================================
# SQL ENGINE HELPER (Independent Connection)
# =====================================================================

_custom_engine = None

def get_engine():
    """
    Creates or returns a cached SQLAlchemy engine based on the environment.
    Includes fixes for Heroku Postgres URL and SQLite locking.
    """
    global _custom_engine
    if _custom_engine:
        return _custom_engine

    # Get DB URL from environment (standard on Heroku)
    db_url = os.environ.get('DATABASE_URL')
    
    if not db_url:
        # Fallback for local testing / SQLite
        db_url = "sqlite:///db.sqlite3"
        # FIX: Add timeout (default is usually 5s, we set 30s) to wait for locks
        _custom_engine = create_engine(db_url, connect_args={'timeout': 30})
    else:
        # Fix for Heroku: SQLAlchemy requires 'postgresql://', Heroku provides 'postgres://'
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
        _custom_engine = create_engine(db_url)

    return _custom_engine


def get_all_batches_sql(session_code):
    """
    Fetch all batches for this session using custom engine.
    Compatible with older SQLAlchemy versions.
    """
    engine = get_engine()
    sql = text("SELECT * FROM img_desc_batch WHERE session_code = :session_code")
    
    with engine.connect() as conn:
        result = conn.execute(sql, session_code=session_code)
        # Robust row-to-dict conversion compatible with older SQLAlchemy
        return [dict(zip(result.keys(), row)) for row in result]

def sql_update_batch(batch_id, **kwargs):
    """
    Update specific fields of a batch row by ID.
    Uses explicit transactions to handle concurrency/locking.
    """
    if not kwargs:
        return

    engine = get_engine()
    set_clauses = []
    params = {'id': batch_id}
    
    for k, v in kwargs.items():
        set_clauses.append(f"{k} = :{k}")
        # Convert bools to integers for SQL compatibility
        if isinstance(v, bool):
            params[k] = 1 if v else 0
        else:
            params[k] = v
    
    sql_str = f"UPDATE img_desc_batch SET {', '.join(set_clauses)} WHERE id = :id"
    sql = text(sql_str)
    
    # FIX: Use transaction block to ensure proper locking/unlocking
    with engine.connect() as conn:
        # .begin() starts a transaction; it commits on exit or rolls back on error
        with conn.begin():
            conn.execute(sql, **params)


# =====================================================================
# SUBSESSION
# =====================================================================

class Subsession(BaseSubsession):
    active_batch = models.IntegerField()
    study_id = models.StringField()
    completion_code = models.StringField()
    full_return_url = models.StringField()

    @property
    def get_active_batch(self):
        all_data = get_all_batches_sql(self.session.code)
        return [b for b in all_data if b['batch'] == self.active_batch]

    def expand_slots(self):
        study_id = self.study_id
        max_users = self.session.vars.get("max_users", 0)
        batch_size = self.session.vars.get("batch_size", 0)

        if not study_id:
            logger.warning("No study id data available! Slot expansion failed.")
            return

        logger.info(f"Expanding slots: study {study_id}, extra {batch_size}")
        max_users = min(max_users, self.session.num_participants)
        pprint(increase_space(study_id=study_id, num_extra=batch_size, max_users=max_users))

    def check_for_batch_completion(self):
        session = self.session
        active_batch = self.active_batch

        logger.info(f"Checking batch {active_batch} completion...")

        all_data = get_all_batches_sql(session.code)
        remaining = [
            b for b in all_data 
            if b['batch'] == active_batch and not b['processed']
        ]
        
        logger.info(f"Active batch: {active_batch}; Remaining: {len(remaining)}")
        
        if remaining:
            return

        session.vars["active_batch"] = active_batch + 1
        logger.info(f"Batch {active_batch} complete. Moving to {active_batch + 1}")

        Subsession.objects.filter(session=session).update(
            active_batch=active_batch + 1
        )

        if session.config.get("expand_slots", False):
            self.expand_slots()


class Group(BaseGroup):
    pass


# =====================================================================
# PLAYER
# =====================================================================

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
    full_return_url = models.StringField()
    vars_dump = models.LongStringField()
    producer_decision = models.LongStringField()
    interpreter_decision = models.LongStringField()
    start_decision_time = djmodels.DateTimeField(null=True)
    end_decision_time = djmodels.DateTimeField(null=True)
    decision_seconds = models.FloatField()
    link_id = models.IntegerField(initial=0)

    def role(self):
        return self.inner_role

    @property
    def link(self):
        if not self.link_id:
            return None
        
        all_data = get_all_batches_sql(self.session.code)
        for b in all_data:
            if b['id'] == self.link_id:
                return b
        return None

    def get_sentences_data(self):
        l = self.link
        if not l:
            return []
        try:
            if l['partner_id'] == 0:
                return json.loads(l['sentences'])
            else:
                return json.loads(self.get_previous_batch().get("sentences"))
        except Exception as e:
            logger.error(f"Error parsing sentences: {e}")
            return []

    def get_previous_batch(self):
        if self.inner_role != INTERPRETER:
            return dict(sentences="[]")

        l = self.link
        if not l or l['partner_id'] == 0:
            return dict(sentences="[]")

        target_batch_idx = self.subsession.active_batch - 1
        all_data = get_all_batches_sql(self.session.code)

        for obj in all_data:
            if (obj['batch'] == target_batch_idx and
                obj['role'] == PRODUCER and
                obj['partner_id'] == l['id_in_group'] and
                obj['id_in_group'] == l['partner_id'] and
                obj['condition'] == l['condition']):
                return obj
                
        logger.error("Previous batch row not found")
        return dict(sentences="[]")

    def update_batch(self):
        if not self.link_id:
            return
        
        updates = {}
        if self.inner_role == PRODUCER:
            updates['sentences'] = self.producer_decision
        if self.inner_role == INTERPRETER:
            updates['rewards'] = self.interpreter_decision
            
        if updates:
            sql_update_batch(self.link_id, **updates)

    def mark_data_processed(self):
        self.participant.vars["full_study_completed"] = True
        
        all_data = get_all_batches_sql(self.session.code)
        my_code = self.participant.code
        
        for b in all_data:
            if b['owner_code'] == my_code:
                sql_update_batch(b['id'], processed=True)

        self.subsession.check_for_batch_completion()

    def get_full_sentences(self):
        prefix = self.session.vars.get("prefix", "")
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []
        sentences = [sub for sub in sentences if "" not in sub]

        res = []
        for sentence in sentences:
            expansion = [str(item) for pair in zip(sentence, suffixes) for item in pair]
            if prefix:
                expansion.insert(0, prefix)
            res.append(" ".join(expansion))
        return res

    def get_image_url(self):
        l = self.link
        image = l['image'] if l else ""
        return get_url_for_image(self, image)

    def start(self):
        session = self.session
        subsession = self.subsession

        # --- ROUND 1: assign free slot ---
        if self.round_number == 1:
            all_data = get_all_batches_sql(session.code)
            
            # Find candidates
            candidates = [
                b for b in all_data 
                if b['batch'] == subsession.active_batch
                and not b['busy']
                and b['owner_code'] == ""
            ]
            
            if not candidates:
                logger.error(f"No free slots for {self.participant.code}!")
                self.faulty = True
                return
        
            candidates.sort(key=lambda b: b['id_in_group'])
            free = candidates[0]
            
            sql_update_batch(
                free['id'], 
                busy=True, 
                owner_code=self.participant.code
            )
        
        # --- EVERY ROUND: Find my row ---
        all_data = get_all_batches_sql(session.code)
        my_row = None
        for b in all_data:
            if b['owner_code'] == self.participant.code and b['round_number'] == self.round_number:
                my_row = b
                break
                
        if not my_row:
            logger.error(f"No row found for round {self.round_number}")
            self.faulty = True
            return
        
        self.link_id = my_row['id']
        self.inner_role = my_row['role']
        self.inner_sentences = json.dumps(self.get_sentences_data())
        

        # --- PROLIFIC ---
        if self.round_number == 1 and session.config.get("for_prolific"):
            vars_ = self.participant.vars
            prolific_id = vars_.get("prolific_id") or vars_.get("prolific_pid")
            prol_study_id = vars_.get("study_id")
            prol_session_id = vars_.get("prolific_session_id") 

            completion_info = dict(
                completion_code=Constants.API_ERR, 
                full_return_url=Constants.API_ERR_URL
            )

            if prol_study_id:
                if not subsession.study_id:
                    c = get_completion_info(prol_study_id)
                    if c:
                        Subsession.objects.filter(session=session).update(
                            study_id=prol_study_id,
                            completion_code=c["completion_code"],
                            full_return_url=c["full_return_url"],
                        )

                if prol_study_id == subsession.study_id:
                    completion_info = dict(
                        completion_code=subsession.completion_code,
                        full_return_url=subsession.full_return_url,
                    )
                else:
                    c = get_completion_info(prol_study_id)
                    if c: completion_info = c

            for_update = dict(
                prolific_id=prolific_id,
                prol_study_id=prol_study_id,
                prol_session_id=prol_session_id,
                **completion_info,
            )

            try:
                if not all([prol_study_id, prolific_id, prol_session_id]):
                    raise Exception("Missing Prolific params")
            except Exception as e:
                logger.error(f"Prolific data error: {e}")
            finally:
                for p in self.in_all_rounds():
                    for k, v in for_update.items():
                        setattr(p, k, v)
                
                if prol_session_id:
                    self.participant.label = prol_session_id


# =====================================================================
# SESSION CREATION
# =====================================================================

def creating_session(subsession: Subsession):
    session = subsession.session
    if subsession.round_number != 1:
        return

    subsession.active_batch = 1
    session.vars["active_batch"] = 1

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Missing 'filename' in session config")

    excel_data = get_data(filename)
    df = excel_data.get("data")

    session.vars["user_data"] = df
    session.vars["num_rounds"] = int(df.group_enumeration.max())

    logger.info(f'TOTAL NUM ROUNDS: {session.vars["num_rounds"]}')
    
    records = df.to_dict(orient="records")
    for r in records:
        Batch.create(
            session_code=session.code,
            owner_code="",
            batch=r.get("Exp"),
            item_nr=r.get("Item.Nr"),
            condition=r.get("Condition"),
            image=r.get("Item"),
            round_number=r.get("group_enumeration"),
            role=r.get("role"),
            id_in_group=r.get("id"),
            partner_id=r.get("partner_id"),
            sentences=r.get("sentences"),
        )

    session.vars["practice_settings"] = excel_data.get("practice_settings")
    session.vars["practice_pages"] = excel_data.get("practice_pages")
    session.vars["user_settings"] = excel_data.get("settings")
    
    settings = excel_data.get("settings") or {}
    for k in ["s3path", "extension", "prefix", "interpreter_choices", 
              "interpreter_input_type", "interpreter_input_choices", "interpreter_title"]:
        session.vars[k] = settings.get(k)

    session.vars["suffixes"] = settings.get("suffixes") or []
    
    av = settings.get("allowed_values") or []
    session.vars["allowed_values"] = [[x for x in sub if x != ""] for sub in av]
    session.vars["allowed_regex"] = settings.get("allowed_regex") or []
    
    session.vars["caseflag"] = settings.get("caseflag") in ["True", "true", "1", "t", "T"]
    session.vars["EndOfIntroText"] = settings.get("EndOfIntroText", "")

    unique_ids_wz = [x for x in df.id.unique() if x != 0]
    unique_exps = df[df.Exp != 0].Exp.unique()
    batch_size = len(unique_ids_wz)
    max_users = batch_size * len(unique_exps)

    if session.config.get("expand_slots"):
        assert max_users <= session.num_participants

    session.vars["max_users"] = max_users
    session.vars["batch_size"] = batch_size
    logger.info(f"{max_users=}; {batch_size=}")


# =====================================================================
# PAGES
# =====================================================================

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

    @staticmethod
    def is_displayed(player):
        if not player.faulty:
            player.start()
        return player.round_number <= player.session.vars["num_rounds"]

    @staticmethod
    def vars_for_template(player):
        if player.link:
            return dict(d=player.link)
        return dict(d=None)

    def post(self):
        logger.info(f"POST {self.participant.code} round {self.round_number}")
        
        for t in ["start_decision_time", "end_decision_time"]:
            v = self.request.POST.get(t)
            if v: setattr(self.player, t, v)

        dec_sec = self.request.POST.get("decision_seconds")
        if dec_sec:
            try:
                self.player.decision_seconds = float(dec_sec)
            except: pass

        field = "producer_decision" if self.player.inner_role == PRODUCER else "interpreter_decision"
        raw = self.request.POST.get(field)
        
        if raw:
            decisions = json.loads(raw)
            if self.player.inner_role == PRODUCER:
                flatten = [list(i.values()) for i in decisions]
                self.player.producer_decision = json.dumps(flatten)
                self.player.inner_sentences = json.dumps(flatten)
            else:
                flatten = [i.get("choice") for i in decisions]
                self.player.interpreter_decision = json.dumps(flatten)

        return super().post()

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.update_batch()
        
        if player.round_number == player.session.vars["num_rounds"]:
            player.mark_data_processed()
            try:
                player.vars_dump = json.dumps(player.participant.vars)
            except Exception as e:
                logger.error(e)


class Feedback(Page):
    form_model = "player"
    form_fields = ["feedback"]

    def is_displayed(self):
        return self.round_number == self.session.vars["num_rounds"]


class FinalForProlific(Page):
    @staticmethod
    def is_displayed(player):
        return (
            player.session.config.get("for_prolific")
            and player.round_number == player.session.vars["num_rounds"]
        )

    def get(self):
        if self.player.full_return_url:
            return redirect(self.player.full_return_url)
        return redirect("https://cnn.com")


page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
