from otree.api import *
import json
import logging
import os
import re
from sqlalchemy import create_engine, text
from django.db import models as djmodels

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


def get_all_batches_sql(session_code):
    engine = get_engine()
    sql = text("SELECT * FROM img_desc_batch WHERE session_code = :session_code")
    with engine.connect() as conn:
        result = conn.execute(sql, {"session_code": session_code})
        return [dict(zip(result.keys(), row)) for row in result]


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
    active_batch = models.IntegerField()
    study_id = models.StringField()
    completion_code = models.StringField()
    full_return_url = models.StringField()

    @property
    def get_active_batch(self):
        all_data = get_all_batches_sql(self.session.code)
        return [b for b in all_data if b["batch"] == self.active_batch]

    def check_for_batch_completion(self):
        session = self.session
        active_batch = self.active_batch
        all_data = get_all_batches_sql(session.code)
        remaining = [b for b in all_data if b["batch"] == active_batch and not b["processed"]]
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
    start_decision_time = djmodels.DateTimeField(null=True)
    end_decision_time = djmodels.DateTimeField(null=True)
    decision_seconds = models.FloatField()
    link_id = models.IntegerField(initial=0)

    def role(self):
        return self.inner_role

    def get_linked_batch(self):
        if not self.link_id:
            return None
        all_data = get_all_batches_sql(self.session.code)
        for b in all_data:
            if b["id"] == self.link_id:
                return b
        return None

    def get_previous_batch(self):
        """
        For interpreters:
        - If paired with producer 0 (partner_id == 0), we use current row's predefined sentences.
        - Otherwise, for Exp n, we need the producer sentence from Exp n-1 matching says:
          (same producer/interpreter/condition pairing, but in previous batch).
        """
        if self.inner_role != INTERPRETER:
            return dict(sentences="[]")

        l = self.get_linked_batch()
        if not l or l.get("partner_id", 0) == 0:
            return dict(sentences="[]")

        target_batch_idx = self.subsession.active_batch - 1
        all_data = get_all_batches_sql(self.session.code)

        for obj in all_data:
            if (
                obj["batch"] == target_batch_idx
                and obj["role"] == PRODUCER
                and obj["partner_id"] == l["id_in_group"]
                and obj["id_in_group"] == l["partner_id"]
                and obj["condition"] == l["condition"]
            ):
                return obj

        return dict(sentences="[]")

    def get_sentences_data(self):
        """
        Returns the sentence matrix (list of lists) for the interpreter page:
        - If interpreter paired with producer 0 -> use l["sentences"] (predefined in sheet)
        - Else -> use previous batch producer's sentences (Exp n-1)
        """
        l = self.get_linked_batch()
        if not l:
            return []

        try:
            if l.get("partner_id", 0) == 0:
                return json.loads(l["sentences"] or "[]")
            else:
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
        all_data = get_all_batches_sql(self.session.code)
        my_code = self.participant.code
        for b in all_data:
            if b["owner_code"] == my_code:
                sql_update_batch(b["id"], processed=True)
        self.subsession.check_for_batch_completion()

    def get_full_sentences(self):
        """
        Robust sentence reconstruction:
        - Ignores None / "None" / "" values
        - Adds suffixes after each non-empty field
        """
        prefix = self.session.vars.get("prefix", "")
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []

        res = []
        for sentence in sentences:
            if not isinstance(sentence, list):
                continue

            tokens = []
            if prefix:
                tokens.append(str(prefix).strip())

            for val, suf in zip(sentence, suffixes):
                if val is None:
                    continue
                sval = str(val).strip()
                if not sval or sval.lower() == "none":
                    continue
                tokens.append(sval)
                if suf:
                    tokens.append(str(suf).strip())

            final = " ".join([t for t in tokens if t])
            if final:
                res.append(final)

        return res

    def get_image_url(self):
        """
        Producer image only.
        IMPORTANT: if a real participant gets assigned producer 0 rows, image may be NA/NA_x.
        We prevent that by excluding id_in_group==0 in start().
        """
        l = self.get_linked_batch()
        if not l:
            return ""
        image_name = l.get("image") or ""
        if not image_name:
            return ""

        ext = self.session.vars.get("extension", "png")
        if not image_name.lower().endswith(f".{ext}"):
            image_name = f"{image_name}.{ext}"

        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        if "amazonaws.com" in base:
            base = base.replace("/practice", "")

        return f"{base}/{image_name}"

    def start(self):
        session = self.session
        subsession = self.subsession
        all_data = get_all_batches_sql(session.code)

        if self.round_number == 1:
            # IMPORTANT FIX: exclude id_in_group==0 (virtual producer)
            candidates = [
                b for b in all_data
                if b["batch"] == subsession.active_batch
                and not b["busy"]
                and b["owner_code"]
