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
    batch = models.IntegerField()          # Exp index (0,1,2,...)
    id_in_group = models.IntegerField()    # "player id" (0..4)
    partner_id = models.IntegerField()     # paired player id
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


def _safe_json_load(x, default):
    try:
        if x is None:
            return default
        if isinstance(x, (dict, list)):
            return x
        s = str(x).strip()
        if not s:
            return default
        return json.loads(s)
    except Exception:
        return default


def _normalize_token(val):
    """Excel sometimes contains the literal string 'None'. Treat it like empty."""
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() == "none":
        return ""
    return s


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
        remaining = [
            b for b in all_data
            if b["batch"] == active_batch and not b["processed"]
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

    # IMPORTANT: allow blank so we can safely submit only one depending on role
    producer_decision = models.LongStringField(blank=True)
    interpreter_decision = models.LongStringField(blank=True)

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

    def _find_producer_row_in_db(self, exp_idx, producer_id, interpreter_id, condition):
        """Find the producer row in the DB for the previous exp with matching key."""
        all_data = get_all_batches_sql(self.session.code)
        for obj in all_data:
            if (
                obj.get("batch") == exp_idx
                and obj.get("role") == PRODUCER
                and obj.get("id_in_group") == producer_id
                and obj.get("partner_id") == interpreter_id
                and obj.get("condition") == condition
            ):
                return obj
        return None

    def get_sentences_data(self):
        """
        Implements Anton's intended logic:

        Let current row = l (this player's row for this round).
        If interpreter is paired with producer 0:
            use sentences from same row (predefined)
        Else:
            use sentences from producer row in Exp-1 with same (producer, interpreter, condition)
            (Works for Exp 1 too, because Exp 0 rows are already in the DB)
        """
        l = self.get_linked_batch()
        if not l:
            return []

        # Only interpreters need this lookup; producers' sentences are their own entry.
        if self.inner_role != INTERPRETER:
            return _safe_json_load(l.get("sentences"), [])

        producer_id = l.get("partner_id")     # interpreter row's partner is producer
        interpreter_id = l.get("id_in_group") # interpreter id is own id_in_group
        condition = l.get("condition")
        exp_idx = l.get("batch")

        # Case 1: paired with virtual producer 0 -> predefined sentence is on same row
        if producer_id == 0:
            return _safe_json_load(l.get("sentences"), [])

        # Case 2: paired with real producer -> lookup producer row in Exp-1
        prev_exp = exp_idx - 1
        if prev_exp < 0:
            return []

        prev_producer_row = self._find_producer_row_in_db(
            exp_idx=prev_exp,
            producer_id=producer_id,
            interpreter_id=interpreter_id,
            condition=condition,
        )

        if not prev_producer_row:
            return []

        return _safe_json_load(prev_producer_row.get("sentences"), [])

    def update_batch(self):
        if not self.link_id:
            return

        updates = {}
        if self.inner_role == PRODUCER:
            updates["sentences"] = self.producer_decision or "[]"
        if self.inner_role == INTERPRETER:
            updates["rewards"] = self.interpreter_decision or "[]"

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
        prefix = self.session.vars.get("prefix", "") or ""
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []

        # sentences should be a list of list tokens: [[tok1,tok2,...],[...]]
        cleaned = []
        for sub in sentences:
            if not isinstance(sub, list):
                continue
            sub2 = [_normalize_token(x) for x in sub]
            # drop if any field empty (prevents broken partial sentences)
            if any(x == "" for x in sub2):
                continue
            cleaned.append(sub2)

        res = []
        for sentence in cleaned:
            # interleave tokens with suffixes
            expansion = []
            for tok, suf in zip(sentence, suffixes):
                expansion.append(str(tok))
                expansion.append(str(suf))
            # If sentence longer than suffixes, append remaining tokens
            if len(sentence) > len(suffixes):
                expansion.extend([str(x) for x in sentence[len(suffixes):]])

            if prefix:
                expansion.insert(0, prefix)

            # remove empty suffix tokens
            expansion = [x for x in expansion if str(x).strip() != ""]
            res.append(" ".join(expansion))

        return res

    def get_image_url(self):
        l = self.get_linked_batch()
        if not l:
            return ""

        # Only real producers should show images; producer 0 never should be assigned anyway
        image_name = l.get("image")
        if not image_name:
            return ""

        image_name = str(image_name).strip()
        if image_name.lower() in {"na", "na_x", "none"}:
            return ""

        ext = self.session.vars.get("extension", "png") or "png"
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
        all_data = get_all_batches_sql(session.code)

        # First round: choose one "player id" line (id_in_group) but NEVER id_in_group==0
        if self.round_number == 1:
            candidates = [
                b for b in all_data
                if b["batch"] == subsession.active_batch
                and not b["busy"]
                and b["owner_code"] == ""
                and int(b.get("id_in_group") or 0) != 0  # IMPORTANT: exclude virtual player 0
            ]
            if not candidates:
                self.faulty = True
                return

            candidates.sort(key=lambda b: b["id_in_group"])
            free = candidates[0]
            chosen_batch = free["batch"]
            chosen_id = free["id_in_group"]

            self.batch = chosen_batch

            # mark ALL rows for this (batch, id_in_group) as owned by this participant
            for b in all_data:
                if b["batch"] == chosen_batch and b["id_in_group"] == chosen_id:
                    sql_update_batch(b["id"], busy=True, owner_code=self.participant.code)

            all_data = get_all_batches_sql(session.code)

        # Find the row for this participant and round
        my_row = None
        for b in all_data:
            if (
                b["owner_code"] == self.participant.code
                and b["round_number"] == self.round_number
            ):
                my_row = b
                break

        if not my_row:
            self.faulty = True
            return

        self.link_id = my_row["id"]
        self.inner_role = my_row["role"]  # "P" or "I"
        self.inner_sentences = json.dumps(self.get_sentences_data())

        if self.round_number == 1 and session.config.get("for_prolific"):
            p = self.participant
            vars_ = p.vars
            prolific_id = vars_.get("prolific_id") or vars_.get("prolific_pid")
            if vars_.get("prolific_session_id"):
                p.label = vars_.get("prolific_session_id")


def creating_session(subsession: Subsession):
    session = subsession.session
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

    settings = excel_data.get("settings") or {}
    clean_settings = {normalize_key(k): v for k, v in settings.items()}
    session.vars["user_settings"] = clean_settings

    # Basic settings
    for k in ["s3path_base", "extension", "prefix", "interpreter_choices", "interpreter_title"]:
        session.vars[k] = clean_settings.get(normalize_key(k))

    session.vars["suffixes"] = clean_settings.get("suffixes") or []

    # Allowed values lists (used for help modal)
    allowed_values = []
    for i in range(1, 6):
        key = f"allowed_values_{i}"
        val = clean_settings.get(key)
        if val:
            allowed_values.append([x.strip() for x in str(val).split(";") if x.strip()])
        else:
            allowed_values.append([])
    session.vars["allowed_values"] = allowed_values

    # Allowed regex patterns (used for validation)
    allowed_regex = []
    for i in range(1, 6):
        key = f"allowed_regex_{i}"
        val = clean_settings.get(key)
        allowed_regex.append(str(val).strip() if val else "")
    session.vars["allowed_regex"] = allowed_regex


class FaultyCatcher(Page):
    @staticmethod
    def is_displayed(player):
        return player.faulty

    def get(self):
        if self.player.faulty:
            return redirect(Constants.FALLBACK_URL)
        return super().get()


class Q(Page):
    form_model = "player"

    @staticmethod
    def is_displayed(player):
        if player.faulty:
            return False

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
        # parse interpreter choices from settings
        raw_choices = player.session.vars.get("interpreter_choices") or ""
        if isinstance(raw_choices, str):
            interpreter_choices = [x.strip() for x in raw_choices.split(";") if x.strip()]
        elif isinstance(raw_choices, list):
            interpreter_choices = raw_choices
        else:
            interpreter_choices = []

        interpreter_title = player.session.vars.get("interpreter_title") or "Buy medals:"

        return dict(
            suffixes=player.session.vars.get("suffixes", []),
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regex=player.session.vars.get("allowed_regex", []),
            interpreter_choices=interpreter_choices,
            interpreter_title=interpreter_title,
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
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


page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
