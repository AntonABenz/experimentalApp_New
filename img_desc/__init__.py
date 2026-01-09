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
    batch = models.IntegerField()        # Exp column
    id_in_group = models.IntegerField()  # player id (0 is virtual)
    partner_id = models.IntegerField()   # paired partner id
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


def _clean_token(x):
    """Treat None/'None'/nan as empty."""
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in ("none", "nan", "null"):
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

    def get_previous_batch_row(self):
        """
        For interpreters paired with a real producer (partner_id != 0),
        get the corresponding producer row from (active_batch - 1).
        """
        if self.inner_role != INTERPRETER:
            return dict(sentences="[]")

        l = self.get_linked_batch()
        if not l:
            return dict(sentences="[]")

        # If paired with virtual producer 0, sentences are on current row.
        if l.get("partner_id", 0) == 0:
            return dict(sentences=l.get("sentences") or "[]")

        target_batch_idx = self.subsession.active_batch - 1
        if target_batch_idx < 0:
            return dict(sentences="[]")

        all_data = get_all_batches_sql(self.session.code)

        # Interpreter row has: id_in_group = interpreter_id, partner_id = producer_id
        interp_id = l.get("id_in_group")
        prod_id = l.get("partner_id")
        cond = l.get("condition")

        for obj in all_data:
            if (
                obj.get("batch") == target_batch_idx
                and obj.get("role") == PRODUCER
                and obj.get("id_in_group") == prod_id
                and obj.get("partner_id") == interp_id
                and obj.get("condition") == cond
            ):
                return obj

        return dict(sentences="[]")

    def get_sentences_data(self):
        l = self.get_linked_batch()
        if not l:
            return []

        try:
            # If interpreter paired with virtual producer (0): read from current row
            if l.get("partner_id", 0) == 0:
                return json.loads(l.get("sentences") or "[]")

            # Otherwise: read producer sentence from previous batch row
            prev = self.get_previous_batch_row()
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
        prefix = _clean_token(self.session.vars.get("prefix", ""))
        suffixes = self.session.vars.get("suffixes") or []

        sentences = self.get_sentences_data() or []

        # Expect sentences like: [["Some", "the A"], ["All", "the B"], ...]
        # Filter out unusable rows (empty, None, etc.)
        cleaned = []
        for sub in sentences:
            if not isinstance(sub, list):
                continue
            tokens = [_clean_token(x) for x in sub]
            if any(t == "" for t in tokens):
                continue
            cleaned.append(tokens)

        res = []
        for sentence in cleaned:
            expansion = []
            for tok, suf in zip(sentence, suffixes):
                tok2 = _clean_token(tok)
                suf2 = _clean_token(suf)
                expansion.append(tok2)
                if suf2:
                    expansion.append(suf2)

            if prefix:
                expansion.insert(0, prefix)

            # collapse whitespace nicely
            final = " ".join([x for x in expansion if x])
            if final:
                res.append(final)

        return res

    def get_image_url(self):
        l = self.get_linked_batch()
        if not l:
            return ""

        image_name = l.get("image")

        # Virtual producer lines (id_in_group==0) can have NA/NA_x: never show an image
        if l.get("id_in_group") == 0:
            return ""

        # Guard NA/empty
        if image_name is None:
            return ""
        image_name = str(image_name).strip()
        if not image_name or image_name.lower() in ("na", "na_x", "none", "nan"):
            return ""

        ext = (self.session.vars.get("extension") or "png").strip()
        if not image_name.lower().endswith(f".{ext}"):
            image_name = f"{image_name}.{ext}"

        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        if not base:
            return ""

        # Keep this only if you truly need it:
        if "amazonaws.com" in base:
            base = base.replace("/practice", "")

        return f"{base}/{image_name}"

    def start(self):
        session = self.session
        subsession = self.subsession
        all_data = get_all_batches_sql(session.code)

        # Round 1: assign a "line" (id_in_group) for the currently active Exp (batch)
        if self.round_number == 1:
            candidates = [
                b for b in all_data
                if b.get("batch") == subsession.active_batch
                and not b.get("busy")
                and (b.get("owner_code") or "") == ""
                and int(b.get("id_in_group") or 0) != 0  # IMPORTANT: never assign virtual player 0
            ]

            if not candidates:
                self.faulty = True
                return

            # deterministic: lowest id_in_group
            candidates.sort(key=lambda b: int(b.get("id_in_group") or 999999))
            free = candidates[0]

            chosen_batch = free["batch"]
            chosen_id = free["id_in_group"]

            self.batch = chosen_batch

            # mark all rows for this (batch, id_in_group) as busy for this participant
            for b in all_data:
                if b.get("batch") == chosen_batch and b.get("id_in_group") == chosen_id:
                    sql_update_batch(
                        b["id"],
                        busy=True,
                        owner_code=self.participant.code
                    )

            all_data = get_all_batches_sql(session.code)

        # Find this participant's row for this round
        my_row = None
        for b in all_data:
            if (
                b.get("owner_code") == self.participant.code
                and b.get("round_number") == self.round_number
            ):
                my_row = b
                break

        if not my_row:
            self.faulty = True
            return

        self.link_id = my_row["id"]
        self.inner_role = my_row["role"]
        self.inner_sentences = json.dumps(self.get_sentences_data())

        # Prolific label handling (unchanged)
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

    # Start at Exp 1 (per Anton’s rule)
    subsession.active_batch = 1
    session.vars["active_batch"] = 1

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Missing filename")

    from reading_xls.get_data import get_data
    excel_data = get_data(filename)
    df = excel_data.get("data")
    session.vars["user_data"] = df

    max_round = int(df["group_enumeration"].max())
    logger.info(f"img_desc: max group_enumeration in data = {max_round}")

    # Load rows into DB
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
    clean_settings = {}
    for k, v in settings.items():
        clean_settings[normalize_key(k)] = v
    session.vars["user_settings"] = clean_settings

    # Instructions URL
    default_url = "https://docs.google.com/document/d/e/2PACX-1vTg_Hd8hXK-TZS77rC6W_BlY2NtWhQqCLzlgW0LeomoEUdhoDNYPNVOO7Pt6g0-JksykUrgRdtcVL3u/pub?embedded=true"
    url_from_settings = clean_settings.get("instructions_url")
    session.vars["instructions_url"] = url_from_settings if url_from_settings else default_url

    # Standard keys
    for k in ["s3path_base", "extension", "prefix", "interpreter_choices", "interpreter_title"]:
        session.vars[k] = clean_settings.get(normalize_key(k))

    session.vars["suffixes"] = clean_settings.get("suffixes") or []

    # Allowed values and allowed regexes (up to 5 fields)
    allowed_values = []
    allowed_regexes = []
    for i in range(1, 6):
        v_key = f"allowed_values_{i}"
        r_key = f"allowed_regex_{i}"

        v = clean_settings.get(v_key)
        r = clean_settings.get(r_key)

        if v:
            allowed_values.append([x.strip() for x in str(v).split(";") if x.strip()])
        else:
            allowed_values.append([])

        allowed_regexes.append(str(r).strip() if r else "")

    session.vars["allowed_values"] = allowed_values
    session.vars["allowed_regexes"] = allowed_regexes


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
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regexes=player.session.vars.get("allowed_regexes", []),
            suffixes=player.session.vars.get("suffixes", []),
            interpreter_choices=interpreter_choices,
            interpreter_title=interpreter_title,
            instructions_url=player.session.vars.get("instructions_url"),
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
