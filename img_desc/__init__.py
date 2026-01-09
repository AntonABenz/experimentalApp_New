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


def _to_int(x, default=0):
    try:
        if x is None:
            return default
        # handle pandas float-ish like 1.0
        return int(float(x))
    except Exception:
        return default


def _clean_str(x):
    if x is None:
        return ""
    if isinstance(x, float) and (str(x) == "nan"):
        return ""
    return str(x).strip()


class Subsession(BaseSubsession):
    active_batch = models.IntegerField()
    study_id = models.StringField()
    completion_code = models.StringField()
    full_return_url = models.StringField()

    @property
    def get_active_batch(self):
        all_data = get_all_batches_sql(self.session.code)
        return [b for b in all_data if _to_int(b["batch"]) == _to_int(self.active_batch)]

    def check_for_batch_completion(self):
        session = self.session
        active_batch = _to_int(self.active_batch)
        all_data = get_all_batches_sql(session.code)
        remaining = [
            b for b in all_data
            if _to_int(b["batch"]) == active_batch and not b["processed"]
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
            if _to_int(b["id"]) == _to_int(self.link_id):
                return b
        return None

    def get_previous_batch(self):
        """
        Implements Anton's rule:
        - If interpreter is paired with producer 0 => use current row sentences (predefined)
        - Else use Exp(active_batch-1) for the matching producer row with swapped (id_in_group, partner_id)
        """
        if self.inner_role != INTERPRETER:
            return dict(sentences="[]")

        l = self.get_linked_batch()
        if not l:
            return dict(sentences="[]")

        if _to_int(l.get("partner_id")) == 0:
            return dict(sentences="[]")

        target_batch_idx = _to_int(self.subsession.active_batch) - 1
        if target_batch_idx < 0:
            return dict(sentences="[]")

        all_data = get_all_batches_sql(self.session.code)

        # robust matching with int casts
        want_partner = _to_int(l.get("id_in_group"))
        want_id = _to_int(l.get("partner_id"))
        want_cond = _clean_str(l.get("condition"))

        for obj in all_data:
            if (
                _to_int(obj.get("batch")) == target_batch_idx
                and _clean_str(obj.get("role")) == PRODUCER
                and _to_int(obj.get("partner_id")) == want_partner
                and _to_int(obj.get("id_in_group")) == want_id
                and _clean_str(obj.get("condition")) == want_cond
            ):
                return obj

        return dict(sentences="[]")

    def get_sentences_data(self):
        l = self.get_linked_batch()
        if not l:
            return []

        try:
            # If current interpreter is paired with producer 0 => use predefined sentences from THIS row
            if _to_int(l.get("partner_id")) == 0:
                return json.loads(l.get("sentences") or "[]")

            # Else use previous-batch producer sentences
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
            if b.get("owner_code") == my_code:
                sql_update_batch(_to_int(b["id"]), processed=True)
        self.subsession.check_for_batch_completion()

    def get_full_sentences(self):
        prefix = _clean_str(self.session.vars.get("prefix", ""))
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []

        # Keep only list rows and drop blanks/None
        cleaned = []
        for sub in sentences:
            if not isinstance(sub, list):
                continue
            if any(x is None or str(x).strip() == "" for x in sub):
                continue
            cleaned.append([str(x) for x in sub])

        res = []
        for sentence in cleaned:
            expansion = [str(item) for pair in zip(sentence, suffixes) for item in pair]
            if prefix:
                expansion.insert(0, prefix)
            res.append(" ".join(expansion))
        return res

    def get_image_url(self):
        l = self.get_linked_batch()
        if not l:
            return ""
        image_name = _clean_str(l.get("image"))
        # never construct NA-based URLs
        if not image_name or image_name.lower().startswith("na"):
            return ""

        ext = _clean_str(self.session.vars.get("extension", "png")) or "png"
        if not image_name.lower().endswith(f".{ext}"):
            image_name = f"{image_name}.{ext}"

        base = _clean_str(self.session.vars.get("s3path_base", "")).rstrip("/")
        if "amazonaws.com" in base:
            base = base.replace("/practice", "")
        if not base:
            return ""
        return f"{base}/{image_name}"

    def start(self):
        session = self.session
        subsession = self.subsession
        all_data = get_all_batches_sql(session.code)

        if self.round_number == 1:
            # IMPORTANT: exclude virtual id_in_group == 0 rows from being assigned to real participants
            candidates = [
                b for b in all_data
                if _to_int(b.get("batch")) == _to_int(subsession.active_batch)
                and not b.get("busy")
                and _clean_str(b.get("owner_code")) == ""
                and _to_int(b.get("id_in_group")) != 0
            ]
            if not candidates:
                self.faulty = True
                return

            candidates.sort(key=lambda b: _to_int(b.get("id_in_group")))
            free = candidates[0]
            chosen_batch = _to_int(free["batch"])
            chosen_id = _to_int(free["id_in_group"])

            self.batch = chosen_batch

            for b in all_data:
                if _to_int(b.get("batch")) == chosen_batch and _to_int(b.get("id_in_group")) == chosen_id:
                    sql_update_batch(
                        _to_int(b["id"]),
                        busy=True,
                        owner_code=self.participant.code
                    )

            all_data = get_all_batches_sql(session.code)

        my_row = None
        for b in all_data:
            if (
                _clean_str(b.get("owner_code")) == self.participant.code
                and _to_int(b.get("round_number")) == _to_int(self.round_number)
            ):
                my_row = b
                break

        if not my_row:
            self.faulty = True
            return

        self.link_id = _to_int(my_row["id"])
        self.inner_role = _clean_str(my_row["role"])
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

    max_round = int(df["group_enumeration"].max())
    logger.info(f"img_desc: max group_enumeration in data = {max_round}")

    records = df.to_dict(orient="records")
    for r in records:
        Batch.create(
            session_code=session.code,
            owner_code="",
            batch=_to_int(r.get("Exp")),
            item_nr=_clean_str(r.get("Item.Nr")),
            condition=_clean_str(r.get("Condition")),
            image=_clean_str(r.get("Item")),
            round_number=_to_int(r.get("group_enumeration")),
            role=_clean_str(r.get("role")),
            id_in_group=_to_int(r.get("id")),
            partner_id=_to_int(r.get("partner_id")),
            sentences=_clean_str(r.get("sentences") or "[]") if r.get("sentences") is not None else "[]",
        )

    settings = excel_data.get("settings") or {}

    clean_settings = {}
    for k, v in settings.items():
        clean_settings[normalize_key(k)] = v
    session.vars["user_settings"] = clean_settings

    # Instructions URL
    default_url = "https://docs.google.com/document/d/e/2PACX-1vTg_Hd8hXK-TZS77rC6W_BlY2NtWhQqCLzlgW0LeomoEUdhoDNYPNVOO7Pt6g0-JksykUrgRdtcVL3u/pub?embedded=true"
    url_from_settings = clean_settings.get("instructions_url")
    session.vars["instructions_url"] = _clean_str(url_from_settings) if url_from_settings else default_url

    # defaults so template never crashes
    session.vars["s3path_base"] = _clean_str(clean_settings.get("s3path_base"))
    session.vars["extension"] = _clean_str(clean_settings.get("extension") or "png")
    session.vars["prefix"] = _clean_str(clean_settings.get("prefix") or "")
    session.vars["suffixes"] = clean_settings.get("suffixes") or []
    session.vars["interpreter_choices"] = clean_settings.get("interpreter_choices") or ""
    session.vars["interpreter_title"] = _clean_str(clean_settings.get("interpreter_title") or "Buy medals:")
    session.vars["caseflag"] = str(clean_settings.get("caseflag") or "").lower() in ["true", "1", "t", "yes", "y"]

    # allowed values & regexes: allowed_values_1..5 and allowed_regex_1..5
    allowed_values = []
    allowed_regexes = []
    for i in range(1, 6):
        vkey = f"allowed_values_{i}"
        rkey = f"allowed_regex_{i}"

        v = clean_settings.get(vkey)
        if v:
            allowed_values.append([x.strip() for x in str(v).split(";") if x.strip()])
        else:
            allowed_values.append([])

        rx = clean_settings.get(rkey)
        allowed_regexes.append(_clean_str(rx))

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
    instructions = True
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

        return dict(
            d=player.get_linked_batch(),
            prefix=player.session.vars.get("prefix", ""),
            caseflag=player.session.vars.get("caseflag", False),
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regexes=player.session.vars.get("allowed_regexes", []),
            suffixes=player.session.vars.get("suffixes", []),
            interpreter_choices=interpreter_choices,
            interpreter_title=player.session.vars.get("interpreter_title") or "Buy medals:",
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
