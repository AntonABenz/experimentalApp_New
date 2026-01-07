from otree.api import *
import json
import logging
import os
import re
from itertools import zip_longest
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


def _build_list_from_numbered_keys(clean_settings, prefix, max_n=10):
    """e.g. suffix_1, suffix_2 ..."""
    out = []
    for i in range(1, max_n + 1):
        v = clean_settings.get(f"{prefix}{i}")
        if v is None or (isinstance(v, float) and str(v) == "nan"):
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out


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
        Interpreter sometimes reads from previous active_batch (design choice).
        If partner mapping fails, returns empty sentences.
        """
        if self.inner_role != INTERPRETER:
            return dict(sentences="[]")
        l = self.get_linked_batch()
        if not l or l["partner_id"] == 0:
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
        l = self.get_linked_batch()
        if not l:
            return []
        try:
            raw = None
            if l["partner_id"] == 0:
                raw = l.get("sentences")
            else:
                raw = self.get_previous_batch().get("sentences")

            if raw in (None, "", "nan"):
                return []
            return json.loads(raw)
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
        prefix = self.session.vars.get("prefix", "") or ""
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []

        # keep only list rows without empty strings
        valid = []
        for sub in sentences:
            if not isinstance(sub, list):
                continue
            if any(str(x).strip() == "" for x in sub):
                continue
            valid.append([str(x).strip() for x in sub])

        res = []
        for sentence in valid:
            # interleave tokens + suffixes robustly
            pieces = []
            for token, suf in zip_longest(sentence, suffixes, fillvalue=""):
                if token:
                    pieces.append(token)
                if suf:
                    pieces.append(str(suf))
            if prefix:
                pieces.insert(0, prefix)
            res.append(" ".join(pieces).strip())
        return res

    def get_image_url(self):
        l = self.get_linked_batch()
        if not l:
            return ""
        image_name = l.get("image") or ""
        if not image_name:
            return ""

        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        if "amazonaws.com" in base:
            base = base.replace("/practice", "")

        ext = (self.session.vars.get("extension") or "png").strip().lstrip(".")
        # If image already has an extension, keep it
        if re.search(r"\.(png|jpg|jpeg|webp)$", image_name, flags=re.I):
            return f"{base}/{image_name}"
        return f"{base}/{image_name}.{ext}"

    def get_image_candidates(self):
        """Try multiple common extensions client-side if one fails."""
        l = self.get_linked_batch()
        if not l:
            return []
        image_name = l.get("image") or ""
        if not image_name:
            return []

        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        if "amazonaws.com" in base:
            base = base.replace("/practice", "")

        # If already has extension, just return it
        if re.search(r"\.(png|jpg|jpeg|webp)$", image_name, flags=re.I):
            return [f"{base}/{image_name}"]

        preferred = (self.session.vars.get("extension") or "png").strip().lstrip(".").lower()
        exts = [preferred, "png", "jpg", "jpeg", "webp"]
        # unique preserve order
        seen = set()
        ordered = []
        for e in exts:
            if e not in seen:
                seen.add(e)
                ordered.append(e)

        return [f"{base}/{image_name}.{e}" for e in ordered]

    def start(self):
        session = self.session
        subsession = self.subsession
        all_data = get_all_batches_sql(session.code)

        if self.round_number == 1:
            candidates = [
                b for b in all_data
                if b["batch"] == subsession.active_batch and not b["busy"] and b["owner_code"] == ""
            ]
            if not candidates:
                self.faulty = True
                return

            candidates.sort(key=lambda b: b["id_in_group"])
            free = candidates[0]
            chosen_batch = free["batch"]
            chosen_id = free["id_in_group"]
            self.batch = chosen_batch

            for b in all_data:
                if b["batch"] == chosen_batch and b["id_in_group"] == chosen_id:
                    sql_update_batch(b["id"], busy=True, owner_code=self.participant.code)

            all_data = get_all_batches_sql(session.code)

        my_row = None
        for b in all_data:
            if b["owner_code"] == self.participant.code and b["round_number"] == self.round_number:
                my_row = b
                break

        if not my_row:
            self.faulty = True
            return

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
        # Ensure sentences is a valid JSON string (seed data)
        seed = r.get("sentences")
        if seed in (None, "", "nan"):
            seed = "[]"

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
            sentences=seed,
        )

    settings = excel_data.get("settings") or {}

    clean_settings = {}
    for k, v in settings.items():
        clean_settings[normalize_key(k)] = v
    session.vars["user_settings"] = clean_settings

    # Instructions URL (always defined)
    default_url = "https://docs.google.com/document/d/e/2PACX-1vTg_Hd8hXK-TZS77rC6W_BlY2NtWhQqCLzlgW0LeomoEUdhoDNYPNVOO7Pt6g0-JksykUrgRdtcVL3u/pub?embedded=true"
    url_from_settings = clean_settings.get("instructions_url")
    session.vars["instructions_url"] = str(url_from_settings).strip() if url_from_settings else default_url

    # Basic settings
    for k in ["s3path_base", "extension", "prefix", "interpreter_choices", "interpreter_title"]:
        session.vars[k] = clean_settings.get(k)

    # Suffixes: accept either a single "suffixes" value or suffix_1..suffix_n
    suffixes = clean_settings.get("suffixes")
    if isinstance(suffixes, str) and suffixes.strip():
        # allow either ";" separated or JSON list string
        if suffixes.strip().startswith("["):
            try:
                session.vars["suffixes"] = json.loads(suffixes)
            except Exception:
                session.vars["suffixes"] = [x.strip() for x in suffixes.split(";") if x.strip()]
        else:
            session.vars["suffixes"] = [x.strip() for x in suffixes.split(";") if x.strip()]
    else:
        session.vars["suffixes"] = _build_list_from_numbered_keys(clean_settings, "suffix_", max_n=10)

    # Allowed values: allowed_values_1..5 are ";" separated
    allowed_values = []
    for i in range(1, 6):
        key = f"allowed_values_{i}"
        val = clean_settings.get(key)
        if val:
            allowed_values.append([x.strip() for x in str(val).split(";") if x.strip()])
        else:
            allowed_values.append([])
    session.vars["allowed_values"] = allowed_values

    # Allowed regexes: allowed_regex_1..5 (raw regex strings, no splitting)
    allowed_regexes = []
    for i in range(1, 6):
        key = f"allowed_regex_{i}"
        val = clean_settings.get(key)
        allowed_regexes.append(str(val).strip() if val else "")
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

        interpreter_title = player.session.vars.get("interpreter_title") or "Buy medals:"

        return dict(
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regexes=player.session.vars.get("allowed_regexes", []),
            suffixes=player.session.vars.get("suffixes", []),
            interpreter_choices=interpreter_choices,
            interpreter_title=interpreter_title,
            instructions_url=player.session.vars.get("instructions_url"),
            image_candidates=player.get_image_candidates(),
            has_interpreter_field=(player.inner_role == INTERPRETER),
        )

    @staticmethod
    def error_message(player, values):
        """
        Server-side validation so bad payloads never enter your DB.
        """
        role = player.field_maybe_none("inner_role")

        if role == PRODUCER:
            raw = values.get("producer_decision") or "[]"
            try:
                rows = json.loads(raw)
            except Exception:
                return "Invalid submission format. Please try again."

            # rows must be a list of list[str]
            if not isinstance(rows, list) or len(rows) == 0:
                return "Please write at least one complete sentence."

            # Validate completeness + regex
            regexes = player.session.vars.get("allowed_regexes") or []
            for row in rows:
                if not isinstance(row, list) or any(str(x).strip() == "" for x in row):
                    return "Please write only complete sentences (no blanks)."

                for j, token in enumerate(row):
                    rx = (regexes[j] if j < len(regexes) else "") or ""
                    if not rx:
                        continue
                    try:
                        if re.fullmatch(rx, str(token).strip(), flags=re.IGNORECASE) is None:
                            return "One of your entries does not match the allowed format. Use the (?) help icon to see allowed options."
                    except re.error:
                        # if regex is malformed, don't block experiment; but log it
                        logger.warning(f"Malformed regex for field {j+1}: {rx}")

        if role == INTERPRETER:
            # If there are no sentences, interpreter must not proceed
            if len(player.get_full_sentences()) == 0:
                return "No sentences were received for interpretation."

            raw = values.get("interpreter_decision") or "[]"
            try:
                rows = json.loads(raw)
            except Exception:
                return "Invalid submission format. Please try again."

            if not isinstance(rows, list) or any((r.get("answer") not in ("Yes", "No")) for r in rows if isinstance(r, dict)):
                return "Please answer Yes/No for every row."

        return None

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
