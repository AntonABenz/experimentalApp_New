# img_desc/__init__.py
from otree.api import *
import json
import logging
import re
import time
from django.shortcuts import redirect

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
    """True for pandas/float NaN-ish values without importing pandas."""
    try:
        return isinstance(x, float) and x != x
    except Exception:
        return False


def _force_str_keep_none_token(x) -> str:
    """
    Convert Excel cell to string safely.
    - Real missing (None / NaN) => "" (empty)
    - Literal "None" (any case) stays as "None"
    - Everything else -> stripped string
    """
    if x is None or _is_nan_like(x):
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    # IMPORTANT: keep literal None token as a string token
    if s.lower() == "none":
        return "None"
    return s


def _ensure_json_list(s: str) -> str:
    """
    Ensure a valid JSON list string.
    - empty => "[]"
    - already JSON list => as-is
    - otherwise => "[]"
    """
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
    """
    Batch.sentences is expected to be JSON list-of-lists:
      [["None","the A or the B"], ["All","the A or the B"], ["None","the C"]]
    This parser keeps "None" as token and DOES NOT drop whole rows because of empty tokens.
    """
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
            # drop truly empty tokens, but keep literal "None"
            if tok_s == "":
                continue
            cleaned.append(tok_s)
        if cleaned:
            out.append(cleaned)
    return out


# ----------------------------
# Constants / Models
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


class Batch(ExtraModel):
    # NOTE: ExtraModel supports .objects like normal models
    session_code = models.StringField()
    owner_code = models.StringField(blank=True)

    # producer output / derived sentences
    sentences = models.LongStringField(blank=True)
    # interpreter output
    rewards = models.LongStringField(blank=True)

    condition = models.StringField(blank=True)
    item_nr = models.StringField(blank=True)
    image = models.StringField(blank=True)

    round_number = models.IntegerField()
    role = models.StringField()
    batch = models.IntegerField()

    id_in_group = models.IntegerField()
    partner_id = models.IntegerField()

    busy = models.BooleanField(initial=False)
    processed = models.BooleanField(initial=False)


class Subsession(BaseSubsession):
    active_batch = models.IntegerField(initial=1)
    study_id = models.StringField(blank=True)
    completion_code = models.StringField(blank=True)
    full_return_url = models.StringField(blank=True)

    def check_for_batch_completion(self):
        """
        If ALL rows in active_batch are processed => increment active_batch.
        Uses ORM (fast enough and avoids SQLAlchemy / sqlite table mismatch).
        """
        active_batch = self.field_maybe_none("active_batch") or 1
        remaining = Batch.objects.filter(
            session_code=self.session.code,
            batch=active_batch,
            processed=False,
        ).exists()
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

    # IMPORTANT: these are overwritten per-round (so they don’t stay “round 1” forever)
    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)

    link_id = models.IntegerField(initial=0)

    def role(self):
        return self.inner_role

    # ----------------------------
    # Linking
    # ----------------------------
    def get_linked_batch(self):
        if not self.link_id:
            return None
        return Batch.objects.filter(id=self.link_id).values().first()

    def get_previous_batch(self):
        """
        Interpreters read producer sentences from previous batch.
        If not interpreter => empty.
        """
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

        # Match producer row in previous batch with swapped ids and same condition
        prev = (
            Batch.objects.filter(
                session_code=self.session.code,
                batch=target_batch,
                role=PRODUCER,
                partner_id=l["id_in_group"],
                id_in_group=l["partner_id"],
                condition=l.get("condition", ""),
            )
            .values("sentences")
            .first()
        )
        return prev or {"sentences": "[]"}

    def get_sentences_data(self):
        l = self.get_linked_batch()
        if not l:
            return []

        # Producer sees its own saved sentences
        if self.inner_role == PRODUCER:
            return _parse_tokens_list(l.get("sentences") or "[]")

        # Interpreter: if no partner => maybe see own? otherwise previous producer batch
        if l.get("partner_id") in (0, None):
            return _parse_tokens_list(l.get("sentences") or "[]")

        prev = self.get_previous_batch()
        return _parse_tokens_list(prev.get("sentences") or "[]")

    # ----------------------------
    # Writing back
    # ----------------------------
    def update_batch(self):
        if not self.link_id:
            return
        if self.inner_role == PRODUCER:
            Batch.objects.filter(id=self.link_id).update(sentences=_ensure_json_list(self.producer_decision))
        elif self.inner_role == INTERPRETER:
            Batch.objects.filter(id=self.link_id).update(rewards=self.interpreter_decision)

    def mark_data_processed(self):
        """
        Mark all rows owned by this participant as processed.
        Then advance active_batch if possible.
        """
        my_code = self.participant.code
        Batch.objects.filter(session_code=self.session.code, owner_code=my_code).update(processed=True)
        self.subsession.check_for_batch_completion()

    # ----------------------------
    # Sentence expansion (fixes "None" shifting bug)
    # ----------------------------
    def get_full_sentences(self):
        prefix = self.session.vars.get("prefix") or ""
        suffixes = self.session.vars.get("suffixes") or []

        sentences = self.get_sentences_data() or []

        res = []
        for sentence_tokens in sentences:
            # zip truncates safely if sentence shorter than suffixes
            expansion = [str(item) for pair in zip(sentence_tokens, suffixes) for item in pair]
            if prefix:
                expansion.insert(0, prefix)
            res.append(" ".join(expansion))
        return res

    # ----------------------------
    # Image URL (avoid bad values, don't try to “fix” NA_x to random file)
    # ----------------------------
    def get_image_url(self):
        l = self.get_linked_batch()
        if not l:
            return ""

        image_name = _force_str_keep_none_token(l.get("image"))

        # If Excel has NA_x / NA / nan / empty => no image (do NOT generate broken URL)
        if not image_name or image_name.lower() in {"na", "na_x", "nan", "x"}:
            return ""

        # Extension
        ext = self.session.vars.get("extension") or "png"
        if not image_name.lower().endswith(f".{ext}"):
            image_name = f"{image_name}.{ext}"

        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        if not base:
            return ""

        # Remove /practice if present (your earlier logic)
        if "amazonaws.com" in base:
            base = base.replace("/practice", "")

        return f"{base}/{image_name}"

    # ----------------------------
    # Allocation per participant
    # ----------------------------
    def start(self):
        session = self.session
        subsession = self.subsession

        # Ensure active_batch always exists
        if subsession.field_maybe_none("active_batch") is None:
            subsession.active_batch = 1
            subsession.save()
        session.vars["active_batch"] = subsession.active_batch

        # Round 1: claim a free “id_in_group” inside active_batch for this participant
        if self.round_number == 1:
            free = (
                Batch.objects.filter(
                    session_code=session.code,
                    batch=subsession.active_batch,
                    busy=False,
                    owner_code="",
                )
                .order_by("id_in_group", "id")
                .values("id_in_group")
                .first()
            )
            if not free:
                self.faulty = True
                return

            chosen_id = free["id_in_group"]
            # Mark BOTH roles for that id_in_group as busy + owned
            Batch.objects.filter(
                session_code=session.code,
                batch=subsession.active_batch,
                id_in_group=chosen_id,
            ).update(busy=True, owner_code=self.participant.code)

            self.batch = subsession.active_batch

        # Find my row for this round
        my_row = (
            Batch.objects.filter(
                session_code=session.code,
                owner_code=self.participant.code,
                round_number=self.round_number,
            )
            .order_by("id")
            .values("id", "role")
            .first()
        )
        if not my_row:
            self.faulty = True
            return

        self.link_id = my_row["id"]
        self.inner_role = my_row["role"]
        self.inner_sentences = json.dumps(self.get_sentences_data())

        # Optional: keep Prolific label
        if self.round_number == 1 and session.config.get("for_prolific"):
            vars_ = self.participant.vars
            if vars_.get("prolific_session_id"):
                self.participant.label = vars_.get("prolific_session_id")


# ----------------------------
# Session creation
# ----------------------------
def creating_session(subsession: Subsession):
    session = subsession.session

    # Always initialize active_batch
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

    # Force key columns to strings (fix: "None" token preserved)
    # Also protects against encoding surprises.
    for col in ["Condition", "Item.Nr", "Item", "role", "sentences"]:
        if col in df.columns:
            df[col] = df[col].apply(_force_str_keep_none_token)

    # If you have Sentence_1..Sentence_5 columns and build sentences elsewhere,
    # ALSO force them to string here:
    for i in range(1, 6):
        c = f"Sentence_{i}"
        if c in df.columns:
            df[c] = df[c].apply(_force_str_keep_none_token)

    # Clear any old Batch rows for this session (important when re-creating session in dev)
    Batch.objects.filter(session_code=session.code).delete()

    records = df.to_dict(orient="records")
    for r in records:
        Batch.create(
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

    # Settings
    settings = excel_data.get("settings") or {}
    clean_settings = {normalize_key(k): v for k, v in settings.items()}

    # Core vars
    session.vars["s3path_base"] = _force_str_keep_none_token(clean_settings.get("s3path_base"))
    session.vars["extension"] = _force_str_keep_none_token(clean_settings.get("extension")) or "png"
    session.vars["prefix"] = _force_str_keep_none_token(clean_settings.get("prefix"))
    session.vars["interpreter_choices"] = clean_settings.get("interpreter_choices") or ""
    session.vars["interpreter_title"] = _force_str_keep_none_token(clean_settings.get("interpreter_title")) or "Buy medals:"

    # suffixes: accept either "suffixes" as list OR suffix_1..suffix_5 keys
    suffixes = []
    if isinstance(clean_settings.get("suffixes"), list):
        suffixes = [str(x) for x in clean_settings.get("suffixes") if str(x).strip()]
    else:
        for i in range(1, 11):
            s = _force_str_keep_none_token(clean_settings.get(f"suffix_{i}"))
            if s:
                suffixes.append(s)
    session.vars["suffixes"] = suffixes

    # allowed values / regexes
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

    # caseflag for template
    caseflag_raw = _force_str_keep_none_token(clean_settings.get("caseflag"))
    session.vars["caseflag"] = caseflag_raw.strip().lower() in {"true", "1", "t", "yes", "y"}

    # instructions url
    default_url = (
        "https://docs.google.com/document/d/e/"
        "2PACX-1vTg_Hd8hXK-TZS77rC6W_BlY2NtWhQqCLzlgW0LeomoEUdhoDNYPNVOO7Pt6g0-JksykUrgRdtcVL3u/"
        "pub?embedded=true"
    )
    url_from_settings = _force_str_keep_none_token(clean_settings.get("instructions_url"))
    session.vars["instructions_url"] = url_from_settings or default_url

    # Prolific completion code (so FinalForProlific never needs cnn.com fallback)
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

        # IMPORTANT FIX:
        # set start time EVERY time the page is displayed (per round),
        # otherwise only round 1 gets a timestamp.
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
        # Use saved URL if present
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


# ----------------------------
# Export (FIX: low-memory + exactly 80 rounds per participant)
# ----------------------------
def custom_export(players):
    """
    Output exactly one row per participant per round (1..Constants.num_rounds),
    including:
      - participant decisions + Batch values (condition/item/image/sentences/rewards)
      - decision_seconds (per round)

    This is streaming + iterator-based to avoid Heroku R14 memory crashes.
    """
    yield [
        "session_code",
        "participant_code",
        "round_number",
        "role",
        "condition",
        "item_nr",
        "image",
        "producer_sentences",
        "interpreter_rewards",
        "decision_seconds",
    ]

    # Accept QuerySet or list
    if hasattr(players, "values_list"):
        pq = (
            players.filter(round_number__gte=1, round_number__lte=Constants.num_rounds)
            .values_list(
                "session__code",
                "participant__code",
                "round_number",
                "inner_role",
                "producer_decision",
                "interpreter_decision",
                "decision_seconds",
            )
            .order_by("session__code", "participant__code", "round_number")
            .iterator(chunk_size=2000)
        )
    else:
        # fallback for list input (rare)
        def gen():
            for p in players:
                if 1 <= p.round_number <= Constants.num_rounds:
                    yield (
                        p.session.code,
                        p.participant.code,
                        p.round_number,
                        p.inner_role,
                        p.producer_decision,
                        p.interpreter_decision,
                        p.decision_seconds,
                    )
        pq = sorted(gen(), key=lambda r: (r[0], r[1], r[2]))

    # Group by (session_code, participant_code) without holding everything in RAM
    from itertools import groupby

    def keyfunc(row):
        return (row[0], row[1])

    for (session_code, participant_code), rows_iter in groupby(pq, key=keyfunc):
        # Build a small map of Batch rows for this participant (max ~80)
        bq = (
            Batch.objects.filter(session_code=session_code, owner_code=participant_code)
            .values_list("round_number", "role", "condition", "item_nr", "image", "sentences", "rewards")
            .iterator(chunk_size=200)
        )
        batch_map = {int(rn): (role, cond, item_nr, img, sent, rew) for rn, role, cond, item_nr, img, sent, rew in bq}

        for row in rows_iter:
            _, _, rnd, inner_role, prod_dec, interp_dec, ds = row
            rnd = int(rnd)
            if rnd < 1 or rnd > Constants.num_rounds:
                continue

            b = batch_map.get(rnd)
            if b:
                role, cond, item_nr, img, sent, rew = b
                producer_sentences = sent or ""
                interpreter_rewards = rew or ""
            else:
                # fall back to Player fields if Batch missing (should be rare)
                role = inner_role or ""
                cond = ""
                item_nr = ""
                img = ""
                producer_sentences = prod_dec or ""
                interpreter_rewards = interp_dec or ""

            yield [
                session_code,
                participant_code,
                rnd,
                role,
                cond,
                item_nr,
                img,
                producer_sentences,
                interpreter_rewards,
                "" if ds is None else ds,
            ]


page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
