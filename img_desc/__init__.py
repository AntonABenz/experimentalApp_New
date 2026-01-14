from otree.api import *
import json
import logging
import os
import re
import time

import pandas as pd
from django.shortcuts import redirect

logger = logging.getLogger("benzapp.img_desc")

PRODUCER = "P"
INTERPRETER = "I"

STUBURL = "https://app.prolific.co/submissions/complete?cc="


class Constants(BaseConstants):
    name_in_url = "img_desc"
    players_per_group = None
    num_rounds = 80

    PLACEMENT_ERR = "ERROR_BATCH_PLACEMENT"
    API_ERR = "API_ERROR"
    FALLBACK_URL = STUBURL + PLACEMENT_ERR
    API_ERR_URL = STUBURL + API_ERR


class Batch(ExtraModel):
    session_code = models.StringField()
    owner_code = models.StringField(blank=True)

    # payload from Excel:
    batch = models.IntegerField()
    round_number = models.IntegerField()
    role = models.StringField()
    id_in_group = models.IntegerField()
    partner_id = models.IntegerField()

    condition = models.StringField()
    item_nr = models.StringField()
    image = models.StringField()

    sentences = models.LongStringField()
    rewards = models.LongStringField(blank=True)

    busy = models.BooleanField(initial=False)
    processed = models.BooleanField(initial=False)


def normalize_key(key):
    if not key:
        return ""
    return re.sub(r"[\s_]+", "_", str(key).lower().strip())


def _truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_excel_strict(filename):
    import os
    from pathlib import Path

    logger.info(f"CWD={os.getcwd()}")
    logger.info(f"filename passed in={filename}")

    logger.info(f"Exists in CWD? {Path(filename).exists()}")
    logger.info(f"Exists in CWD by name? {Path('disjunctionExpTest8EN_1.xlsx').exists()}")

    logger.info(f"Exists start/data? {Path('start/data/disjunctionExpTest8EN_1.xlsx').exists()}")
    logger.info(f"Exists /app/start/data? {Path('/app/start/data/disjunctionExpTest8EN_1.xlsx').exists()}")
    logger.info(f"Exists /app/data? {Path('/app/data/disjunctionExpTest8EN_1.xlsx').exists()}")

    xls = pd.ExcelFile(filename)

    # Heuristics: find data sheet + settings sheet
    sheet_names = [s.lower() for s in xls.sheet_names]
    data_sheet = xls.sheet_names[0]
    settings_sheet = None

    for i, s in enumerate(sheet_names):
        if "setting" in s:
            settings_sheet = xls.sheet_names[i]
        if s in {"data", "trials", "items"}:
            data_sheet = xls.sheet_names[i]

    df = pd.read_excel(
        filename,
        sheet_name=data_sheet,
        dtype=str,                 # read as strings first
        keep_default_na=False,     # <-- THIS is the big fix for "None"
        na_filter=False,           # don't auto-detect NA strings
        engine="openpyxl",
    )

    settings = {}
    if settings_sheet:
        raw = pd.read_excel(
            filename,
            sheet_name=settings_sheet,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            engine="openpyxl",
        )
        # support either "key/value" columns or a 2-col sheet
        cols = [c.lower().strip() for c in raw.columns]
        if len(raw.columns) >= 2:
            kcol, vcol = raw.columns[0], raw.columns[1]
            for _, row in raw.iterrows():
                k = str(row.get(kcol, "")).strip()
                v = str(row.get(vcol, "")).strip()
                if k:
                    settings[k] = v

    return df, settings


def _safe_int(x, default=0):
    try:
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _safe_str(x):
    # DO NOT collapse "None" -> "", that is your main bug.
    if x is None:
        return ""
    return str(x).strip()


class Subsession(BaseSubsession):
    # Keep it simple: persist the active batch in session.vars,
    # not via Subsession.objects updates.
    active_batch = models.IntegerField(initial=1)

    def get_active_batch_number(self) -> int:
        return int(self.session.vars.get("active_batch", 1))

    def set_active_batch_number(self, n: int) -> None:
        self.session.vars["active_batch"] = int(n)
        self.active_batch = int(n)

    def check_for_batch_completion(self):
        active_batch = self.get_active_batch_number()
        remaining = Batch.filter(
            session_code=self.session.code,
            batch=active_batch,
            processed=False,
        )
        if remaining:
            return
        self.set_active_batch_number(active_batch + 1)


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    inner_role = models.StringField()
    link_id = models.IntegerField(initial=0)

    faulty = models.BooleanField(initial=False)
    feedback = models.LongStringField(label="")

    # decisions
    producer_decision = models.LongStringField()
    interpreter_decision = models.LongStringField()

    # timing
    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)

    # prolific
    full_return_url = models.StringField(blank=True)

    def role(self):
        return self.inner_role

    def _linked_batch(self):
        if not self.link_id:
            return None
        rows = Batch.filter(id=self.link_id)
        return rows[0] if rows else None

    def get_linked_batch(self):
        b = self._linked_batch()
        if not b:
            return None
        # keep template compatibility: return dict-like
        return dict(
            id=b.id,
            session_code=b.session_code,
            owner_code=b.owner_code,
            sentences=b.sentences,
            rewards=b.rewards,
            condition=b.condition,
            item_nr=b.item_nr,
            image=b.image,
            round_number=b.round_number,
            role=b.role,
            batch=b.batch,
            id_in_group=b.id_in_group,
            partner_id=b.partner_id,
            busy=b.busy,
            processed=b.processed,
        )

    def get_sentences_data(self):
        b = self._linked_batch()
        if not b:
            return []
        raw = b.sentences or "[]"
        try:
            return json.loads(raw)
        except Exception:
            return []

    def update_batch(self):
        b = self._linked_batch()
        if not b:
            return
        if self.inner_role == PRODUCER:
            b.sentences = self.producer_decision or "[]"
        elif self.inner_role == INTERPRETER:
            b.rewards = self.interpreter_decision or ""
        b.save()

    def mark_data_processed(self):
        # mark all this participant's rows as processed
        rows = Batch.filter(session_code=self.session.code, owner_code=self.participant.code)
        for r in rows:
            r.processed = True
            r.save()
        self.subsession.check_for_batch_completion()

    def get_image_url(self):
        b = self._linked_batch()
        if not b:
            return ""

        image_name = _safe_str(b.image)
        if image_name == "":
            return ""

        # Do NOT treat "None" as missing unless you truly want no image:
        # If "None" is a valid image name in your design, remove this block.
        if image_name.lower() in {"nan"}:
            return ""

        image_name = image_name.replace(" ", "_")

        ext = (self.session.vars.get("extension") or "png").strip()
        if ext and not image_name.lower().endswith("." + ext.lower()):
            image_name = f"{image_name}.{ext}"

        base = (_safe_str(self.session.vars.get("s3path_base"))).rstrip("/")
        if not base:
            return ""
        if "amazonaws.com" in base:
            base = base.replace("/practice", "")

        return f"{base}/{image_name}"

    def start(self):
        """
        Assign the participant to one id_in_group in the current active batch (round 1),
        then find the row for each subsequent round.
        """
        session_code = self.session.code
        active_batch = self.subsession.get_active_batch_number()

        # Round 1: claim an unowned id_in_group within active_batch
        if self.round_number == 1:
            candidate = (
                Batch.objects
                .filter(session_code=session_code, batch=active_batch, busy=False, owner_code="")
                .order_by("id_in_group")
                .first()
            )
            if not candidate:
                self.faulty = True
                return

            chosen_id = candidate.id_in_group

            # claim ALL rows for that id_in_group in this batch
            rows = Batch.filter(session_code=session_code, batch=active_batch, id_in_group=chosen_id)
            if not rows:
                self.faulty = True
                return

            for r in rows:
                r.busy = True
                r.owner_code = self.participant.code
                r.save()

        # Find this participant's row for this round
        my_row = (
            Batch.objects
            .filter(session_code=session_code, owner_code=self.participant.code, round_number=self.round_number)
            .first()
        )
        if not my_row:
            self.faulty = True
            return

        self.link_id = my_row.id
        self.inner_role = my_row.role


def creating_session(subsession: Subsession):
    session = subsession.session

    # Ensure active_batch is set (in vars)
    subsession.set_active_batch_number(session.vars.get("active_batch", 1))

    if subsession.round_number != 1:
        return

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Missing filename in session config")

    df, settings = _read_excel_strict(filename)

    # IMPORTANT:
    # If you replaced the Excel and want new content, you MUST clear old rows for this session
    Batch.objects.filter(session_code=session.code).delete()

    # Normalize + store settings
    clean_settings = {normalize_key(k): v for k, v in (settings or {}).items()}
    session.vars["user_settings"] = clean_settings

    # core settings
    session.vars["s3path_base"] = clean_settings.get("s3path_base", "")
    session.vars["extension"] = clean_settings.get("extension", "png")
    session.vars["prefix"] = clean_settings.get("prefix", "")
    session.vars["caseflag"] = _truthy(clean_settings.get("caseflag", "false"))

    # suffixes: suffix_1, suffix_2, ...
    suffixes = []
    i = 1
    while True:
        key = f"suffix_{i}"
        v = clean_settings.get(key)
        if v is None or str(v).strip() == "":
            break
        suffixes.append(str(v).strip())
        i += 1
        if i > 20:
            break
    session.vars["suffixes"] = suffixes

    # interpreter choices + title
    session.vars["interpreter_choices"] = clean_settings.get("interpreter_choices", "")
    session.vars["interpreter_title"] = clean_settings.get("interpreter_title", "Buy medals:")

    # instructions url
    default_url = "https://docs.google.com/document/d/e/2PACX-1vTg_Hd8hXK-TZS77rC6W_BlY2NtWhQqCLzlgW0LeomoEUdhoDNYPNVOO7Pt6g0-JksykUrgRdtcVL3u/pub?embedded=true"
    session.vars["instructions_url"] = clean_settings.get("instructions_url") or default_url

    # allowed values/regexes aligned by index
    allowed_values = []
    allowed_regexes = []
    for i in range(1, 11):
        v = clean_settings.get(f"allowed_values_{i}", "")
        r = clean_settings.get(f"allowed_regex_{i}", "")
        if str(v).strip() == "" and str(r).strip() == "":
            continue
        allowed_values.append([x.strip() for x in str(v).split(";") if x.strip()])
        allowed_regexes.append(str(r).strip())
    session.vars["allowed_values"] = allowed_values
    session.vars["allowed_regexes"] = allowed_regexes

    # Build Batch rows from df
    # Expect columns like: Exp, group_enumeration, role, id, partner_id, Condition, Item.Nr, Item, sentences
    for _, r in df.iterrows():
        Batch.create(
            session_code=session.code,
            owner_code="",
            batch=_safe_int(r.get("Exp"), 0),
            round_number=_safe_int(r.get("group_enumeration"), 0),
            role=_safe_str(r.get("role")),
            id_in_group=_safe_int(r.get("id"), 0),
            partner_id=_safe_int(r.get("partner_id"), 0),
            condition=_safe_str(r.get("Condition")),
            item_nr=_safe_str(r.get("Item.Nr")),
            image=_safe_str(r.get("Item")),
            sentences=_safe_str(r.get("sentences")) or "[]",
            rewards="",
            busy=False,
            processed=False,
        )


class FaultyCatcher(Page):
    @staticmethod
    def is_displayed(player):
        return player.faulty

    def get(self):
        return redirect(Constants.FALLBACK_URL)


class Q(Page):
    form_model = "player"

    @staticmethod
    def is_displayed(player: Player):
        player.start()
        if player.faulty:
            return False

        # timing start on first render of each round
        if player.start_decision_time == 0:
            player.start_decision_time = time.time()

        return player.round_number <= Constants.num_rounds

    @staticmethod
    def get_form_fields(player: Player):
        if player.inner_role == PRODUCER:
            return ["producer_decision"]
        if player.inner_role == INTERPRETER:
            return ["interpreter_decision"]
        return []

    @staticmethod
    def vars_for_template(player: Player):
        raw_choices = player.session.vars.get("interpreter_choices") or ""
        if isinstance(raw_choices, str):
            interpreter_choices = [x.strip() for x in raw_choices.split(";") if x.strip()]
        else:
            interpreter_choices = list(raw_choices) if raw_choices else []

        return dict(
            d=player.get_linked_batch(),
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regexes=player.session.vars.get("allowed_regexes", []),
            suffixes=player.session.vars.get("suffixes", []),
            prefix=player.session.vars.get("prefix", ""),
            caseflag=player.session.vars.get("caseflag", False),
            interpreter_choices=interpreter_choices,
            interpreter_title=player.session.vars.get("interpreter_title", "Buy medals:"),
            instructions_url=player.session.vars.get("instructions_url"),
        )

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        player.end_decision_time = time.time()
        if player.start_decision_time:
            player.decision_seconds = player.end_decision_time - player.start_decision_time

        player.update_batch()

        if player.round_number == Constants.num_rounds:
            player.mark_data_processed()


class Feedback(Page):
    form_model = "player"
    form_fields = ["feedback"]

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == Constants.num_rounds


class FinalForProlific(Page):
    @staticmethod
    def is_displayed(player: Player):
        return bool(player.session.config.get("for_prolific")) and player.round_number == Constants.num_rounds

    def get(self):
        completion_code = (
            self.player.session.config.get("completion_code")
            or self.player.session.vars.get("completion_code")
        )
        if not completion_code:
            return redirect(Constants.API_ERR_URL)
        return redirect(STUBURL + str(completion_code))


def custom_export(players):
    """
    STREAMED export, one row per participant per round (1..80).
    No huge in-memory maps. Safe on Heroku.
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

    # Stream players (minimal fields)
    p_it = (
        players
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

    for session_code, participant_code, rnd, inner_role, prod_dec, interp_dec, ds in p_it:
        rnd = int(rnd)
        if rnd < 1 or rnd > Constants.num_rounds:
            continue

        b = (
            Batch.objects
            .filter(session_code=session_code, owner_code=participant_code, round_number=rnd)
            .values_list("role", "condition", "item_nr", "image", "sentences", "rewards")
            .first()
        )

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
