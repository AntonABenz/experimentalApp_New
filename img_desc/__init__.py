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
    import pandas as pd

    logger.info(f"CWD={os.getcwd()}")
    logger.info(f"filename passed in={filename}")

    # --- Resolve Excel path robustly ---
    candidates = [
        Path(filename),                               # absolute or relative
        Path("start/data") / filename,                # /app/start/data/<file>
        Path("/app/start/data") / filename,           # explicit for Heroku
        Path("data") / filename,                      # optional fallback
        Path("/app/data") / filename,                 # optional fallback
    ]

    xlsx_path = None
    for p in candidates:
        if p.exists():
            xlsx_path = p
            break

    if not xlsx_path:
        tried = ", ".join(str(p) for p in candidates)
        raise FileNotFoundError(f"Excel file not found: {filename}. Tried: {tried}")

    logger.info(f"Excel resolved to: {xlsx_path}")

    # --- Load workbook ---
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")

    # --- Detect data sheet + settings sheet ---
    sheet_names = xls.sheet_names
    sheet_names_l = [s.lower().strip() for s in sheet_names]

    data_sheet = sheet_names[0]
    settings_sheet = None

    for i, s in enumerate(sheet_names_l):
        if "setting" in s:
            settings_sheet = sheet_names[i]
        if s in {"data", "trials", "items"}:
            data_sheet = sheet_names[i]

    # --- Read DATA sheet (CRITICAL: preserve "None" as string) ---
    df = pd.read_excel(
        xlsx_path,
        sheet_name=data_sheet,
        dtype=str,
        keep_default_na=False,  # do NOT convert "None", "NA", etc.
        na_filter=False,        # do NOT auto-detect NA values
        engine="openpyxl",
    )

    # --- Read SETTINGS sheet (if present) ---
    settings = {}
    if settings_sheet:
        raw = pd.read_excel(
            xlsx_path,
            sheet_name=settings_sheet,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            engine="openpyxl",
        )

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
        session = self.session
        active_batch = self.active_batch or 1
        all_data = get_all_batches_sql(session.code)
    
        remaining = [
            b for b in all_data
            if b["batch"] == active_batch and not b["processed"]
        ]
        if remaining:
            return
    
        # just bump the value in this subsession and session.vars
        self.active_batch = active_batch + 1
        session.vars["active_batch"] = active_batch + 1

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
        session = self.session
        subsession = self.subsession
    
        # active_batch safe default
        if subsession.field_maybe_none("active_batch") is None:
            subsession.active_batch = 1
    
        all_data = get_all_batches_sql(session.code)
    
        # ---- allocate participant to a free id_in_group only ONCE (round 1) ----
        if self.round_number == 1:
            active_batch = subsession.active_batch or 1
    
            candidates = [
                b for b in all_data
                if b["batch"] == active_batch
                and not b["busy"]
                and (b.get("owner_code") or "") == ""
            ]
            if not candidates:
                self.faulty = True
                return
    
            candidates.sort(key=lambda b: b["id_in_group"])
            chosen = candidates[0]
    
            self.batch = chosen["batch"]
    
            # mark ALL rows for that participant id_in_group in this batch as busy/owned
            for b in all_data:
                if b["batch"] == chosen["batch"] and b["id_in_group"] == chosen["id_in_group"]:
                    sql_update_batch(b["id"], busy=True, owner_code=self.participant.code)
    
            # refresh after updates
            all_data = get_all_batches_sql(session.code)
    
        # ---- find the row for this participant + round ----
        my_row = None
        for b in all_data:
            if (b.get("owner_code") == self.participant.code) and (int(b.get("round_number") or 0) == int(self.round_number)):
                my_row = b
                break
    
        if not my_row:
            self.faulty = True
            return
    
        # link + role/sentences
        self.link_id = my_row["id"]
        self.inner_role = my_row["role"] or ""
        self.inner_sentences = json.dumps(self.get_sentences_data())

def _force_str(val) -> str:
    """
    Turn cell value into string safely.
    - Keep literal 'None' as 'None'
    - Convert real None -> ''
    - Convert pandas-ish nan -> ''
    """
    if val is None:
        return ""
    s = str(val).strip()
    if s.lower() == "nan":
        return ""
    return s


def _to_int(val, default=0) -> int:
    if val is None:
        return default
    s = str(val).strip()
    if s == "" or s.lower() == "nan":
        return default
    try:
        return int(s)
    except ValueError:
        # handles "3.0" etc.
        try:
            return int(float(s))
        except Exception:
            return default


def _parse_bool(val) -> bool:
    s = _force_str(val).lower()
    return s in {"1", "true", "t", "yes", "y"}


def _collect_indexed(settings: dict, prefix: str, max_n: int = 20):
    """
    Collect keys like prefix_1, prefix_2,... in order.
    Returns a list of strings.
    """
    out = []
    for i in range(1, max_n + 1):
        v = settings.get(f"{prefix}_{i}")
        if v is None:
            continue
        vv = _force_str(v)
        if vv != "":
            out.append(vv)
    return out


def creating_session(subsession: Subsession):
    session = subsession.session

    # ensure subsession.active_batch always has a value
    if subsession.field_maybe_none("active_batch") is None:
        subsession.active_batch = 1

    # only do heavy init once
    if subsession.round_number != 1:
        return

    subsession.active_batch = 1
    session.vars["active_batch"] = 1

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Missing session.config['filename']")

    # ---- read excel robustly (uses /app/start/data fallback) ----
    df_raw, settings_raw = _read_excel_strict(filename)

    # ---- normalize settings keys ----
    clean_settings = {}
    for k, v in (settings_raw or {}).items():
        clean_settings[normalize_key(k)] = v
    session.vars["user_settings"] = clean_settings

    # ---- core settings ----
    session.vars["s3path_base"] = _force_str(clean_settings.get("s3path_base"))
    session.vars["extension"] = _force_str(clean_settings.get("extension")) or "png"
    session.vars["prefix"] = _force_str(clean_settings.get("prefix"))
    session.vars["interpreter_title"] = _force_str(clean_settings.get("interpreter_title")) or "Buy medals:"
    session.vars["caseflag"] = _parse_bool(clean_settings.get("caseflag"))

    # interpreter choices can be either "a;b;c" or already list-like text
    raw_choices = clean_settings.get("interpreter_choices", "")
    if isinstance(raw_choices, str):
        session.vars["interpreter_choices"] = [x.strip() for x in raw_choices.split(";") if x.strip()]
    else:
        session.vars["interpreter_choices"] = []

    # suffixes are stored as suffix_1, suffix_2, ...
    session.vars["suffixes"] = _collect_indexed(clean_settings, "suffix", max_n=20)

    # allowed values / regexes aligned by field index
    allowed_values = []
    allowed_regexes = []
    for i in range(1, 21):
        v = clean_settings.get(f"allowed_values_{i}")
        r = clean_settings.get(f"allowed_regex_{i}")
        if v is None and r is None:
            # if both missing, continue (don’t early-break because sheets sometimes skip indices)
            continue
        v_str = _force_str(v)
        r_str = _force_str(r)
        allowed_values.append([x.strip() for x in v_str.split(";") if x.strip()] if v_str else [])
        allowed_regexes.append(r_str)
    session.vars["allowed_values"] = allowed_values
    session.vars["allowed_regexes"] = allowed_regexes

    # instructions url
    default_url = (
        "https://docs.google.com/document/d/e/"
        "2PACX-1vTg_Hd8hXK-TZS77rC6W_BlY2NtWhQqCLzlgW0LeomoEUdhoDNYPNVOO7Pt6g0-JksykUrgRdtcVL3u/"
        "pub?embedded=true"
    )
    session.vars["instructions_url"] = _force_str(clean_settings.get("instructions_url")) or default_url

    # optional: completion_code for prolific
    if session.config.get("completion_code"):
        session.vars["completion_code"] = str(session.config["completion_code"]).strip()

    # ---- build Batch rows ----
    # IMPORTANT: DO NOT store df in session.vars (memory)
    records = df_raw.to_dict(orient="records")

    for r in records:
        Batch.create(
            session_code=session.code,
            owner_code="",
            batch=_to_int(r.get("Exp"), 0),
            item_nr=_force_str(r.get("Item.Nr")),
            condition=_force_str(r.get("Condition")),
            image=_force_str(r.get("Item")),          # keeps literal "None"
            round_number=_to_int(r.get("group_enumeration"), 0),
            role=_force_str(r.get("role")),
            id_in_group=_to_int(r.get("id"), 0),
            partner_id=_to_int(r.get("partner_id"), 0),
            sentences=_force_str(r.get("sentences") or "[]"),
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
    def is_displayed(player):
        if player.round_number > Constants.num_rounds:
            return False

        # run start() once per round (guard with link_id)
        if not player.link_id:
            player.start()

        if player.faulty:
            return False

        # start timing only when page is actually shown and link is valid
        if player.start_decision_time == 0:
            player.start_decision_time = time.time()

        return True

    @staticmethod
    def get_form_fields(player):
        if player.inner_role == PRODUCER:
            return ["producer_decision"]
        if player.inner_role == INTERPRETER:
            return ["interpreter_decision"]
        return []

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.end_decision_time = time.time()

        if player.start_decision_time:
            player.decision_seconds = float(player.end_decision_time - player.start_decision_time)

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
