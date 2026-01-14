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


def normalize_key(key):
    if not key:
        return ""
    return re.sub(r"[\s_]+", "_", str(key).lower().strip())


def _truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def _read_excel_strict(filename):
    from pathlib import Path
    import pandas as pd

    logger.info(f"CWD={os.getcwd()}")
    logger.info(f"filename passed in={filename}")

    # --- Resolve Excel path robustly ---
    candidates = [
        Path(filename),
        Path("start/data") / filename,
        Path("/app/start/data") / filename,
        Path("data") / filename,
        Path("/app/data") / filename,
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

    # --- Read DATA sheet ---
    df = pd.read_excel(
        xlsx_path,
        sheet_name=data_sheet,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        engine="openpyxl",
    )

    # --- Read SETTINGS sheet ---
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
    if x is None:
        return ""
    return str(x).strip()


def _force_str(val) -> str:
    """Turn cell value into string safely."""
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
        try:
            return int(float(s))
        except Exception:
            return default


def _parse_bool(val) -> bool:
    s = _force_str(val).lower()
    return s in {"1", "true", "t", "yes", "y"}


def _collect_indexed(settings: dict, prefix: str, max_n: int = 20):
    """Collect keys like prefix_1, prefix_2,... in order."""
    out = []
    for i in range(1, max_n + 1):
        v = settings.get(f"{prefix}_{i}")
        if v is None:
            continue
        vv = _force_str(v)
        if vv != "":
            out.append(vv)
    return out


# ============================================================================
# BATCH DATA MANAGEMENT (using session.vars instead of ExtraModel)
# ============================================================================

def get_all_batches_from_session(session):
    """Get batch data from session.vars"""
    return session.vars.get('batches', [])


def update_batch_in_session(session, batch_id, **updates):
    """Update a batch in session.vars"""
    batches = session.vars.get('batches', [])
    for b in batches:
        if b.get('id') == batch_id:
            b.update(updates)
            break
    session.vars['batches'] = batches


# ============================================================================
# OTREE MODELS
# ============================================================================

class Subsession(BaseSubsession):
    active_batch = models.IntegerField(initial=1)

    def get_active_batch_number(self) -> int:
        return int(self.session.vars.get("active_batch", 1))

    def set_active_batch_number(self, n: int) -> None:
        self.session.vars["active_batch"] = int(n)
        self.active_batch = int(n)

    def check_for_batch_completion(self):
        session = self.session
        active_batch = self.active_batch or 1
        all_data = get_all_batches_from_session(session)
    
        remaining = [
            b for b in all_data
            if b["batch"] == active_batch and not b["processed"]
        ]
        if remaining:
            return
    
        # bump the value in this subsession and session.vars
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
        """Get the batch dict linked to this player"""
        if not self.link_id:
            return None
        batches = get_all_batches_from_session(self.session)
        for b in batches:
            if b.get('id') == self.link_id:
                return b
        return None

    def get_linked_batch(self):
        """Get linked batch for template compatibility"""
        return self._linked_batch()

    def get_sentences_data(self):
        b = self._linked_batch()
        if not b:
            return []
        raw = b.get('sentences', "[]")
        try:
            return json.loads(raw)
        except Exception:
            return []

    def update_batch(self):
        """Update batch data after player makes decision"""
        b = self._linked_batch()
        if not b:
            return
        
        updates = {}
        if self.inner_role == PRODUCER:
            updates['sentences'] = self.producer_decision or "[]"
        elif self.inner_role == INTERPRETER:
            updates['rewards'] = self.interpreter_decision or ""
        
        if updates:
            update_batch_in_session(self.session, self.link_id, **updates)

    def mark_data_processed(self):
        """Mark all this participant's rows as processed"""
        batches = get_all_batches_from_session(self.session)
        for b in batches:
            if b.get('owner_code') == self.participant.code:
                update_batch_in_session(self.session, b['id'], processed=True)
        
        self.subsession.check_for_batch_completion()

    def get_image_url(self):
        b = self._linked_batch()
        if not b:
            return ""

        image_name = _safe_str(b.get('image', ''))
        if image_name == "":
            return ""

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
        """Initialize player for this round"""
        session = self.session
        subsession = self.subsession
    
        # active_batch safe default
        if subsession.field_maybe_none("active_batch") is None:
            subsession.active_batch = 1
    
        all_data = get_all_batches_from_session(session)
    
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
                logger.error(f"No candidates found for participant {self.participant.code} in batch {active_batch}")
                self.faulty = True
                return
    
            candidates.sort(key=lambda b: b["id_in_group"])
            chosen = candidates[0]
    
            # mark ALL rows for that participant id_in_group in this batch as busy/owned
            for b in all_data:
                if b["batch"] == chosen["batch"] and b["id_in_group"] == chosen["id_in_group"]:
                    update_batch_in_session(session, b["id"], busy=True, owner_code=self.participant.code)
    
            logger.info(f"Assigned participant {self.participant.code} to id_in_group {chosen['id_in_group']} in batch {chosen['batch']}")
            
            # refresh after updates
            all_data = get_all_batches_from_session(session)
    
        # ---- find the row for this participant + round ----
        my_row = None
        for b in all_data:
            if (b.get("owner_code") == self.participant.code) and (int(b.get("round_number") or 0) == int(self.round_number)):
                my_row = b
                break
    
        if not my_row:
            logger.error(f"No batch row found for participant {self.participant.code}, round {self.round_number}")
            logger.error(f"Available batches for this participant: {[b for b in all_data if b.get('owner_code') == self.participant.code][:5]}")
            self.faulty = True
            return
    
        # link + role/sentences
        self.link_id = my_row["id"]
        self.inner_role = my_row["role"] or ""
        
        logger.info(f"Round {self.round_number}: Linked participant {self.participant.code} to batch_id {my_row['id']}, role {self.inner_role}")


# ============================================================================
# SESSION CREATION
# ============================================================================

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

    # ---- read excel robustly ----
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

    # interpreter choices
    raw_choices = clean_settings.get("interpreter_choices", "")
    if isinstance(raw_choices, str):
        session.vars["interpreter_choices"] = [x.strip() for x in raw_choices.split(";") if x.strip()]
    else:
        session.vars["interpreter_choices"] = []

    # suffixes
    session.vars["suffixes"] = _collect_indexed(clean_settings, "suffix", max_n=20)

    # allowed values / regexes
    allowed_values = []
    allowed_regexes = []
    for i in range(1, 21):
        v = clean_settings.get(f"allowed_values_{i}")
        r = clean_settings.get(f"allowed_regex_{i}")
        if v is None and r is None:
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

    # completion_code for prolific
    if session.config.get("completion_code"):
        session.vars["completion_code"] = str(session.config["completion_code"]).strip()

    # ---- build Batch data in session.vars ----
    records = df_raw.to_dict(orient="records")
    
    batches = []
    batch_id = 1
    for r in records:
        batches.append({
            'id': batch_id,
            'session_code': session.code,
            'owner_code': '',
            'batch': _to_int(r.get("Exp"), 0),
            'round_number': _to_int(r.get("group_enumeration"), 0),
            'role': _force_str(r.get("role")),
            'id_in_group': _to_int(r.get("id"), 0),
            'partner_id': _to_int(r.get("partner_id"), 0),
            'condition': _force_str(r.get("Condition")),
            'item_nr': _force_str(r.get("Item.Nr")),
            'image': _force_str(r.get("Item")),
            'sentences': _force_str(r.get("sentences") or "[]"),
            'rewards': '',
            'busy': False,
            'processed': False,
        })
        batch_id += 1
    
    session.vars['batches'] = batches
    logger.info(f"Created {len(batches)} batch records in session.vars")


# ============================================================================
# PAGES
# ============================================================================

class DebugBatches(Page):
    """Temporary debug page - remove after testing"""
    @staticmethod
    def is_displayed(player):
        # Only show for first participant, round 1
        return player.round_number == 1 and player.id_in_subsession == 1
    
    @staticmethod
    def vars_for_template(player):
        batches = get_all_batches_from_session(player.session)
        logger.info(f"Total batches: {len(batches)}")
        logger.info(f"First 3 batches: {batches[:3]}")
        
        # Count by round
        rounds_count = {}
        for b in batches:
            rnd = b.get('round_number', 0)
            rounds_count[rnd] = rounds_count.get(rnd, 0) + 1
        
        logger.info(f"Batches per round: {rounds_count}")
        
        return {
            'total_batches': len(batches),
            'first_batches': batches[:10],
            'rounds_count': rounds_count,
        }


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
            logger.info(f"Calling start() for participant {player.participant.code}, round {player.round_number}")
            player.start()
            logger.info(f"After start(): link_id={player.link_id}, faulty={player.faulty}, role={player.inner_role}")

        if player.faulty:
            logger.warning(f"Player {player.participant.code} marked as faulty in round {player.round_number}")
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


# ============================================================================
# DATA EXPORT
# ============================================================================

def custom_export(players):
    """
    STREAMED export, one row per participant per round (1..80).
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

    # Get all sessions involved
    sessions = {}
    for p in players.select_related('session', 'participant'):
        if p.session.code not in sessions:
            sessions[p.session.code] = p.session

    # Stream players
    p_it = (
        players
        .select_related('session', 'participant')
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

        # Get batch data from session.vars
        session = sessions.get(session_code)
        if session:
            batches = get_all_batches_from_session(session)
            batch_data = None
            for b in batches:
                if b.get('owner_code') == participant_code and b.get('round_number') == rnd:
                    batch_data = b
                    break
            
            if batch_data:
                role = batch_data.get('role', inner_role or '')
                cond = batch_data.get('condition', '')
                item_nr = batch_data.get('item_nr', '')
                img = batch_data.get('image', '')
                producer_sentences = batch_data.get('sentences', '')
                interpreter_rewards = batch_data.get('rewards', '')
            else:
                role = inner_role or ""
                cond = ""
                item_nr = ""
                img = ""
                producer_sentences = prod_dec or ""
                interpreter_rewards = interp_dec or ""
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


page_sequence = [DebugBatches, FaultyCatcher, Q, Feedback, FinalForProlific]
