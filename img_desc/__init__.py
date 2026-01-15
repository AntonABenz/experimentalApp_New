from otree.api import *
import json
import logging
import re
import time
import os
import hashlib
from pathlib import Path
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


# =============================================================================
# Helpers
# =============================================================================

def normalize_key(k: str) -> str:
    return re.sub(r"[\s_]+", "_", str(k or "").strip().lower())


def _truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def _sha256_file(path: Path, max_bytes: int = 50_000_000) -> str:
    # fast enough for your file sizes; capped for safety
    h = hashlib.sha256()
    with path.open("rb") as f:
        remaining = max_bytes
        while remaining > 0:
            chunk = f.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    return h.hexdigest()


def _resolve_excel_path(filename: str) -> Path:
    # Match what you logged on Heroku
    candidates = [
        Path("/app") / filename,
        Path("/app/start/data") / filename,
        Path("/app/data") / filename,
        Path(filename),
        Path("start/data") / filename,
        Path("data") / filename,
    ]

    logger.info(f"Excel filename requested: {filename}")
    logger.info(f"Excel candidates: {[str(p) for p in candidates]}")

    for p in candidates:
        if p.exists():
            stat = p.stat()
            fp = {
                "path": str(p),
                "size_bytes": stat.st_size,
                "mtime_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                "sha256": _sha256_file(p),
            }
            logger.info(f"Using Excel file fingerprint: {fp}")
            return p

    raise FileNotFoundError(f"Excel file not found: {filename}")


def _clean_cell(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    # keep "None" as literal string if it is literally "None" in Excel
    # but remove pandas nan-like
    if s.lower() == "nan":
        return ""
    return s


def _parse_sentences_from_row(row: dict) -> str:
    """
    Your new file has Sentence_1_1, Sentence_1_2 ... Sentence_5_2
    We store them as JSON list-of-lists:
      [["Some","the A"], ["None","the C"], ...]
    """
    out = []
    for i in range(1, 6):
        a = _clean_cell(row.get(f"Sentence_{i}_1"))
        b = _clean_cell(row.get(f"Sentence_{i}_2"))
        if a or b:
            out.append([a, b])
    return json.dumps(out)


def _safe_int(x, default=0) -> int:
    try:
        # handles "2", "2.0"
        return int(float(str(x).strip()))
    except Exception:
        return default


# =============================================================================
# Models
# =============================================================================

class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # who am I in the Excel file? (0-based integer)
    excel_id = models.IntegerField(initial=-1)

    # schedule + state per round for this participant (JSON list)
    batch_history = models.LongStringField(initial="[]")

    inner_role = models.StringField()
    feedback = models.LongStringField(blank=True)

    producer_decision = models.LongStringField(blank=True)
    interpreter_decision = models.LongStringField(blank=True)

    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)

    faulty = models.BooleanField(initial=False)
    full_return_url = models.StringField(blank=True)

    # -----------------------------
    # Core schedule access
    # -----------------------------
    def _history(self):
        try:
            return json.loads(self.batch_history or "[]")
        except Exception:
            return []

    def current_item(self) -> dict:
        hist = self._history()
        idx = self.round_number - 1
        if idx < 0 or idx >= len(hist):
            return {}
        return hist[idx]

    def update_current_item(self, updates: dict):
        hist = self._history()
        idx = self.round_number - 1
        if 0 <= idx < len(hist):
            hist[idx].update(updates)
            self.batch_history = json.dumps(hist)

    # -----------------------------
    # Sentences / image URL
    # -----------------------------
    def get_sentences_data(self):
        item = self.current_item()
        # Interpreters see sentences from the row (preloaded).
        # Producers do not need sentences.
        try:
            raw = item.get("sentences", "[]") if self.inner_role == INTERPRETER else "[]"
            return json.loads(raw or "[]")
        except Exception:
            return []

    def get_full_sentences(self):
        prefix = self.session.vars.get("prefix", "") or ""
        suffixes = self.session.vars.get("suffixes") or []

        sentences = self.get_sentences_data() or []
        sentences = [s for s in sentences if isinstance(s, list)]

        res = []
        for sentence in sentences:
            expansion = []
            if prefix:
                expansion.append(prefix)
            for val, suf in zip(sentence, suffixes):
                expansion.append(str(val))
                expansion.append(str(suf))
            res.append(" ".join([x for x in expansion if x != ""]))
        return res

    def get_image_url(self):
        item = self.current_item()
        img = _clean_cell(item.get("image"))

        # Excel uses NA_x sometimes
        if img == "NA_x":
            img = self.session.vars.get("fallback_image", "") or ""

        # if still empty or NA-like, show nothing
        if not img or img.lower() in {"none", "nan", "na", "x"}:
            return ""

        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        ext = self.session.vars.get("extension", "png") or "png"

        # if someone pasted console URL in settings, you should *not* use it as base
        # (we just use what's in settings; you can keep your console->public fix in reading_xls if needed)

        clean_name = img.replace(" ", "_")
        if not clean_name.lower().endswith(f".{ext}"):
            clean_name = f"{clean_name}.{ext}"

        # your old adjustment
        if "amazonaws.com" in base:
            base = base.replace("/practice", "")

        return f"{base}/{clean_name}"

    # -----------------------------
    # Excel ID assignment
    # -----------------------------
    def assign_excel_id_if_needed(self):
        if self.excel_id >= 0:
            return

        session = self.session
        pool = session.vars.get("excel_id_pool") or []
        assigned = session.vars.get("assigned_excel_ids") or []

        # pick smallest free id deterministically
        free = [x for x in pool if x not in assigned]
        if not free:
            self.faulty = True
            return

        chosen = sorted(free)[0]
        assigned.append(chosen)
        session.vars["assigned_excel_ids"] = assigned

        self.excel_id = chosen

        histories = session.vars.get("histories_by_excel_id") or {}
        my_hist = histories.get(str(chosen)) or []
        self.batch_history = json.dumps(my_hist)

        # sanity check
        if len(my_hist) != Constants.num_rounds:
            logger.warning(f"Excel ID {chosen} has {len(my_hist)} rounds (expected {Constants.num_rounds})")
            # you can mark faulty if you want strictness:
            # self.faulty = True


# =============================================================================
# Session creation
# =============================================================================

def creating_session(subsession: Subsession):
    session = subsession.session
    if subsession.round_number != 1:
        return

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Missing filename in session config")

    # --- Load Excel (and log fingerprint) ---
    xlsx_path = _resolve_excel_path(filename)
    xl = pd.ExcelFile(xlsx_path, engine="openpyxl")

    # Pick sheet: if "data" exists use it, else first sheet
    sheet_names_lower = [s.lower().strip() for s in xl.sheet_names]
    if "data" in sheet_names_lower:
        data_sheet = xl.sheet_names[sheet_names_lower.index("data")]
    else:
        data_sheet = xl.sheet_names[0]

    df = xl.parse(
        data_sheet,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    df.columns = [str(c).strip() for c in df.columns]
    logger.info(f"Excel columns found: {list(df.columns)}")
    if len(df) > 0:
        logger.info(f"First row sample: {df.iloc[0].to_dict()}")

    # --- Normalize columns we will use ---
    # Your sample has: Trial, Round, Producer, Interpreter, Condition, Item.Nr, Item, Sentence_1_1...
    # Some files might have slightly different names; be defensive:
    colmap = {normalize_key(c): c for c in df.columns}

    def C(name, fallback=None):
        return colmap.get(normalize_key(name)) or fallback

    exp_col = C("exp", fallback=df.columns[0])  # first column often is Exp-like
    trial_col = C("trial", fallback=C("round"))  # if missing, trial isn't used
    round_col = C("round", fallback=C("group_enumeration"))
    prod_col = C("producer")
    interp_col = C("interpreter")
    cond_col = C("condition")
    itemnr_col = C("item.nr")
    item_col = C("item")

    if not prod_col or not interp_col or not round_col or not item_col:
        raise RuntimeError(
            f"Missing required columns. Need at least Producer, Interpreter, Round, Item. "
            f"Found: {list(df.columns)}"
        )

    # --- Build fallback image from valid items ---
    valid_images = []
    for _, r in df.iterrows():
        img = _clean_cell(r.get(item_col))
        if img and img not in {"NA_x"} and not img.lower().startswith("na"):
            valid_images.append(img)
    fallback_image = valid_images[0] if valid_images else "d-A-B-BC-3"
    session.vars["fallback_image"] = fallback_image

    # --- Load settings from reading_xls (if you still use it) OR from a "settings" sheet ---
    # If your pipeline already uses reading_xls/get_data.py for settings, keep using it:
    try:
        from reading_xls.get_data import get_data
        payload = get_data(filename)
        settings = payload.get("settings") or {}
    except Exception as e:
        logger.warning(f"Could not read settings via reading_xls.get_data: {e}")
        settings = {}

    # Normalize settings keys
    clean_settings = {normalize_key(k): v for k, v in settings.items()}

    # Core settings used by template + url builder
    session.vars["s3path_base"] = clean_settings.get("s3path_base") or clean_settings.get("s3path") or ""
    session.vars["extension"] = clean_settings.get("extension") or "png"
    session.vars["prefix"] = clean_settings.get("prefix") or ""
    session.vars["interpreter_title"] = clean_settings.get("interpreter_title") or "Buy medals:"
    session.vars["caseflag"] = _truthy(clean_settings.get("caseflag"))
    session.vars["instructions_url"] = clean_settings.get("instructions_url") or ""

    # Suffixes (suffix_1, suffix_2...)
    suffixes = []
    for i in range(1, 21):
        v = clean_settings.get(f"suffix_{i}")
        if v is not None and str(v).strip() != "":
            suffixes.append(str(v).strip())
    session.vars["suffixes"] = suffixes

    # Allowed values/regex
    allowed_values = []
    allowed_regexes = []
    for i in range(1, 21):
        v = clean_settings.get(f"allowed_values_{i}")
        r = clean_settings.get(f"allowed_regex_{i}") or clean_settings.get(f"allowed_regex_{i}".replace("_", ""))
        if v or r:
            allowed_values.append([x.strip() for x in str(v).split(";") if x.strip()] if v else [])
            allowed_regexes.append(str(r).strip() if r else "")
    session.vars["allowed_values"] = allowed_values
    session.vars["allowed_regexes"] = allowed_regexes

    # Interpreter choices
    raw_choices = clean_settings.get("interpreter_choices") or ""
    if isinstance(raw_choices, list):
        session.vars["interpreter_choices"] = raw_choices
    else:
        session.vars["interpreter_choices"] = [x.strip() for x in str(raw_choices).split(";") if x.strip()]

    # Prolific completion code
    if session.config.get("completion_code"):
        session.vars["completion_code"] = str(session.config["completion_code"])

    # --- Build histories by excel_id ---
    # Each row produces 2 schedule entries:
    #   producer_id gets a PRODUCER entry
    #   interpreter_id gets an INTERPRETER entry
    histories = {}

    def add(excel_id: int, entry: dict):
        histories.setdefault(str(excel_id), []).append(entry)

    for _, r in df.iterrows():
        row = r.to_dict()

        exp = _safe_int(row.get(exp_col), 0)
        rnd = _safe_int(row.get(round_col), 0)
        trial = _safe_int(row.get(trial_col), 0) if trial_col else 0

        prod_id = _safe_int(row.get(prod_col), -1)
        int_id = _safe_int(row.get(interp_col), -1)

        condition = _clean_cell(row.get(cond_col))
        item_nr = _clean_cell(row.get(itemnr_col))
        image = _clean_cell(row.get(item_col))

        # Sentence columns into JSON
        sent_json = _parse_sentences_from_row(row)

        sort_key = (exp, rnd, trial)

        if prod_id >= 0:
            add(prod_id, dict(
                sort_key=sort_key,
                role=PRODUCER,
                partner_excel_id=int_id,
                exp=exp,
                round_in_excel=rnd,
                trial=trial,
                condition=condition,
                item_nr=item_nr,
                image=image,
                sentences="[]",  # producers don't need sentence display
                producer_sentences="",  # will be filled when they submit
                interpreter_rewards="",
            ))

        if int_id >= 0:
            add(int_id, dict(
                sort_key=sort_key,
                role=INTERPRETER,
                partner_excel_id=prod_id,
                exp=exp,
                round_in_excel=rnd,
                trial=trial,
                condition=condition,
                item_nr=item_nr,
                image=image,        # interpreter row keeps image for reference/export
                sentences=sent_json,  # interpreters display this
                producer_sentences="",
                interpreter_rewards="",
            ))

    # Sort and then renumber to 1..N
    for k, lst in histories.items():
        lst.sort(key=lambda x: x["sort_key"])
        for i, entry in enumerate(lst):
            entry["round_number"] = i + 1
            entry.pop("sort_key", None)

    # Pool of IDs (0-based)
    excel_id_pool = sorted([int(k) for k in histories.keys()])
    session.vars["excel_id_pool"] = excel_id_pool
    session.vars["assigned_excel_ids"] = []
    session.vars["histories_by_excel_id"] = histories

    # Debug sanity
    logger.info(f"Excel ID pool: {excel_id_pool}")
    if excel_id_pool:
        sample_id = str(excel_id_pool[0])
        logger.info(f"Rounds for first excel_id={sample_id}: {len(histories.get(sample_id, []))}")


# =============================================================================
# Pages
# =============================================================================

class FaultyCatcher(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.faulty

    def get(self):
        return redirect(Constants.FALLBACK_URL)


class Q(Page):
    form_model = "player"

    @staticmethod
    def is_displayed(player: Player):
        if player.round_number > Constants.num_rounds:
            return False

        # assign excel_id + load schedule on first displayed round
        player.assign_excel_id_if_needed()
        if player.faulty:
            return False

        item = player.current_item()
        if not item:
            player.faulty = True
            return False

        player.inner_role = item.get("role", "")

        if player.start_decision_time == 0:
            player.start_decision_time = time.time()

        return True

    @staticmethod
    def get_form_fields(player: Player):
        if player.inner_role == PRODUCER:
            return ["producer_decision"]
        if player.inner_role == INTERPRETER:
            return ["interpreter_decision"]
        return []

    @staticmethod
    def vars_for_template(player: Player):
        item = player.current_item()

        raw_choices = player.session.vars.get("interpreter_choices") or []
        if isinstance(raw_choices, str):
            interpreter_choices = [x.strip() for x in raw_choices.split(";") if x.strip()]
        else:
            interpreter_choices = raw_choices

        return dict(
            d=item,
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regexes=player.session.vars.get("allowed_regexes", []),
            suffixes=player.session.vars.get("suffixes", []),
            prefix=player.session.vars.get("prefix", ""),
            interpreter_choices=interpreter_choices,
            interpreter_title=player.session.vars.get("interpreter_title") or "Buy medals:",
            instructions_url=player.session.vars.get("instructions_url") or "",
            server_image_url=player.get_image_url(),
            excel_id=player.excel_id,
        )

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        player.end_decision_time = time.time()
        if player.start_decision_time:
            player.decision_seconds = player.end_decision_time - player.start_decision_time

        # write decisions into history item
        if player.inner_role == PRODUCER:
            player.update_current_item({"producer_sentences": player.producer_decision or ""})
        elif player.inner_role == INTERPRETER:
            player.update_current_item({"interpreter_rewards": player.interpreter_decision or ""})

        # reset timer for next round
        player.start_decision_time = 0


class Feedback(Page):
    form_model = "player"
    form_fields = ["feedback"]

    @staticmethod
    def is_displayed(player: Player):
        return player.round_number == Constants.num_rounds


class FinalForProlific(Page):
    @staticmethod
    def is_displayed(player: Player):
        return player.session.config.get("for_prolific") and player.round_number == Constants.num_rounds

    def get(self):
        cc = (self.player.session.vars.get("completion_code")
              or self.player.session.config.get("completion_code"))
        if not cc:
            return redirect(Constants.API_ERR_URL)
        return redirect(STUBURL + str(cc))


# =============================================================================
# Export
# =============================================================================

def custom_export(players):
    yield [
        "session_code",
        "participant_code",
        "excel_id",
        "round_number",
        "role",
        "partner_excel_id",
        "exp",
        "round_in_excel",
        "trial",
        "condition",
        "item_nr",
        "image",
        "producer_decision",
        "interpreter_decision",
        "decision_seconds",
        "feedback",
    ]

    seen = set()
    for p in players:
        if p.participant.code in seen:
            continue
        seen.add(p.participant.code)

        # pull per-round timing from Player model rows
        timing_by_round = {}
        fb = ""
        for pr in p.participant.get_players():
            timing_by_round[pr.round_number] = pr.decision_seconds
            if pr.round_number == Constants.num_rounds:
                fb = pr.feedback or ""

        # schedule state lives in p.batch_history (for this participant)
        try:
            hist = json.loads(p.batch_history or "[]")
        except Exception:
            hist = []

        for item in hist:
            rnd = _safe_int(item.get("round_number"), 0)
            if rnd < 1 or rnd > Constants.num_rounds:
                continue

            yield [
                p.session.code,
                p.participant.code,
                p.excel_id,
                rnd,
                item.get("role", ""),
                item.get("partner_excel_id", ""),
                item.get("exp", ""),
                item.get("round_in_excel", ""),
                item.get("trial", ""),
                item.get("condition", ""),
                item.get("item_nr", ""),
                item.get("image", ""),
                item.get("producer_sentences", ""),
                item.get("interpreter_rewards", ""),
                timing_by_round.get(rnd, 0),
                fb if rnd == Constants.num_rounds else "",
            ]


page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
