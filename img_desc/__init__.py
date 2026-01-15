# img_desc/__init__.py

from otree.api import *
import json
import logging
import time
import random
import re
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


# ----------------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------------
class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # per-participant schedule (list[dict]) stored as JSON
    batch_history = models.LongStringField(initial="[]")

    inner_role = models.StringField()
    faulty = models.BooleanField(initial=False)
    feedback = models.LongStringField(label="")

    producer_decision = models.LongStringField()
    interpreter_decision = models.LongStringField()

    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)

    full_return_url = models.StringField(blank=True)

    # -------------------------
    # schedule helpers
    # -------------------------
    def _history(self):
        try:
            return json.loads(self.batch_history or "[]")
        except Exception:
            return []

    def get_current_batch_data(self):
        rnd = int(self.round_number or 0)
        for item in self._history():
            if int(item.get("round_number", 0)) == rnd:
                return item
        return {}

    def update_current_batch_data(self, updates: dict):
        try:
            hist = self._history()
            rnd = int(self.round_number or 0)
            changed = False
            for item in hist:
                if int(item.get("round_number", 0)) == rnd:
                    item.update(updates)
                    changed = True
                    break
            if changed:
                self.batch_history = json.dumps(hist)
        except Exception:
            pass

    # -------------------------
    # content helpers used by template
    # -------------------------
    def get_image_url(self):
        data = self.get_current_batch_data()
        img = clean_str(data.get("image", ""))

        if not img or img.lower() in {"nan", "na", "na_x", "none", "", "x"}:
            return ""

        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        ext = self.session.vars.get("extension") or "png"

        # keep your prior behavior: strip "/practice" if present in AWS URLs
        if "amazonaws" in base:
            base = base.replace("/practice", "")

        clean_name = img.replace(" ", "_")
        if not clean_name.lower().endswith(f".{ext}"):
            clean_name = f"{clean_name}.{ext}"

        return f"{base}/{clean_name}"

    def get_sentences_data(self):
        """
        For PRODUCER: show nothing (they produce).
        For INTERPRETER: use producer_sentences from schedule (already resolved),
        including for Producer=0 rows (virtual producer).
        """
        data = self.get_current_batch_data()
        if not data:
            return []

        role = data.get("role", "")
        if role == PRODUCER:
            return []

        raw = data.get("producer_sentences") or data.get("sentences") or "[]"
        try:
            return json.loads(raw) if raw else []
        except Exception:
            return []

    def get_full_sentences(self):
        prefix = self.session.vars.get("prefix") or ""
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []
        sentences = [s for s in sentences if isinstance(s, list)]

        res = []
        for sentence in sentences:
            parts = []
            if prefix:
                parts.append(str(prefix))
            # pair each value with suffix_1, suffix_2, ...
            for val, suf in zip(sentence, suffixes):
                parts.append(str(val))
                if suf:
                    parts.append(str(suf))
            # if sentence longer than suffixes, append remaining values
            if len(sentence) > len(suffixes):
                for extra in sentence[len(suffixes):]:
                    parts.append(str(extra))
            res.append(" ".join([p for p in parts if p is not None and str(p).strip() != ""]))
        return res


# ----------------------------------------------------------------------------
# UTIL
# ----------------------------------------------------------------------------
def safe_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def clean_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() in {"nan", "none"}:
        return ""
    return s


def normalize_key(key):
    if not key:
        return ""
    return re.sub(r"[\s_]+", "_", str(key).lower().strip())


def _truthy(v) -> bool:
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def is_valid_real_image(img: str) -> bool:
    img = clean_str(img)
    if not img:
        return False
    low = img.lower()
    if low in {"na_x", "na", "nan", "none", "x"}:
        return False
    # reject practice placeholders like D_5_...
    if img.startswith("D_") or img.startswith("d_"):
        return False
    # your real images are like d-A-B-...
    if not img.startswith("d-"):
        return False
    return True


def extract_sentences_from_row(r: dict) -> str:
    """
    Your new Excel has Sentence_1_1, Sentence_1_2, ..., Sentence_5_2.
    We convert them into JSON list of pairs.
    """
    pairs = []
    for i in range(1, 6):
        a = clean_str(r.get(f"Sentence_{i}_1"))
        b = clean_str(r.get(f"Sentence_{i}_2"))
        if a or b:
            pairs.append([a, b])
    return json.dumps(pairs)


def fix_s3_url(raw_s3: str) -> str:
    raw_s3 = clean_str(raw_s3)
    # convert AWS console bucket URL to public-ish bucket endpoint (best-effort)
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            bucket = raw_s3.split("buckets/")[1].split("?")[0].strip("/")
            # keep your eu-central-1 assumption (change if needed)
            return f"https://{bucket}.s3.eu-central-1.amazonaws.com"
        except Exception:
            return raw_s3
    return raw_s3


# ----------------------------------------------------------------------------
# SESSION CREATION
# ----------------------------------------------------------------------------
def creating_session(subsession: Subsession):
    session = subsession.session
    if subsession.round_number != 1:
        return

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("No filename in session config")

    # reading_xls/get_data.py should return dict(data=..., settings=..., ...)
    from reading_xls.get_data import get_data
    excel_payload = get_data(filename)

    raw_data = excel_payload.get("data")
    settings = excel_payload.get("settings") or {}

    # raw_data might be a DataFrame (old path) or a list-of-dicts (new path)
    if hasattr(raw_data, "to_dict"):
        raw_records = raw_data.to_dict(orient="records")
    elif isinstance(raw_data, list):
        raw_records = raw_data
    else:
        # last resort
        raw_records = []

    # ---- SETTINGS INTO session.vars (normalize keys) ----
    clean_settings = {normalize_key(k): clean_str(v) for k, v in settings.items()}

    s3_raw = clean_settings.get("s3path") or clean_settings.get("s3path_base") or ""
    s3_fixed = fix_s3_url(s3_raw)

    session.vars["s3path_base"] = s3_fixed
    session.vars["extension"] = clean_settings.get("extension") or "png"
    session.vars["prefix"] = clean_settings.get("prefix") or ""
    session.vars["interpreter_title"] = clean_settings.get("interpreter_title") or "Buy medals:"
    session.vars["caseflag"] = _truthy(clean_settings.get("caseflag"))

    # suffix_1..suffix_10 (keep order)
    suffixes = []
    for i in range(1, 11):
        v = clean_settings.get(f"suffix_{i}")
        if v:
            suffixes.append(v)
    session.vars["suffixes"] = suffixes

    # interpreter_choices can be a semicolon string or list (get_data often converts)
    ic = settings.get("interpreter_choices")
    if isinstance(ic, str):
        interpreter_choices = [x.strip() for x in ic.split(";") if x.strip()]
    elif isinstance(ic, list):
        interpreter_choices = ic
    else:
        interpreter_choices = []
    session.vars["interpreter_choices"] = interpreter_choices

    # allowed values/regex from get_data normalization
    session.vars["allowed_values"] = settings.get("allowed_values", []) or []
    session.vars["allowed_regexes"] = settings.get("allowed_regex", []) or []

    session.vars["instructions_url"] = clean_settings.get("instructions_url") or "https://google.com"

    if session.config.get("completion_code"):
        session.vars["completion_code"] = str(session.config["completion_code"])

    # ---- BUILD VALID IMAGE POOL: only real producer rows Producer != 0 ----
    valid_pool = []
    for r in raw_records:
        prod_excel_id = safe_int(r.get("Producer"), 0)
        img = clean_str(r.get("Item"))
        if prod_excel_id != 0 and is_valid_real_image(img):
            valid_pool.append(img)

    if not valid_pool:
        valid_pool = ["d-A-B-BC-3"]  # hard fallback

    # ---- BUILD SCHEDULE: each row creates Producer entry + Interpreter entry,
    #      unless Producer==0, then only Interpreter entry (with image from pool),
    #      but interpreter still uses THIS row's sentences. ----
    from collections import defaultdict
    data_by_excel_id = defaultdict(list)

    for r in raw_records:
        exp_num = safe_int(r.get("Exp"), 0)
        round_in_excel = safe_int(r.get("Round"), 0)
        trial = safe_int(r.get("Trial"), 0)

        condition = clean_str(r.get("Condition"))
        item_nr = clean_str(r.get("Item.Nr"))
        image_raw = clean_str(r.get("Item"))

        producer_excel_id = safe_int(r.get("Producer"), 0)
        interpreter_excel_id = safe_int(r.get("Interpreter"), 0)

        # if interpreter missing, row is useless for 3P/5I design
        if interpreter_excel_id == 0:
            continue

        # interpreter sentences come from the row always
        sentences_json = extract_sentences_from_row(r)

        # Producer schedule entry exists only if Producer != 0
        if producer_excel_id != 0:
            # PRODUCER gets the row image as-is (even if it's weird);
            # BUT if it's not a real image, swap to a valid one.
            prod_image = image_raw
            if not is_valid_real_image(prod_image):
                prod_image = random.choice(valid_pool)

            data_by_excel_id[producer_excel_id].append(
                {
                    "role": PRODUCER,
                    "partner_excel_id": interpreter_excel_id,
                    "exp": exp_num,
                    "round_in_excel": round_in_excel,
                    "trial": trial,
                    "condition": condition,
                    "item_nr": item_nr,
                    "image": prod_image,
                    "sentences": "[]",
                    "producer_sentences": "",
                    "interpreter_rewards": "",
                }
            )

            # INTERPRETER entry paired with producer
            data_by_excel_id[interpreter_excel_id].append(
                {
                    "role": INTERPRETER,
                    "partner_excel_id": producer_excel_id,
                    "exp": exp_num,
                    "round_in_excel": round_in_excel,
                    "trial": trial,
                    "condition": condition,
                    "item_nr": item_nr,
                    "image": prod_image,
                    "sentences": sentences_json,
                    "producer_sentences": sentences_json,
                    "interpreter_rewards": "",
                }
            )
        else:
            # Producer==0 case:
            # - ignore image from this row
            # - use ANY valid real image from pool
            # - interpreter still uses THIS row's sentences_json
            picked = random.choice(valid_pool)

            data_by_excel_id[interpreter_excel_id].append(
                {
                    "role": INTERPRETER,
                    "partner_excel_id": 0,
                    "exp": exp_num,
                    "round_in_excel": round_in_excel,
                    "trial": trial,
                    "condition": condition,
                    "item_nr": item_nr,
                    "image": picked,
                    "sentences": sentences_json,
                    "producer_sentences": sentences_json,
                    "interpreter_rewards": "",
                }
            )

    # ---- ASSIGN EACH PARTICIPANT A SEQUENTIAL ROUND SCHEDULE ----
    # IMPORTANT: this assumes participant id_in_subsession == excel_id (i.e., 1..N),
    # and your excel ids are also 1..N (NOT 0..N-1).
    # If your Excel uses 0..N-1, then you must +1 everywhere above.
    players = subsession.get_players()

    for p in players:
        excel_id = p.id_in_subsession

        my_items = data_by_excel_id.get(excel_id, [])
        # this ordering is what controls “round 3 is interpreter”
        my_items.sort(key=lambda x: (safe_int(x.get("exp")), safe_int(x.get("round_in_excel")), safe_int(x.get("trial"))))

        final_history = []
        for i, item in enumerate(my_items):
            item["round_number"] = i + 1
            final_history.append(item)

        p.batch_history = json.dumps(final_history)
        p.participant.vars["batch_history"] = p.batch_history

        if len(final_history) == 0:
            logger.warning(f"Participant excel_id={excel_id} got EMPTY schedule.")

    logger.info(
        "Schedule built. "
        f"players={len(players)} "
        f"valid_pool={len(valid_pool)} "
        f"rows_in_excel={len(raw_records)}"
    )


# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
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

        # hydrate from participant vars (first time)
        if (player.batch_history == "[]" or not player.batch_history) and "batch_history" in player.participant.vars:
            player.batch_history = player.participant.vars["batch_history"]

        data = player.get_current_batch_data()
        if not data:
            if player.round_number == 1:
                player.faulty = True
            return False

        player.inner_role = data.get("role", "")

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
            d=player.get_current_batch_data(),
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regexes=player.session.vars.get("allowed_regexes", []),
            suffixes=player.session.vars.get("suffixes", []),
            prefix=player.session.vars.get("prefix", ""),
            interpreter_choices=interpreter_choices,
            interpreter_title=interpreter_title,
            instructions_url=player.session.vars.get("instructions_url"),
            # template can use player.get_image_url directly, but keeping this is handy too
            server_image_url=player.get_image_url(),
            caseflag=player.session.vars.get("caseflag"),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.end_decision_time = time.time()
        if player.start_decision_time:
            player.decision_seconds = player.end_decision_time - player.start_decision_time

        updates = {}
        if player.inner_role == PRODUCER:
            updates["producer_sentences"] = player.producer_decision
        elif player.inner_role == INTERPRETER:
            updates["interpreter_rewards"] = player.interpreter_decision

        if updates:
            player.update_current_batch_data(updates)
            player.participant.vars["batch_history"] = player.batch_history


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
        cc = self.player.session.vars.get("completion_code") or self.player.session.config.get("completion_code")
        if not cc:
            return redirect(Constants.API_ERR_URL)
        return redirect(STUBURL + str(cc))


# ----------------------------------------------------------------------------
# EXPORT
# ----------------------------------------------------------------------------
def custom_export(players):
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
        "feedback",
    ]

    processed = set()
    for p in players:
        if p.participant.code in processed:
            continue
        processed.add(p.participant.code)

        history_json = p.participant.vars.get("batch_history", "[]")
        try:
            history = json.loads(history_json)
        except Exception:
            history = []

        # timing + feedback from Player rows
        timing_map = {}
        feedback_str = ""
        for sub_p in p.participant.get_players():
            timing_map[sub_p.round_number] = sub_p.decision_seconds
            if sub_p.round_number == Constants.num_rounds:
                feedback_str = sub_p.feedback or ""

        history.sort(key=lambda x: safe_int(x.get("round_number"), 0))

        for item in history:
            rnd = safe_int(item.get("round_number"), 0)
            if rnd < 1 or rnd > Constants.num_rounds:
                continue

            yield [
                p.session.code,
                p.participant.code,
                rnd,
                item.get("role", ""),
                item.get("condition", ""),
                item.get("item_nr", ""),
                item.get("image", ""),
                item.get("producer_sentences", ""),
                item.get("interpreter_rewards", ""),
                timing_map.get(rnd, 0),
                feedback_str if rnd == Constants.num_rounds else "",
            ]


page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
