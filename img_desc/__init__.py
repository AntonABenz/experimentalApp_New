# img_desc/__init__.py
from otree.api import *
import json
import logging
import time
import re
import random
from starlette.responses import RedirectResponse

logger = logging.getLogger("benzapp.img_desc")

PRODUCER = "P"
INTERPRETER = "I"
STUBURL = "https://app.prolific.co/submissions/complete?cc="


class Constants(BaseConstants):
    name_in_url = "img_desc"
    players_per_group = None

    # You said: 80 rounds total with a 3P/5I schema in this app run
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

    # ---------- batch helpers ----------
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

    # ---------- rendering helpers ----------
    def get_image_url(self):
        data = self.get_current_batch_data()
        img = clean_str(data.get("image", ""))
        if not img or img.lower() in {"nan", "na", "na_x", "", "x"}:
            return ""

        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        ext = self.session.vars.get("extension") or "png"

        clean_name = img.replace(" ", "_")
        if not clean_name.lower().endswith(f".{ext}"):
            clean_name = f"{clean_name}.{ext}"
        return f"{base}/{clean_name}"

    def _resolve_sentence_lookup(self, lookup):
        """
        lookup:
          {"source_exp": int, "producer_slot": int, "interpreter_slot": int, "condition": str}
        resolves from session.vars["sentences_by_key"]

        Rule (per you):
          - when Exp=e, we look in Exp=e-1
          - uniqueness is by (Producer, Interpreter, Condition)
          - item can differ across experiments; do NOT key on item
        """
        if not lookup or not isinstance(lookup, dict):
            return "[]"

        src_exp = safe_int(lookup.get("source_exp"), -1)
        prod = safe_int(lookup.get("producer_slot"), 0)
        interp = safe_int(lookup.get("interpreter_slot"), 0)
        cond = clean_str(lookup.get("condition"))

        key = sentence_key(src_exp, prod, interp, cond)
        store = self.session.vars.get("sentences_by_key") or {}
        raw = store.get(key)

        if raw is None:
            return "[]"
        if isinstance(raw, str):
            return raw
        try:
            return json.dumps(raw)
        except Exception:
            return "[]"

    def get_sentences_data(self):
        """
        For interpreters, sentences come from:
          1) batch item producer_sentences if non-empty and not []
          2) else from sentence_lookup (resolved via session store)
        """
        data = self.get_current_batch_data()
        if not data:
            return []

        if data.get("role") == PRODUCER:
            return []

        raw = data.get("producer_sentences")

        if not raw or (isinstance(raw, str) and raw.strip() in {"", "[]"}):
            raw = self._resolve_sentence_lookup(data.get("sentence_lookup"))

        try:
            return json.loads(raw) if raw else []
        except Exception:
            return []

    def get_full_sentences(self):
        """
        Render sentence pairs using prefix/suffixes, preserving literal "None"
        """
        try:
            prefix = self.session.vars.get("prefix") or ""
            suffixes = self.session.vars.get("suffixes") or []
            sentences = self.get_sentences_data() or []
            sentences = [s for s in sentences if isinstance(s, list)]

            res = []
            for sentence in sentences:
                parts = []
                if prefix:
                    parts.append(str(prefix).strip())

                for val, suf in zip(sentence, suffixes):
                    val_str = clean_str(val)  # preserves "None"
                    if not val_str:
                        val_str = "None"
                    parts.append(val_str)
                    if suf:
                        parts.append(str(suf).strip())

                # tolerate extra fields
                if len(sentence) > len(suffixes):
                    for extra in sentence[len(suffixes) :]:
                        extra_str = clean_str(extra)
                        if not extra_str:
                            extra_str = "None"
                        parts.append(extra_str)

                res.append(" ".join([p for p in parts if str(p).strip() != ""]).strip())

            return res
        except Exception as e:
            logger.error(f"Error in get_full_sentences: {e}")
            return []

    def save_to_chain(self):
        pass


# ----------------------------------------------------------------------------
# UTIL
# ----------------------------------------------------------------------------
def safe_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def clean_str(x) -> str:
    """
    IMPORTANT: do NOT drop literal 'None' tokens.
    Only treat actual empty / NaN as empty.
    """
    if x is None:
        return ""
    s = str(x).strip()
    if s == "":
        return ""
    if s.lower() == "nan":
        return ""
    if s.lower() == "none":
        return "None"
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
    if low in {"na_x", "na", "nan", "", "x"}:
        return False
    if img.startswith("D_") or (img.startswith("d_") and not img.startswith("d-")):
        return False
    if img.startswith("d-") or img.startswith("D-"):
        return True
    return False


def get_exp_num(r: dict) -> int:
    """
    Robust Exp column detection.
    """
    for key in ["Exp", "Experiment", "Exp.", "experiment", "d111"]:
        v = r.get(key)
        if v not in (None, ""):
            return safe_int(v, 0)
    return 0


def extract_sentences_from_row(r: dict) -> str:
    """
    JSON list of pairs:
      [[Sentence_1_1, Sentence_1_2], ...]
    Includes rows where at least one side is non-empty.
    Preserves literal 'None'.
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
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            bucket = raw_s3.split("buckets/")[1].split("?")[0].strip("/")
            return f"https://{bucket}.s3.eu-central-1.amazonaws.com"
        except Exception:
            return raw_s3
    return raw_s3


def sentence_key(exp_num: int, producer_slot: int, interpreter_slot: int, condition: str) -> str:
    """
    Your rule: uniqueness for sentence fallback is (exp, producer, interpreter, condition).
    Do NOT include item.
    """
    condition = clean_str(condition)
    return f"{int(exp_num)}|{int(producer_slot)}|{int(interpreter_slot)}|{condition}"


def safe_json_loads(x, default):
    try:
        if x is None:
            return default
        if isinstance(x, (list, dict)):
            return x
        s = str(x).strip()
        if not s:
            return default
        return json.loads(s)
    except Exception:
        return default


# ----------------------------------------------------------------------------
# SESSION CREATION
# ----------------------------------------------------------------------------
def creating_session(subsession: Subsession):
    session = subsession.session
    if subsession.round_number != 1:
        return

    try:
        filename = session.config.get("filename")
        if not filename:
            raise RuntimeError("No filename in session config")

        logger.info(f"Starting session creation with filename: {filename}")

        # load data/settings via your existing loader
        try:
            from reading_xls.get_data import get_data
        except ImportError:
            import sys
            from pathlib import Path

            reading_xls_path = Path(__file__).parent.parent / "reading_xls"
            sys.path.insert(0, str(reading_xls_path))
            from get_data import get_data

        excel_payload = get_data(filename)
        raw_data = excel_payload.get("data")
        settings = excel_payload.get("settings") or {}

        if hasattr(raw_data, "to_dict"):
            rows = raw_data.to_dict(orient="records")
        else:
            rows = list(raw_data or [])

        # --- 1) Global settings ---
        clean_settings = {}
        for k, v in settings.items():
            if isinstance(v, str):
                clean_settings[normalize_key(k)] = clean_str(v)

        s3_raw = clean_settings.get("s3path") or clean_settings.get("s3path_base") or ""
        session.vars["s3path_base"] = fix_s3_url(s3_raw)
        session.vars["extension"] = clean_settings.get("extension") or "png"
        session.vars["prefix"] = clean_settings.get("prefix") or ""
        session.vars["interpreter_title"] = clean_settings.get("interpreter_title") or "Buy medals:"
        session.vars["caseflag"] = _truthy(clean_settings.get("caseflag"))
        session.vars["instructions_url"] = clean_settings.get("instructions_url") or "https://google.com"

        if session.config.get("completion_code"):
            session.vars["completion_code"] = str(session.config["completion_code"])

        # lists
        session.vars["allowed_values"] = settings.get("allowed_values", []) or []
        session.vars["allowed_regexes"] = settings.get("allowed_regex", []) or []

        suffixes = []
        for i in range(1, 11):
            v = clean_settings.get(f"suffix_{i}")
            if v:
                suffixes.append(v)
        session.vars["suffixes"] = suffixes

        ic = settings.get("interpreter_choices")
        if isinstance(ic, str):
            session.vars["interpreter_choices"] = [x.strip() for x in ic.split(";") if x.strip()]
        elif isinstance(ic, list):
            session.vars["interpreter_choices"] = ic
        else:
            session.vars["interpreter_choices"] = []

        # --- 2) Sentence store (Exp0 preload) ---
        session.vars["sentences_by_key"] = {}

        for r in rows:
            exp_num = get_exp_num(r)
            if exp_num != 0:
                continue
            prod = safe_int(r.get("Producer"), 0)
            interp = safe_int(r.get("Interpreter"), 0)
            cond = clean_str(r.get("Condition"))
            sent = extract_sentences_from_row(r)
            if sent and sent.strip() != "[]":
                k = sentence_key(0, prod, interp, cond)
                session.vars["sentences_by_key"][k] = sent

        # --- 3) Slot mapping ---
        slot_ids = set()
        for r in rows:
            p = safe_int(r.get("Producer"), 0)
            i = safe_int(r.get("Interpreter"), 0)
            if p not in {0, 9}:
                slot_ids.add(p)
            if i != 0:
                slot_ids.add(i)

        if not slot_ids:
            raise RuntimeError("No participant slot IDs found in data.")
        K = max(slot_ids)

        players = subsession.get_players()
        slot_to_pid = {}
        for idx, pl in enumerate(players, start=1):
            if idx <= K:
                slot_to_pid[idx] = pl.id_in_subsession

        # valid image pool (real images from Exp>=1, producer_slot not 0/9)
        valid_pool = []
        for r in rows:
            exp_num = get_exp_num(r)
            if exp_num < 1:
                continue
            producer_slot = safe_int(r.get("Producer"), 0)
            img = clean_str(r.get("Item"))
            if producer_slot not in {0, 9} and is_valid_real_image(img):
                valid_pool.append(img)
        if not valid_pool:
            valid_pool = ["d-A-B-BC-3"]

        # --- 4) Build schedules per pid in EXACT SHEET ORDER (Exp>=1 only) ---
        from collections import defaultdict

        data_by_pid = defaultdict(list)

        for idx, r in enumerate(rows):
            exp_num = get_exp_num(r)
            if exp_num < 1:
                continue  # Exp0 never scheduled

            round_in_excel = safe_int(r.get("Round"), 0)
            trial = safe_int(r.get("Trial"), 0)
            condition = clean_str(r.get("Condition"))
            item_nr = clean_str(r.get("Item.Nr"))
            image_raw = clean_str(r.get("Item"))
            producer_slot = safe_int(r.get("Producer"), 0)
            interpreter_slot = safe_int(r.get("Interpreter"), 0)

            sort_key = (exp_num, idx)   # exact sheet order inside each experiment
            
            # Debug row tracking (best-effort; depends on loader/header)
            excel_row_index0 = idx
            excel_row_number_guess = idx + 2

            # image choice
            if producer_slot == 0:
                picked = image_raw if image_raw else random.choice(valid_pool)
            else:
                picked = image_raw if is_valid_real_image(image_raw) else random.choice(valid_pool)

            extracted = extract_sentences_from_row(r)
            has_extracted = extracted and extracted.strip() != "[]"

            # ----------------------------
            # PRODUCER-ONLY ROWS (Interpreter == 0)
            # ----------------------------
            if interpreter_slot == 0:
                prod_pid = slot_to_pid.get(producer_slot) if producer_slot not in {0, 9} else None
                if not prod_pid:
                    continue

                data_by_pid[prod_pid].append(
                    {
                        "sort_key": sort_key,
                        "role": PRODUCER,
                        "partner_id": 0,
                        "exp": exp_num,
                        "round_in_excel": round_in_excel,
                        "trial": trial,
                        "condition": condition,
                        "item_nr": item_nr,
                        "image": picked,
                        "producer_sentences": "",
                        "sentence_lookup": None,
                        "interpreter_rewards": "",
                        "producer_slot": producer_slot,
                        "interpreter_slot": 0,
                        "sentence_store_key": {
                            "exp": exp_num,
                            "producer_slot": producer_slot,
                            "interpreter_slot": 0,
                            "condition": condition,
                        },
                        "excel_row_index0": excel_row_index0,
                        "excel_row_number_guess": excel_row_number_guess,
                    }
                )
                continue

            # ----------------------------
            # Interpreter present (1..4)
            # ----------------------------
            interp_pid = slot_to_pid.get(interpreter_slot)
            if not interp_pid:
                continue

            prod_pid = slot_to_pid.get(producer_slot) if producer_slot not in {0, 9} else None

            # Interpreter sentence logic (your 3 cases):
            #   - if sentences exist in row: ALWAYS use them (even producer=0 and item NA)
            #   - if no sentences and producer != 0: lookup in previous experiment (exp-1)
            #   - do NOT attempt lookup when producer=0 and sentences empty (avoid keys like 0|0|1|...)
            sentence_lookup = None
            producer_sentences_for_interpreter = extracted if has_extracted else "[]"
            if (not has_extracted) and producer_slot != 0:
                sentence_lookup = {
                    "source_exp": exp_num - 1,  # Exp1->0, Exp2->1, ...
                    "producer_slot": producer_slot,
                    "interpreter_slot": interpreter_slot,
                    "condition": condition,
                }

            # interpreter-only row (Producer == 0)
            if producer_slot == 0:
                data_by_pid[interp_pid].append(
                    {
                        "sort_key": sort_key,
                        "role": INTERPRETER,
                        "partner_id": 0,
                        "exp": exp_num,
                        "round_in_excel": round_in_excel,
                        "trial": trial,
                        "condition": condition,
                        "item_nr": item_nr,
                        "image": picked,
                        "producer_sentences": producer_sentences_for_interpreter,
                        "sentence_lookup": sentence_lookup,
                        "interpreter_rewards": "",
                        "producer_slot": producer_slot,
                        "interpreter_slot": interpreter_slot,
                        "excel_row_index0": excel_row_index0,
                        "excel_row_number_guess": excel_row_number_guess,
                    }
                )
                continue

            # real producer/interpreter pair
            if not prod_pid:
                continue

            # producer item
            data_by_pid[prod_pid].append(
                {
                    "sort_key": sort_key,
                    "role": PRODUCER,
                    "partner_id": interp_pid,
                    "exp": exp_num,
                    "round_in_excel": round_in_excel,
                    "trial": trial,
                    "condition": condition,
                    "item_nr": item_nr,
                    "image": picked,
                    "producer_sentences": "",
                    "sentence_lookup": None,
                    "interpreter_rewards": "",
                    "producer_slot": producer_slot,
                    "interpreter_slot": interpreter_slot,
                    "sentence_store_key": {
                        "exp": exp_num,
                        "producer_slot": producer_slot,
                        "interpreter_slot": interpreter_slot,
                        "condition": condition,
                    },
                    "excel_row_index0": excel_row_index0,
                    "excel_row_number_guess": excel_row_number_guess,
                }
            )

            # interpreter item
            data_by_pid[interp_pid].append(
                {
                    "sort_key": sort_key,
                    "role": INTERPRETER,
                    "partner_id": prod_pid,
                    "exp": exp_num,
                    "round_in_excel": round_in_excel,
                    "trial": trial,
                    "condition": condition,
                    "item_nr": item_nr,
                    "image": picked,
                    "producer_sentences": producer_sentences_for_interpreter,
                    "sentence_lookup": sentence_lookup,
                    "interpreter_rewards": "",
                    "producer_slot": producer_slot,
                    "interpreter_slot": interpreter_slot,
                    "excel_row_index0": excel_row_index0,
                    "excel_row_number_guess": excel_row_number_guess,
                }
            )

        # --- 5) Finalize per-player history with 3P/5I mixing (NO STEALING / NO ROLE CONVERSION) ---
        # Important: do NOT "break early" (or you auto-advance to Feedback).
        # If the plan asks for a role that has no remaining items, we "skip that slot" and keep going,
        # and after the plan we append remaining items until we hit num_rounds.
        empty = []

        def round_plan(n):
            plan = []
            while len(plan) < n:
                plan.extend([PRODUCER] * 3)
                plan.extend([INTERPRETER] * 5)
            return plan[:n]

        for p in players:
            my_items = data_by_pid.get(p.id_in_subsession, [])
            my_items.sort(key=lambda x: x["sort_key"])

            producer_items = [it for it in my_items if it.get("role") == PRODUCER]
            interpreter_items = [it for it in my_items if it.get("role") == INTERPRETER]
            if p.id_in_subsession == 1:
                for it in interpreter_items[:15]:
                    logger.info(f"PID1 I_ITEM excel_row={it.get('excel_row_number_guess')} prod={it.get('producer_slot')} interp={it.get('interpreter_slot')} cond={it.get('condition')} sent={it.get('producer_sentences')[:40] if isinstance(it.get('producer_sentences'), str) else it.get('producer_sentences')}")


            final_history = []
            p_idx = 0
            i_idx = 0
            round_counter = 1

            plan = round_plan(Constants.num_rounds)

            # 5A) Follow the 3P/5I plan, skipping missing-role slots
            for need_role in plan:
                if round_counter > Constants.num_rounds:
                    break

                item = None
                if need_role == PRODUCER and p_idx < len(producer_items):
                    item = producer_items[p_idx].copy()
                    p_idx += 1
                elif need_role == INTERPRETER and i_idx < len(interpreter_items):
                    item = interpreter_items[i_idx].copy()
                    i_idx += 1

                if item is None:
                    # skip this plan slot, do NOT break
                    continue

                item.pop("sort_key", None)
                item["round_number"] = round_counter
                final_history.append(item)

                try:
                    logger.info(
                        f"SCHEDULE pid={p.id_in_subsession} round={round_counter} "
                        f"role={item.get('role')} exp={item.get('exp')} "
                        f"round_in_excel={item.get('round_in_excel')} trial={item.get('trial')} "
                        f"prod={item.get('producer_slot')} interp={item.get('interpreter_slot')} "
                        f"item={item.get('image')} excel_row_guess={item.get('excel_row_number_guess')}"
                    )
                except Exception:
                    pass

                round_counter += 1

            # 5B) Fill remaining rounds with leftover items (preserve each list order; no role conversion)
            while round_counter <= Constants.num_rounds and (p_idx < len(producer_items) or i_idx < len(interpreter_items)):
                # alternate preference: keep pattern-ish by taking whichever list is non-empty;
                # if both non-empty, take interpreter next (often more plentiful) to avoid starving it
                item = None
                if i_idx < len(interpreter_items):
                    item = interpreter_items[i_idx].copy()
                    i_idx += 1
                elif p_idx < len(producer_items):
                    item = producer_items[p_idx].copy()
                    p_idx += 1

                if item is None:
                    break

                item.pop("sort_key", None)
                item["round_number"] = round_counter
                final_history.append(item)
                round_counter += 1

            p.batch_history = json.dumps(final_history)
            p.participant.vars["batch_history"] = p.batch_history

            try:
                logger.info(
                    f"FINAL pid={p.id_in_subsession} total={len(final_history)} "
                    f"P={sum(1 for x in final_history if x.get('role') == PRODUCER)} "
                    f"I={sum(1 for x in final_history if x.get('role') == INTERPRETER)}"
                )
            except Exception:
                pass

            if not final_history:
                empty.append(p.id_in_subsession)

        if empty:
            logger.warning(f"EMPTY schedules for: {empty}")

    except Exception as e:
        logger.error(f"ERROR in creating_session: {e}", exc_info=True)
        raise


# ----------------------------------------------------------------------------
# PAGES
# ----------------------------------------------------------------------------
class FaultyCatcher(Page):
    @staticmethod
    def is_displayed(player):
        return player.faulty

    def get(self):
        return RedirectResponse(Constants.FALLBACK_URL, status_code=302)


class Q(Page):
    form_model = "player"

    @staticmethod
    def is_displayed(player):
        if player.round_number > Constants.num_rounds:
            return False

        # restore history if needed
        if (player.batch_history == "[]" or not player.batch_history) and "batch_history" in player.participant.vars:
            player.batch_history = player.participant.vars["batch_history"]

        data = player.get_current_batch_data()
        if not data:
            # if we don't have a schedule on round 1, something went wrong -> FaultyCatcher
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

        d = player.get_current_batch_data()

        # ensure interpreters can see resolved sentences in the template
        if d.get("role") == INTERPRETER:
            raw = d.get("producer_sentences")
            if not raw or (isinstance(raw, str) and raw.strip() in {"", "[]"}):
                resolved = player._resolve_sentence_lookup(d.get("sentence_lookup"))
                d = d.copy()
                d["producer_sentences"] = resolved

        return dict(
            d=d,
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regexes=player.session.vars.get("allowed_regexes", []),
            suffixes=player.session.vars.get("suffixes", []),
            prefix=player.session.vars.get("prefix", ""),
            interpreter_choices=interpreter_choices,
            interpreter_title=player.session.vars.get("interpreter_title") or "Buy medals:",
            instructions_url=player.session.vars.get("instructions_url"),
            server_image_url=player.get_image_url(),
            caseflag=player.session.vars.get("caseflag"),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        player.end_decision_time = time.time()
        if player.start_decision_time:
            player.decision_seconds = player.end_decision_time - player.start_decision_time
    
        data = player.get_current_batch_data() or {}
        updates = {}
    
        if player.inner_role == PRODUCER:
            updates["producer_sentences"] = player.producer_decision
    
            # store produced sentences for later lookup (Exp2 looks in Exp1, etc.)
            store_key = data.get("sentence_store_key")
            if isinstance(store_key, dict):
                exp_num = safe_int(store_key.get("exp"), -1)
                prod = safe_int(store_key.get("producer_slot"), 0)
                interp = safe_int(store_key.get("interpreter_slot"), 0)
                cond = clean_str(store_key.get("condition"))
    
                # Store only keys that can be looked up later: interpreter must be in 1..4
                if exp_num >= 1 and prod and interp and cond:
                    k = sentence_key(exp_num, prod, interp, cond)
                    store = player.session.vars.get("sentences_by_key")
                    if not isinstance(store, dict):
                        store = {}
                        player.session.vars["sentences_by_key"] = store
                    store[k] = player.producer_decision
                    logger.info(f"Stored produced sentences key={k}")
    
        elif player.inner_role == INTERPRETER:
            # Normalize interpreter answers so export is stable.
            choices = player.session.vars.get("interpreter_choices") or []
            if isinstance(choices, str):
                choices = [x.strip() for x in choices.split(";") if x.strip()]
            elif not isinstance(choices, list):
                choices = []
            choices = [str(x).strip() for x in choices if str(x).strip()][:4]
            while len(choices) < 4:
                choices.append(f"Option_{len(choices)+1}")
    
            raw = player.interpreter_decision
            data_in = safe_json_loads(raw, None)
    
            labeled = []
    
            # Case A: list aligned with choices
            if isinstance(data_in, list) and (not any(isinstance(x, dict) for x in data_in)):
                for opt, val in zip(choices, data_in[:4]):
                    labeled.append({"option": opt, "answer": normalize_yesno_to_01(val)})
    
            # Case B: list of dicts
            elif isinstance(data_in, list) and any(isinstance(x, dict) for x in data_in):
                by_label = {}
                for d in data_in:
                    if not isinstance(d, dict):
                        continue
                    opt = d.get("option") or d.get("choice") or d.get("label") or d.get("text") or d.get("name")
                    if opt is None:
                        continue
                    opt = str(opt).strip()
                    by_label[opt] = normalize_yesno_to_01(d.get("answer", d.get("value", d.get("selected"))))
    
                if by_label:
                    for opt in choices:
                        labeled.append({"option": opt, "answer": by_label.get(opt, "")})
                else:
                    for i, opt in enumerate(choices):
                        if i >= len(data_in):
                            labeled.append({"option": opt, "answer": ""})
                            continue
                        d = data_in[i] if isinstance(data_in[i], dict) else {}
                        labeled.append({"option": opt, "answer": normalize_yesno_to_01(d.get("answer", d.get("value", d.get("selected"))))})
    
            # Case C: plain string
            else:
                s = "" if raw is None else str(raw).strip()
                parts = [p.strip() for p in re.split(r"[,\;\|\s]+", s) if p.strip()]
                for opt, val in zip(choices, parts[:4]):
                    labeled.append({"option": opt, "answer": normalize_yesno_to_01(val)})
                if not labeled:
                    labeled = [{"option": opt, "answer": ""} for opt in choices]
    
            updates["interpreter_rewards"] = json.dumps(labeled)
    
        # ✅ MUST be here (outside role branches)
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
            return RedirectResponse(Constants.API_ERR_URL, status_code=302)
        return RedirectResponse(STUBURL + str(cc), status_code=302)

def normalize_yesno_to_01(v):
    if v is None:
        return ""
    s = str(v).strip().lower()
    if s in {"yes", "y", "1", "true", "t"}:
        return 1
    if s in {"no", "n", "0", "false", "f"}:
        return 0
    try:
        if int(v) in (0, 1):
            return int(v)
    except Exception:
        pass
    return ""

# ----------------------------------------------------------------------------
# EXPORT
# ----------------------------------------------------------------------------
def extract_sentence_cells(raw_sentences_json):
    """
    Returns 10 cells:
      Sentence_1_1 ... Sentence_5_2

    Rule:
      - Use participant-entered values (or resolved fallback for interpreters)
      - If a sentence or slot is missing → EMPTY CELL
      - Literal "None" string is preserved if participant typed it
    """
    pairs = safe_json_loads(raw_sentences_json, [])
    out = []
    for i in range(5):
        a = ""
        b = ""
        if isinstance(pairs, list) and i < len(pairs) and isinstance(pairs[i], list):
            if len(pairs[i]) >= 1:
                a = clean_str(pairs[i][0])
            if len(pairs[i]) >= 2:
                b = clean_str(pairs[i][1])
        out.extend([a, b])
    return out


def custom_export(players):
    """
    CSV requirements:
      - Sentences exported as 10 raw columns:
          Sentence_1_1, Sentence_1_2, ... , Sentence_5_1, Sentence_5_2
        (empty cells if fewer than 5 sentences)
      - Interpreter answers exported as 4 columns:
          column names EXACTLY equal to the option text shown to participants
          values are 0/1 based on participant input
      - For role == I, if producer_sentences empty/[] -> resolve via sentence_lookup/session store
    """
    from collections import defaultdict

    def resolve_lookup_sentences(item, session_vars):
        lookup = item.get("sentence_lookup")
        if not isinstance(lookup, dict):
            return ""
        src_exp = safe_int(lookup.get("source_exp"), -1)
        prod = safe_int(lookup.get("producer_slot"), 0)
        interp = safe_int(lookup.get("interpreter_slot"), 0)
        cond = clean_str(lookup.get("condition"))
        k = sentence_key(src_exp, prod, interp, cond)
        store = session_vars.get("sentences_by_key") or {}
        return store.get(k) or ""

    def get_interpreter_choices_from_session(any_player):
        ic = any_player.session.vars.get("interpreter_choices") or []
        if isinstance(ic, str):
            choices = [x.strip() for x in ic.split(";") if x.strip()]
        elif isinstance(ic, list):
            choices = [str(x).strip() for x in ic if str(x).strip()]
        else:
            choices = []
        choices = choices[:4]
        while len(choices) < 4:
            choices.append(f"Option_{len(choices)+1}")
        return choices

    def parse_interpreter_answers(raw_interp, choices):
        """
        Returns dict option_text -> 0/1/""
        Supports stored formats:
          A) [1,0,1,0] aligned with choices order
          B) [{"option":"...", "answer":1}, ...]
          C) [{"answer":1}, {"answer":0}, ...] aligned by index
        """
        out = {c: "" for c in choices}
        data = safe_json_loads(raw_interp, [])
        if not isinstance(data, list):
            return out

        # dict-labeled format
        if any(isinstance(x, dict) for x in data):
            # if dicts include option labels, use them
            has_labels = any(
                isinstance(x, dict) and (x.get("option") is not None or x.get("choice") is not None or x.get("label") is not None)
                for x in data
            )
            if has_labels:
                for d in data:
                    if not isinstance(d, dict):
                        continue
                    opt = d.get("option") or d.get("choice") or d.get("label")
                    if opt is None:
                        continue
                    opt = str(opt).strip()
                    if opt in out:
                        out[opt] = normalize_yesno_to_01(d.get("answer", d.get("value", d.get("selected"))))
                return out

            # otherwise assume aligned by index using "answer" field
            for i, c in enumerate(choices):
                if i >= len(data):
                    break
                d = data[i]
                if isinstance(d, dict):
                    out[c] = normalize_yesno_to_01(d.get("answer", d.get("value", d.get("selected"))))
            return out

        # plain list aligned by index
        for i, c in enumerate(choices):
            if i >= len(data):
                break
            out[c] = normalize_yesno_to_01(data[i])
        return out

    demo_keys = [
        "gender",
        "age",
        "handedness",
        "grewUpInCountry",
        "currentlyLivingInCountry",
        "nativeLanguage",
        "nativeLanguageOther",
        "education",
    ]

    # Determine interpreter choice headers once (assume consistent per session)
    any_player = players[0] if players else None
    choice_headers = get_interpreter_choices_from_session(any_player) if any_player else ["Option_1", "Option_2", "Option_3", "Option_4"]

    # HEADER
    yield [
        "session",
        "participant",
        "prolific_id",
        *[f"demo_{k}" for k in demo_keys],
        "exp_num",
        "round",
        "role",
        "producer_id",
        "interpreter_id",
        "condition",
        "item_nr",
        "image",
        "Sentence_1_1",
        "Sentence_1_2",
        "Sentence_2_1",
        "Sentence_2_2",
        "Sentence_3_1",
        "Sentence_3_2",
        "Sentence_4_1",
        "Sentence_4_2",
        "Sentence_5_1",
        "Sentence_5_2",
        *choice_headers,  # EXACT option text
        "rewards_raw",
        "seconds",
        "feedback",
        "excel_row_number_guess",
        "excel_row_index0",
    ]

    # group by participant
    players_by_participant = defaultdict(list)
    for p in players:
        players_by_participant[p.participant.code].append(p)

    for participant_code, participant_players in players_by_participant.items():
        try:
            first_player = participant_players[0]
            prolific_id = first_player.participant.vars.get("prolific_id", "")
            excel_slot = first_player.id_in_subsession

            # demographics
            demo_obj = {}
            try:
                if "demographics" in first_player.participant.vars:
                    demo_obj = safe_json_loads(first_player.participant.vars["demographics"], {})
                else:
                    start_players = [pp for pp in first_player.participant.get_players() if hasattr(pp, "survey_data")]
                    if start_players:
                        demo_obj = safe_json_loads(start_players[0].survey_data, {})
            except Exception:
                demo_obj = {}
            demo_cols = [demo_obj.get(k, "") for k in demo_keys]

            # history
            history = safe_json_loads(first_player.participant.vars.get("batch_history", "[]"), [])
            history.sort(key=lambda x: int(x.get("round_number", 0)))

            # timing + feedback
            timing_map = {}
            feedback_str = ""
            for pp in participant_players:
                if pp.round_number:
                    timing_map[pp.round_number] = pp.decision_seconds or 0
                    if pp.round_number == Constants.num_rounds and pp.feedback:
                        feedback_str = pp.feedback

            for item in history:
                rnd = int(item.get("round_number", 0))
                if rnd < 1 or rnd > Constants.num_rounds:
                    continue

                my_role = item.get("role", "")
                partner_id = item.get("partner_id", 0)

                if my_role == PRODUCER:
                    prod_id = excel_slot
                    interp_id = partner_id
                else:
                    prod_id = partner_id
                    interp_id = excel_slot

                exp_num = item.get("exp", "")

                # sentences: participant entered for producer; for interpreter we export the shown sentences
                raw_sentences = item.get("producer_sentences") or item.get("sentences") or ""
                if my_role == INTERPRETER:
                    if (not raw_sentences) or (isinstance(raw_sentences, str) and raw_sentences.strip() in {"", "[]"}):
                        resolved = resolve_lookup_sentences(item, first_player.session.vars)
                        if resolved:
                            raw_sentences = resolved

                sentence_cells = extract_sentence_cells(raw_sentences)

                # interpreter answers -> 4 option columns
                raw_interp = item.get("interpreter_rewards") or item.get("rewards") or ""
                ans_map = parse_interpreter_answers(raw_interp, choice_headers)
                interp_cols = [ans_map.get(opt, "") for opt in choice_headers]

                seconds = timing_map.get(rnd, 0)

                yield [
                    first_player.session.code,
                    participant_code,
                    prolific_id,
                    *demo_cols,
                    exp_num,
                    rnd,
                    my_role,
                    prod_id,
                    interp_id,
                    item.get("condition", ""),
                    item.get("item_nr", ""),
                    item.get("image", ""),
                    *sentence_cells,
                    *interp_cols,
                    raw_interp,
                    seconds,
                    feedback_str if rnd == Constants.num_rounds else "",
                    item.get("excel_row_number_guess", ""),
                    item.get("excel_row_index0", ""),
                ]

        except Exception:
            continue


page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
