from otree.api import *
import json
import logging
import time
import re
from starlette.responses import RedirectResponse
from .utils import STUBURL

logger = logging.getLogger("benzapp.img_desc")

PRODUCER = "P"
INTERPRETER = "I"

class Constants(BaseConstants):
    name_in_url = "img_desc"
    players_per_group = None
    num_rounds = 80

    PLACEMENT_ERR = "ERROR_BATCH_PLACEMENT"
    API_ERR = "API_ERROR"
    FALLBACK_URL = STUBURL + PLACEMENT_ERR
    API_ERR_URL = STUBURL + API_ERR

    BAD_PROLIFIC_STATUSES = {"TIMED-OUT"}

# ----------------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------------
class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    batch_history = models.LongStringField(initial="[]")

    inner_role = models.StringField(blank=True)
    faulty = models.BooleanField(initial=False)
    feedback = models.LongStringField(label="")

    producer_decision = models.LongStringField(blank=True)
    interpreter_decision = models.LongStringField(blank=True)

    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)

    full_return_url = models.StringField(blank=True)

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
            for item in hist:
                if int(item.get("round_number", 0)) == rnd:
                    item.update(updates)
                    self.batch_history = json.dumps(hist)
                    return
        except Exception:
            pass

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
        Lookup sentences by (source_exp, producer_slot, interpreter_slot, condition).
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
        Sentences shown to BOTH roles:
          1) if batch item has producer_sentences and it's not empty/[]
          2) else resolve via sentence_lookup
        Returns: list (decoded JSON)
        """
        data = self.get_current_batch_data()
        if not data:
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
        Render sentence pairs using prefix/suffixes.
        Produces a list of strings for the template.
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
                    val_str = clean_str(val)
                    if not val_str:
                        val_str = "None"
                    parts.append(val_str)
                    if suf:
                        parts.append(str(suf).strip())

                if len(sentence) > len(suffixes):
                    for extra in sentence[len(suffixes):]:
                        extra_str = clean_str(extra)
                        if not extra_str:
                            extra_str = "None"
                        parts.append(extra_str)

                res.append(" ".join([p for p in parts if str(p).strip() != ""]).strip())

            return res
        except Exception as e:
            logger.error(f"Error in get_full_sentences: {e}")
            return []


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


def participant_completed_app(participant, app_name: str) -> bool:
    return bool(participant.vars.get(f"{app_name}_completed", False))


def mark_participant_completed_app(participant, app_name: str):
    participant.vars[f"{app_name}_completed"] = True


def reset_this_app_for_participant(participant):
    """
    Hard-reset THIS app for this participant (used ONLY when Prolific webhook says TIMED-OUT).
    """
    participant.vars.pop("batch_history", None)
    participant.vars.pop("needed_sentence_keys", None)
    participant.vars.pop("missing_sentence_keys", None)
    participant.vars.pop("exp_target", None)
    participant.vars.pop("local_slot", None)
    participant.vars.pop("cohort_index", None)
    participant.vars.pop("_needed_sentence_keys_set", None)


    for pp in participant.get_players():
        if pp.subsession._meta.app_config.name != Constants.name_in_url:
            continue
        pp.batch_history = "[]"
        pp.inner_role = ""
        pp.faulty = False
        pp.feedback = ""
        pp.producer_decision = ""
        pp.interpreter_decision = ""
        pp.start_decision_time = 0
        pp.end_decision_time = 0
        pp.decision_seconds = 0

    participant.vars.pop(f"{Constants.name_in_url}_completed", None)
    try:
        participant._index_in_pages = 0
    except Exception:
        pass


# ----------------------------------------------------------------------------
# COHORT HELPERS
# ----------------------------------------------------------------------------
def cohort_size(session) -> int:
    return safe_int(session.config.get("cohort_size", 4), 4)


def cohort_index_from_pid(pid: int, csize: int) -> int:
    return (pid - 1) // csize


def local_slot_from_pid(pid: int, csize: int) -> int:
    return ((pid - 1) % csize) + 1


def exp_for_pid(pid: int, csize: int) -> int:
    return cohort_index_from_pid(pid, csize) + 1


def cohort_slot_to_pid(pid: int, csize: int, max_pid: int) -> dict:
    cidx = cohort_index_from_pid(pid, csize)
    base = cidx * csize
    m = {}
    for s in range(1, csize + 1):
        real_pid = base + s
        m[s] = real_pid if 1 <= real_pid <= max_pid else 0
    return m


def sentences_for_line(row: dict, exp_target: int):
    producer_slot = safe_int(row.get("Producer"), 0)
    interpreter_slot = safe_int(row.get("Interpreter"), 0)
    cond = clean_str(row.get("Condition"))

    if producer_slot == 0:
        extracted = extract_sentences_from_row(row)
        if extracted and extracted.strip() != "[]":
            return extracted, None
        return "[]", None

    return "[]", {
        "source_exp": exp_target - 1,
        "producer_slot": producer_slot,
        "interpreter_slot": interpreter_slot,
        "condition": cond,
    }


def render_full_sentences_from_json(raw_sentences_json, prefix, suffixes):
    pairs = safe_json_loads(raw_sentences_json, [])
    if not isinstance(pairs, list):
        return []

    prefix = clean_str(prefix)
    suffixes = suffixes or []

    out = []
    for pair in pairs:
        if not isinstance(pair, list):
            continue
        parts = []
        if prefix:
            parts.append(prefix)

        for val, suf in zip(pair, suffixes):
            v = clean_str(val) or "None"
            parts.append(v)
            if suf:
                parts.append(str(suf).strip())

        if len(pair) > len(suffixes):
            for extra in pair[len(suffixes):]:
                v = clean_str(extra) or "None"
                parts.append(v)

        out.append(" ".join([p for p in parts if str(p).strip()]).strip())

    return out


def required_sentence_keys_for_player(player) -> set:
    needed = set()
    hist = safe_json_loads(player.batch_history, [])
    for item in hist:
        lookup = item.get("sentence_lookup")
        if not isinstance(lookup, dict):
            continue
        src_exp = safe_int(lookup.get("source_exp"), -1)
        prod = safe_int(lookup.get("producer_slot"), 0)
        interp = safe_int(lookup.get("interpreter_slot"), 0)
        cond = clean_str(lookup.get("condition"))
        if src_exp >= 0 and prod and interp and cond:
            needed.add(sentence_key(src_exp, prod, interp, cond))
    return needed

# ---- NEW: dynamic cohort/slot assignment (replacement-safe) ----
def _cohort_state(session) -> dict:
    st = session.vars.get("cohorts")
    if not isinstance(st, dict):
        st = {}
        session.vars["cohorts"] = st
    return st


def _get_or_create_cohort(session, exp_num: int) -> dict:
    st = _cohort_state(session)
    k = str(int(exp_num))
    c = st.get(k)
    if not isinstance(c, dict):
        c = {"slots": {}, "complete": False}
        st[k] = c
    if "slots" not in c or not isinstance(c["slots"], dict):
        c["slots"] = {}
    if "complete" not in c:
        c["complete"] = False
    return c


def assign_slot_if_needed(player):
    """
    Assign participant to the lowest available slot in the lowest incomplete cohort.
    This avoids pid-based gaps and supports replacements.
    """
    p = player.participant

    # already assigned?
    exp_target = p.vars.get("exp_target")
    local_slot = p.vars.get("local_slot")
    if exp_target and local_slot:
        return int(exp_target), int(local_slot)

    session = player.session
    csize = cohort_size(session)

    # find earliest cohort with a free slot
    exp_num = 1
    while True:
        cohort = _get_or_create_cohort(session, exp_num)
        slots = cohort["slots"]

        # available slot = 1..csize not yet filled
        taken = set(int(s) for s in slots.keys())
        free = [s for s in range(1, csize + 1) if s not in taken]

        if free:
            slot = free[0]
            slots[str(slot)] = p.code  # store participant code
            p.vars["exp_target"] = exp_num
            p.vars["local_slot"] = slot
            p.vars["cohort_index"] = exp_num - 1
            return exp_num, slot

        exp_num += 1


def free_slot_for_participant(session, participant_code: str):
    """
    If a participant is timed-out/replaced, free their slot so the next joiner can take it.
    Also remove them from the cohort's "done" map so completion can't be faked.
    """
    st = _cohort_state(session)

    for exp_k, cohort in st.items():
        slots = cohort.get("slots", {}) or {}

        for s_k, pcode in list(slots.items()):
            if pcode == participant_code:
                # remove "done" marker if present
                done = cohort.get("done")
                if isinstance(done, dict):
                    done.pop(participant_code, None)

                # free the slot
                del slots[s_k]

                # mark cohort as not complete (it now has a missing participant)
                cohort["complete"] = False
                return

def cohort_filled(session, exp_num: int) -> bool:
    csize = cohort_size(session)
    cohort = _get_or_create_cohort(session, exp_num)
    return len(cohort.get("slots", {})) >= csize

def mark_cohort_participant_complete(session, exp_num: int, participant_code: str):
    cohort = _get_or_create_cohort(session, exp_num)

    done = cohort.get("done")
    if not isinstance(done, dict):
        done = {}
        cohort["done"] = done

    done[participant_code] = True

    # Only complete if fully filled AND all slots finished
    csize = cohort_size(session)
    slots = cohort.get("slots", {}) or {}

    if len(slots) >= csize:
        if all(done.get(pcode) for pcode in slots.values()):
            cohort["complete"] = True


def cohort_complete(session, exp_num: int) -> bool:
    cohort = _get_or_create_cohort(session, exp_num)
    return bool(cohort.get("complete"))

# ----------------------------------------------------------------------------
# NEW: Gate page (only blocks if webhook marked them timed-out)
# ----------------------------------------------------------------------------
class ProlificStatusGate(Page):
    @staticmethod
    def is_displayed(player):
        status = clean_str(player.participant.vars.get("prolific_submission_status", ""))
        return status in Constants.BAD_PROLIFIC_STATUSES

    def get(self):
        try:
            free_slot_for_participant(self.player.session, self.player.participant.code)
            reset_this_app_for_participant(self.player.participant)  # <-- add this
        except Exception:
            pass
        return RedirectResponse(Constants.FALLBACK_URL, status_code=302)


# ----------------------------------------------------------------------------
# SESSION CREATION
# ----------------------------------------------------------------------------
def creating_session(subsession: Subsession):
    session = subsession.session

    # Only run once
    if subsession.round_number != 1:
        return

    try:
        filename = session.config.get("filename")
        if not filename:
            raise RuntimeError("No filename in session config")

        logger.info(f"Starting session creation with filename: {filename}")

        # ------------------------------------------------------------------
        # Load Excel
        # ------------------------------------------------------------------
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

        rows = (
            raw_data.to_dict(orient="records")
            if hasattr(raw_data, "to_dict")
            else list(raw_data or [])
        )

        # ------------------------------------------------------------------
        # Normalize settings
        # ------------------------------------------------------------------
        clean_settings = {}
        for k, v in settings.items():
            if isinstance(v, str):
                clean_settings[normalize_key(k)] = clean_str(v)

        # ------------------------------------------------------------------
        # Store session configuration
        # ------------------------------------------------------------------
        s3_raw = clean_settings.get("s3path") or clean_settings.get("s3path_base") or ""
        session.vars["s3path_base"] = fix_s3_url(s3_raw)
        session.vars["extension"] = clean_settings.get("extension") or "png"
        session.vars["prefix"] = clean_settings.get("prefix") or ""
        session.vars["interpreter_title"] = (
            clean_settings.get("interpreter_title") or "Buy medals:"
        )
        session.vars["caseflag"] = _truthy(clean_settings.get("caseflag"))

        session.vars["instructions_url"] = (
            session.config.get("instructions_url")
            or clean_settings.get("instructions_url")
            or "https://google.com"
        )
        session.vars["introduction_text"] = (
            session.config.get("introduction_text") or ""
        )
        session.vars["doc_link"] = session.config.get("doc_link") or ""

        if session.config.get("completion_code"):
            session.vars["completion_code"] = str(session.config["completion_code"])

        session.vars["allowed_values"] = settings.get("allowed_values", []) or []
        session.vars["allowed_regex"] = settings.get("allowed_regex", []) or []

        # ------------------------------------------------------------------
        # Suffixes
        # ------------------------------------------------------------------
        suffixes = []
        for i in range(1, 11):
            v = clean_settings.get(f"suffix_{i}")
            if v:
                suffixes.append(v)
        session.vars["suffixes"] = suffixes

        # ------------------------------------------------------------------
        # Interpreter choices
        # ------------------------------------------------------------------
        ic = settings.get("interpreter_choices")
        if isinstance(ic, str):
            session.vars["interpreter_choices"] = [
                x.strip() for x in ic.split(";") if x.strip()
            ]
        elif isinstance(ic, list):
            session.vars["interpreter_choices"] = ic
        else:
            session.vars["interpreter_choices"] = []

        # ------------------------------------------------------------------
        # Store raw rows (needed for dynamic schedule building)
        # ------------------------------------------------------------------
        session.vars["excel_rows"] = rows

        # ------------------------------------------------------------------
        # Precompute rows grouped by experiment
        # This allows fast rebuilding of schedules for replacements
        # ------------------------------------------------------------------
        rows_by_exp = {}
        for idx0, r in enumerate(rows):
            exp_num = get_exp_num(r)
            if exp_num <= 0:
                continue

            rr = dict(r)
            rr["_idx0"] = idx0
            rr["_excel_row_number"] = idx0 + 2

            rows_by_exp.setdefault(exp_num, []).append(rr)

        session.vars["rows_by_exp"] = rows_by_exp

        # ------------------------------------------------------------------
        # Preload Exp=0 sentence store (baseline sentences)
        # ------------------------------------------------------------------
        session.vars["sentences_by_key"] = {}

        for r in rows:
            if get_exp_num(r) != 0:
                continue

            prod = safe_int(r.get("Producer"), 0)
            interp = safe_int(r.get("Interpreter"), 0)
            cond = clean_str(r.get("Condition"))

            sent = extract_sentences_from_row(r)
            if sent and sent.strip() != "[]":
                session.vars["sentences_by_key"][
                    sentence_key(0, prod, interp, cond)
                ] = sent

        # ------------------------------------------------------------------
        # Initialize cohort tracking state
        # (no assignment happens here!)
        # ------------------------------------------------------------------
        if not isinstance(session.vars.get("cohorts"), dict):
            session.vars["cohorts"] = {}

        session.vars["cohort_size"] = cohort_size(session)

        logger.info("Session creation completed successfully.")

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


class WaitForPrevExperiment(Page):
    template_name = "img_desc/WaitForPrevExperiment.html"

    @staticmethod
    def is_displayed(player):
        csize = cohort_size(player.session)

        # ensure assignment exists (important on refresh)
        assign_slot_if_needed(player)

        if player.batch_history in {"", "[]"}:
            if "batch_history" in player.participant.vars and player.participant.vars["batch_history"] not in {"", "[]"}:
                player.batch_history = player.participant.vars["batch_history"]
            else:
                build_batch_history_for_player(player)
        
                
        exp_target = int(player.participant.vars.get("exp_target") or 1)
        if exp_target <= 1:
            return False
        
        # also block until previous cohort is filled (this is the “replacement-safe” part)
        if not cohort_complete(player.session, exp_target - 1):
            player.participant.vars["needed_sentence_keys"] = 0
            player.participant.vars["missing_sentence_keys"] = 0
            return True

        if (player.batch_history == "[]" or not player.batch_history) and "batch_history" in player.participant.vars:
            player.batch_history = player.participant.vars["batch_history"]

        # Cache needed keys so we don't re-derive from history every refresh
        needed = player.participant.vars.get("_needed_sentence_keys_set")
        if not isinstance(needed, list):
            needed_set = required_sentence_keys_for_player(player)
            player.participant.vars["_needed_sentence_keys_set"] = list(needed_set)
            needed = list(needed_set)

        if not needed:
            return False

        store = player.session.vars.get("sentences_by_key") or {}
        missing = [k for k in needed if k not in store]

        player.participant.vars["needed_sentence_keys"] = len(needed)
        player.participant.vars["missing_sentence_keys"] = len(missing)

        return len(missing) > 0

    @staticmethod
    def vars_for_template(player):
        return dict(
            needed=player.participant.vars.get("needed_sentence_keys", 0),
            missing=player.participant.vars.get("missing_sentence_keys", 0),
        )

def build_batch_history_for_player(player):
    """
    Build (or rebuild) this participant's 80-round schedule based on:
      - dynamically assigned exp_target + local_slot (participant.vars)
      - Excel rows cached in session.vars["rows_by_exp"]
    Safe for replacements: if the participant is assigned to an existing cohort slot,
    they get the exact schedule for that exp + slot.

    Writes:
      - player.batch_history
      - participant.vars["batch_history"]
    """
    session = player.session
    p = player.participant

    # Ensure assignment exists (replacement-safe)
    exp_target, local_slot = assign_slot_if_needed(player)

    rows_by_exp = session.vars.get("rows_by_exp") or {}
    exp_rows = rows_by_exp.get(int(exp_target), [])
    if not exp_rows:
        logger.error(f"No rows found for Exp={exp_target} (participant={p.code}).")
        player.batch_history = "[]"
        p.vars["batch_history"] = player.batch_history
        return

    # Group exp rows by Round (as in your original code)
    rounds_map = {}
    for rr in exp_rows:
        rnum = safe_int(rr.get("Round"), 0)
        if rnum >= 1:
            rounds_map.setdefault(rnum, []).append(rr)

    round_numbers = sorted(rounds_map.keys())
    if not round_numbers:
        logger.error(f"No valid Round>=1 rows for Exp={exp_target} (participant={p.code}).")
        player.batch_history = "[]"
        p.vars["batch_history"] = player.batch_history
        return

    # Build slot->participant_code map for this cohort (for partner lookup)
    # Note: partner_id in your UI is informational; we store participant codes in cohort slots.
    cohort = _get_or_create_cohort(session, int(exp_target))
    slots = cohort.get("slots", {}) or {}
    slot_to_pcode = {safe_int(k, 0): v for k, v in slots.items() if safe_int(k, 0) > 0}

    final_history = []
    otree_round_counter = 1

    for rnum in round_numbers:
        block = rounds_map[rnum]

        # Keep your original caps: 3 producer trials, 5 interpreter trials per Round block
        p_hits = [row for row in block if safe_int(row.get("Producer"), 0) == local_slot][:3]
        i_hits = [row for row in block if safe_int(row.get("Interpreter"), 0) == local_slot][:5]

        # ------------------------------------------------------------
        # PRODUCER ITEMS
        # ------------------------------------------------------------
        for row in p_hits:
            if otree_round_counter > Constants.num_rounds:
                break

            producer_slot = safe_int(row.get("Producer"), 0)
            interpreter_slot = safe_int(row.get("Interpreter"), 0)
            cond = clean_str(row.get("Condition"))
            item_nr = clean_str(row.get("Item.Nr"))
            image_raw = clean_str(row.get("Item"))

            picked = image_raw or "d-A-B-BC-3"
            if producer_slot != 0 and not is_valid_real_image(picked):
                picked = image_raw or "d-A-B-BC-3"

            # Partner id: store partner participant_code if available (optional)
            partner_code = slot_to_pcode.get(interpreter_slot, "") if interpreter_slot else ""

            ps, lookup = sentences_for_line(row, int(exp_target))

            final_history.append(
                dict(
                    round_number=otree_round_counter,
                    role=PRODUCER,
                    partner_id=partner_code,
                    exp=int(exp_target),
                    round_in_excel=rnum,
                    trial=safe_int(row.get("Trial"), 0),
                    condition=cond,
                    item_nr=item_nr,
                    image=picked,
                    producer_sentences=ps,
                    sentence_lookup=lookup,
                    interpreter_rewards="",
                    producer_slot=producer_slot,
                    interpreter_slot=interpreter_slot,
                    sentence_store_key=dict(
                        exp=int(exp_target),
                        producer_slot=producer_slot,
                        interpreter_slot=interpreter_slot,
                        condition=cond,
                    ),
                    excel_row_index0=row.get("_idx0", ""),
                    excel_row_number_guess=row.get("_excel_row_number", ""),
                )
            )
            otree_round_counter += 1

        # ------------------------------------------------------------
        # INTERPRETER ITEMS
        # ------------------------------------------------------------
        for row in i_hits:
            if otree_round_counter > Constants.num_rounds:
                break

            producer_slot = safe_int(row.get("Producer"), 0)
            interpreter_slot = safe_int(row.get("Interpreter"), 0)
            cond = clean_str(row.get("Condition"))
            item_nr = clean_str(row.get("Item.Nr"))
            image_raw = clean_str(row.get("Item"))

            picked = image_raw or ("NA_x" if producer_slot == 0 else "d-A-B-BC-3")
            if producer_slot != 0 and not is_valid_real_image(picked):
                picked = image_raw or "d-A-B-BC-3"

            # Partner id: store partner participant_code if available (optional)
            partner_code = slot_to_pcode.get(producer_slot, "") if producer_slot not in {0, 9} else ""

            ps, lookup = sentences_for_line(row, int(exp_target))

            final_history.append(
                dict(
                    round_number=otree_round_counter,
                    role=INTERPRETER,
                    partner_id=partner_code,
                    exp=int(exp_target),
                    round_in_excel=rnum,
                    trial=safe_int(row.get("Trial"), 0),
                    condition=cond,
                    item_nr=item_nr,
                    image=picked,
                    producer_sentences=ps,
                    sentence_lookup=lookup,
                    interpreter_rewards="",
                    producer_slot=producer_slot,
                    interpreter_slot=interpreter_slot,
                    excel_row_index0=row.get("_idx0", ""),
                    excel_row_number_guess=row.get("_excel_row_number", ""),
                )
            )
            otree_round_counter += 1

        if otree_round_counter > Constants.num_rounds:
            break

    final_history = final_history[: Constants.num_rounds]
    if len(final_history) != Constants.num_rounds:
        logger.warning(
            f"Schedule len mismatch for participant={p.code} exp={exp_target}: "
            f"got={len(final_history)} expected={Constants.num_rounds}"
        )

    player.batch_history = json.dumps(final_history)
    p.vars.pop("_needed_sentence_keys_set", None)
    p.vars["batch_history"] = player.batch_history

class Q(Page):
    form_model = "player"

    @staticmethod
    def is_displayed(player):
        if player.round_number > Constants.num_rounds:
            return False
    
        assign_slot_if_needed(player)
    
        if player.batch_history in {"", "[]"}:
            if "batch_history" in player.participant.vars and player.participant.vars["batch_history"] not in {"", "[]"}:
                player.batch_history = player.participant.vars["batch_history"]
            else:
                build_batch_history_for_player(player)

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

        d = player.get_current_batch_data()

        raw = d.get("producer_sentences")
        if not raw or (isinstance(raw, str) and raw.strip() in {"", "[]"}):
            resolved = player._resolve_sentence_lookup(d.get("sentence_lookup"))
            if resolved and isinstance(resolved, str) and resolved.strip() not in {"", "[]"}:
                d = d.copy()
                d["producer_sentences"] = resolved

        return dict(
            d=d,
            allowed_values=player.session.vars.get("allowed_values", []),
            allowed_regex=player.session.vars.get("allowed_regex", []),
            suffixes=player.session.vars.get("suffixes", []),
            prefix=player.session.vars.get("prefix", ""),
            interpreter_choices=interpreter_choices,
            interpreter_title=player.session.vars.get("interpreter_title") or "Buy medals:",
            instructions_url=player.session.vars.get("instructions_url"),
            introduction_text=player.session.vars.get("introduction_text", ""),
            doc_link=player.session.vars.get("doc_link", ""),
            server_image_url=player.get_image_url(),
            caseflag=player.session.vars.get("caseflag"),
            full_sentences=render_full_sentences_from_json(
                d.get("producer_sentences", "[]"),
                player.session.vars.get("prefix", ""),
                player.session.vars.get("suffixes", []),
            ),
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

            store_key = data.get("sentence_store_key")
            if isinstance(store_key, dict):
                exp_num = safe_int(store_key.get("exp"), -1)
                prod = safe_int(store_key.get("producer_slot"), 0)
                interp = safe_int(store_key.get("interpreter_slot"), 0)
                cond = clean_str(store_key.get("condition"))
                if exp_num >= 1 and prod and interp and cond:
                    k = sentence_key(exp_num, prod, interp, cond)
                    store = player.session.vars.get("sentences_by_key")
                    if not isinstance(store, dict):
                        store = {}
                        player.session.vars["sentences_by_key"] = store
                    store[k] = player.producer_decision

        elif player.inner_role == INTERPRETER:
            choices = player.session.vars.get("interpreter_choices") or []
            if isinstance(choices, str):
                choices = [x.strip() for x in choices.split(";") if x.strip()]
            elif not isinstance(choices, list):
                choices = []
            choices = [str(x).strip() for x in choices if str(x).strip()][:4]
            while len(choices) < 4:
                choices.append(f"Option_{len(choices)+1}")

            raw = player.interpreter_decision
            parsed = safe_json_loads(raw, None)

            labeled = []

            if isinstance(parsed, list) and (not any(isinstance(x, dict) for x in parsed)):
                for opt, val in zip(choices, parsed[:4]):
                    labeled.append({"option": opt, "answer": normalize_yesno_to_01(val)})

            elif isinstance(parsed, list) and any(isinstance(x, dict) for x in parsed):
                by_label = {}
                for d in parsed:
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
                        if i >= len(parsed):
                            labeled.append({"option": opt, "answer": ""})
                            continue
                        d = parsed[i] if isinstance(parsed[i], dict) else {}
                        labeled.append({"option": opt, "answer": normalize_yesno_to_01(d.get("answer", d.get("value", d.get("selected"))))})

            else:
                s = "" if raw is None else str(raw).strip()
                parts = [p.strip() for p in re.split(r"[,;\|\s]+", s) if p.strip()]
                for opt, val in zip(choices, parts[:4]):
                    labeled.append({"option": opt, "answer": normalize_yesno_to_01(val)})

            if not labeled:
                labeled = [{"option": opt, "answer": ""} for opt in choices]

            updates["interpreter_rewards"] = json.dumps(labeled)

        if updates:
            player.update_current_batch_data(updates)
            player.participant.vars["batch_history"] = player.batch_history


class Feedback(Page):
    form_model = "player"
    form_fields = ["feedback"]

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.num_rounds

    @staticmethod
    def before_next_page(player, timeout_happened):
        mark_participant_completed_app(player.participant, Constants.name_in_url)

        # mark cohort completion
        exp_target = int(player.participant.vars.get("exp_target") or 1)
        mark_cohort_participant_complete(player.session, exp_target, player.participant.code)
        
class FinalForProlific(Page):
    @staticmethod
    def is_displayed(player):
        return player.session.config.get("for_prolific") and player.round_number == Constants.num_rounds

    def get(self):
        cc = self.player.session.vars.get("completion_code") or self.player.session.config.get("completion_code")
        if not cc:
            return RedirectResponse(Constants.API_ERR_URL, status_code=302)
        return RedirectResponse(STUBURL + str(cc), status_code=302)


# ----------------------------------------------------------------------------
# EXPORT (unchanged)
# ----------------------------------------------------------------------------
def extract_sentence_cells(raw_sentences_json):
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
    Memory-safe streaming export.

    Key design principles:
    - Sort at DB level, never in Python (avoids materializing full queryset)
    - Process one participant at a time, flush immediately
    - Read batch_history from player DB field (not participant.vars blob)
    - No participant.get_players() calls
    - session.vars fetched once per session, cached locally
    - All helpers inline or reused from module scope
    """

    # ------------------------------------------------------------------
    # Helpers (local, no module-level state)
    # ------------------------------------------------------------------
    def _get_choices(session_vars):
        ic = session_vars.get("interpreter_choices") or []
        if isinstance(ic, str):
            choices = [x.strip() for x in ic.split(";") if x.strip()]
        elif isinstance(ic, list):
            choices = [str(x).strip() for x in ic if str(x).strip()]
        else:
            choices = []
        choices = choices[:4]
        while len(choices) < 4:
            choices.append(f"Option_{len(choices) + 1}")
        return choices

    def _resolve_lookup(item, sentences_by_key):
        lookup = item.get("sentence_lookup")
        if not isinstance(lookup, dict):
            return ""
        src_exp = safe_int(lookup.get("source_exp"), -1)
        prod = safe_int(lookup.get("producer_slot"), 0)
        interp = safe_int(lookup.get("interpreter_slot"), 0)
        cond = clean_str(lookup.get("condition"))
        k = sentence_key(src_exp, prod, interp, cond)
        return sentences_by_key.get(k) or ""

    def _parse_interp_answers(raw_interp, choices):
        out = {c: "" for c in choices}
        data = safe_json_loads(raw_interp, [])
        if not isinstance(data, list):
            return out
        for d in data:
            if not isinstance(d, dict):
                continue
            opt = str(d.get("option", "")).strip()
            if opt in out:
                out[opt] = normalize_yesno_to_01(d.get("answer"))
        return out

    def _emit_participant(bucket, session_vars, choice_headers, demo_keys):
        """
        Yield export rows for a single participant.
        bucket: list of player objects for this participant (all rounds, sorted by round_number)
        """
        if not bucket:
            return

        first = bucket[0]
        participant = first.participant
        prolific_id = participant.vars.get("prolific_id", "")

        # -- Demographics: only from participant.vars (no get_players() fallback) --
        demo_obj = {}
        try:
            raw_demo = participant.vars.get("demographics")
            if raw_demo:
                demo_obj = safe_json_loads(raw_demo, {})
        except Exception:
            demo_obj = {}
        demo_cols = [demo_obj.get(k, "") for k in demo_keys]

        # -- Build timing map and grab feedback from DB fields --
        timing_map = {}
        feedback_str = ""
        for pp in bucket:
            rn = pp.round_number or 0
            if rn:
                timing_map[rn] = pp.decision_seconds or 0
            if rn == Constants.num_rounds and pp.feedback:
                feedback_str = pp.feedback

        # -- Read batch_history from the round-1 player's DB field --
        # (batch_history is written in round 1 and carried via participant.vars;
        #  we prefer the DB field to avoid loading the full participant.vars blob)
        sentences_by_key = session_vars.get("sentences_by_key") or {}

        round1 = next((pp for pp in bucket if pp.round_number == 1), first)
        raw_hist = round1.batch_history
        if not raw_hist or raw_hist in {"", "[]"}:
            # fall back to participant.vars only if the DB field is empty
            raw_hist = participant.vars.get("batch_history", "[]")

        history = safe_json_loads(raw_hist, [])

        session_code = first.session.code
        participant_code = participant.code

        for item in history:
            rnd = int(item.get("round_number", 0))
            if rnd < 1 or rnd > Constants.num_rounds:
                continue

            my_role = item.get("role", "")
            prod_id = safe_int(item.get("producer_slot"), 0)
            interp_id = safe_int(item.get("interpreter_slot"), 0)
            exp_num = item.get("exp", "")

            # -- Sentences --
            raw_sentences = item.get("producer_sentences") or item.get("sentences") or ""
            if not raw_sentences or (
                isinstance(raw_sentences, str) and raw_sentences.strip() in {"", "[]"}
            ):
                resolved = _resolve_lookup(item, sentences_by_key)
                if resolved:
                    raw_sentences = resolved

            sentence_cells = extract_sentence_cells(raw_sentences)

            # -- Interpreter answers --
            raw_interp = item.get("interpreter_rewards") or item.get("rewards") or ""
            ans_map = _parse_interp_answers(raw_interp, choice_headers)
            interp_cols = [ans_map.get(opt, "") for opt in choice_headers]

            seconds = timing_map.get(rnd, 0)

            yield [
                session_code,
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

    # ------------------------------------------------------------------
    # Demo keys
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Sort at DB level — critical to avoid materializing the full list
    # ------------------------------------------------------------------
    players = players.select_related("participant", "session").order_by(
        "participant__code", "round_number"
    )

    # ------------------------------------------------------------------
    # Peek at first player to get choice headers (needed for header row)
    # ------------------------------------------------------------------
    first_player = None
    choice_headers = ["Option_1", "Option_2", "Option_3", "Option_4"]
    session_vars_cache = {}  # session.code -> session.vars

    for p in players:
        first_player = p
        sv = session_vars_cache.setdefault(p.session.code, p.session.vars)
        choice_headers = _get_choices(sv)
        break

    # ------------------------------------------------------------------
    # Header row
    # ------------------------------------------------------------------
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
        "Sentence_1_1", "Sentence_1_2",
        "Sentence_2_1", "Sentence_2_2",
        "Sentence_3_1", "Sentence_3_2",
        "Sentence_4_1", "Sentence_4_2",
        "Sentence_5_1", "Sentence_5_2",
        *choice_headers,
        "rewards_raw",
        "seconds",
        "feedback",
        "excel_row_number_guess",
        "excel_row_index0",
    ]

    if first_player is None:
        return

    # ------------------------------------------------------------------
    # Stream: accumulate one participant at a time, flush immediately
    # ------------------------------------------------------------------
    # Re-query to restart iteration from the beginning (we consumed one row above)
    players = players.select_related("participant", "session").order_by(
        "participant__code", "round_number"
    )

    current_code = None
    bucket = []

    for p in players:
        code = p.participant.code
        sv = session_vars_cache.setdefault(p.session.code, p.session.vars)

        if current_code is None:
            current_code = code

        if code != current_code:
            # Flush previous participant immediately
            yield from _emit_participant(bucket, sv, choice_headers, demo_keys)
            bucket = []
            current_code = code

        bucket.append(p)

    # Flush the last participant
    if bucket:
        last_sv = session_vars_cache.get(bucket[0].session.code, {})
        yield from _emit_participant(bucket, last_sv, choice_headers, demo_keys)

page_sequence = [ProlificStatusGate, FaultyCatcher, WaitForPrevExperiment, Q, Feedback, FinalForProlific]
