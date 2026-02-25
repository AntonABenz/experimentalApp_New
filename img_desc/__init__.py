# img_desc/__init__.py
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

    BAD_PROLIFIC_STATUSES = {"TIMED-OUT", "RETURNED", "REJECTED"}


# ----------------------------------------------------------------------------
# MODELS
# ----------------------------------------------------------------------------
class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    inner_role = models.StringField(blank=True)
    faulty = models.BooleanField(initial=False)
    feedback = models.LongStringField(label="", blank=True)
    completed_experiment = models.BooleanField(initial=False)
    producer_decision = models.LongStringField(blank=True)
    interpreter_decision = models.LongStringField(blank=True)

    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)

    full_return_url = models.StringField(blank=True)

    # ---- schedule access (DB-backed) ----
    def get_current_batch_data(self) -> dict:
        rn = int(self.round_number or 0)
        item = get_schedule_item(self, self.participant.code, rn)
        if not item:
            return {}
        return safe_json_loads(item.data, {}) or {}

    def update_current_batch_data(self, updates: dict):
        rn = int(self.round_number or 0)
        item = get_schedule_item(self, self.participant.code, rn)
        if not item:
            return
        d = safe_json_loads(item.data, {}) or {}
        if not isinstance(d, dict):
            d = {}
        d.update(updates or {})
        item.data = json.dumps(d)
        try:
            item.save()  # IMPORTANT: persist ExtraModel update
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
        DB-backed SentenceStore.
        """
        if not lookup or not isinstance(lookup, dict):
            return "[]"

        src_exp = safe_int(lookup.get("source_exp"), -1)
        prod = safe_int(lookup.get("producer_slot"), 0)
        interp = safe_int(lookup.get("interpreter_slot"), 0)
        cond = clean_str(lookup.get("condition"))

        key = sentence_key(src_exp, prod, interp, cond)
        raw = get_sentence(self, key)

        if not raw:
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
          1) if schedule item has producer_sentences and it's not empty/[]
          2) else resolve via sentence_lookup
        Returns list (decoded JSON)
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


# ----------------------------------------------------------------------------
# DB BACKING (ExtraModel)
# ----------------------------------------------------------------------------
class TrialRow(ExtraModel):
    """
    Persisted view of Excel rows (slim).
    """
    subsession = models.Link(Subsession)  # anchor to round-1 subsession
    exp_num = models.IntegerField()
    round_in_excel = models.IntegerField()
    trial = models.IntegerField()
    producer_slot = models.IntegerField()
    interpreter_slot = models.IntegerField()
    condition = models.StringField(blank=True)
    item_nr = models.StringField(blank=True)
    item = models.StringField(blank=True)

    # If Producer==0, we store extracted sentences here; else it's "[]"
    producer_sentences = models.LongStringField(blank=True)

    excel_row_index0 = models.IntegerField()
    excel_row_number_guess = models.IntegerField()


class ScheduleItem(ExtraModel):
    """
    One row per participant per oTree round (80 rows).
    """
    subsession = models.Link(Subsession)  # anchor to round-1 subsession
    participant_code = models.StringField()
    round_number = models.IntegerField()
    data = models.LongStringField()  # JSON dict for ONE round


class SentenceStore(ExtraModel):
    subsession = models.Link(Subsession)  # anchor to round-1 subsession
    key = models.StringField()
    value = models.LongStringField()


class CohortSlot(ExtraModel):
    """
    Cohort assignment with replacement safety.
    One row per (exp_num, slot) that is currently occupied.
    """
    subsession = models.Link(Subsession)  # anchor to round-1 subsession
    exp_num = models.IntegerField()
    slot = models.IntegerField()
    participant_code = models.StringField()
    active = models.BooleanField(initial=True)
    completed = models.BooleanField(initial=False)


def _root_subsession(obj):
    """
    Accept Player/Subsession, and Session-like objects.
    """
    try:
        if obj is None:
            return None

        # Player (has .subsession)
        if hasattr(obj, "subsession") and getattr(obj, "subsession", None):
            return obj.subsession.in_round(1)

        # Subsession itself
        if hasattr(obj, "in_round"):
            return obj.in_round(1)

        # Session in oTree typically has get_subsessions()
        if hasattr(obj, "get_subsessions"):
            subs = obj.get_subsessions()
            if subs:
                return subs[0].in_round(1)

    except Exception:
        return None
    return None


# ---- Schedule store ----
def get_schedule_item(obj, participant_code: str, round_number: int):
    root = _root_subsession(obj)
    if not root or not participant_code or not round_number:
        return None
    qs = ScheduleItem.filter(
        subsession=root,
        participant_code=participant_code,
        round_number=int(round_number),
    )
    return qs[0] if qs else None


def schedule_exists(obj, participant_code: str) -> bool:
    return bool(get_schedule_item(obj, participant_code, 1))


def delete_schedule_for_participant(obj, participant_code: str):
    root = _root_subsession(obj)
    if not root:
        return
    items = ScheduleItem.filter(subsession=root, participant_code=participant_code)
    for it in items:
        try:
            it.delete()
        except Exception:
            pass


def ensure_schedule_built(player):
    pcode = player.participant.code
    if not pcode:
        return
    if schedule_exists(player, pcode):
        return
    build_schedule_for_participant(player)


# ---- Sentence store ----
def set_sentence(obj, key: str, value: str):
    root = _root_subsession(obj)
    if not root:
        return
    qs = SentenceStore.filter(subsession=root, key=key)
    if qs:
        qs[0].value = value
        try:
            qs[0].save()  # IMPORTANT
        except Exception:
            pass
    else:
        SentenceStore.create(subsession=root, key=key, value=value)


def get_sentence(obj, key: str):
    root = _root_subsession(obj)
    if not root:
        return None
    qs = SentenceStore.filter(subsession=root, key=key)
    return qs[0].value if qs else None


def has_sentence(obj, key: str) -> bool:
    return bool(get_sentence(obj, key))


# ---- Cohort assignment (DB-backed) ----
def cohort_size(session) -> int:
    return safe_int(session.config.get("cohort_size", 4), 4)


def _active_slot_row(root, exp_num: int, slot: int):
    qs = CohortSlot.filter(subsession=root, exp_num=int(exp_num), slot=int(slot), active=True)
    return qs[0] if qs else None


def _participant_active_assignment(root, participant_code: str):
    qs = CohortSlot.filter(subsession=root, participant_code=participant_code, active=True)
    return qs[0] if qs else None


def cohort_complete(session, exp_num: int) -> bool:
    # complete when all slots 1..csize exist as active AND completed
    root = _root_subsession(session)
    if not root:
        return False
    csize = cohort_size(session)

    for s in range(1, csize + 1):
        row = _active_slot_row(root, exp_num, s)
        if not row:
            return False
        if not bool(row.completed):
            return False
    return True


def _cohort_has_free_slot(session, exp_num: int) -> bool:
    root = _root_subsession(session)
    if not root:
        return True
    csize = cohort_size(session)
    for s in range(1, csize + 1):
        if not _active_slot_row(root, exp_num, s):
            return True
    return False


def assign_slot_if_needed(player):
    """
    Guarantees (fixed):
      - Fill Exp 1 slots first.
      - Exp N only starts AFTER Exp N-1 is COMPLETE.
      - If a cohort is FULL but NOT complete, new participants WAIT (slot=0).
      - Replacement participants can re-fill freed slots in earlier cohorts.
    """
    p = player.participant
    session = player.session
    root = _root_subsession(player)
    if not root:
        p.vars["exp_target"] = 1
        p.vars["local_slot"] = 0
        return 1, 0

    # already assigned?
    existing = _participant_active_assignment(root, p.code)
    if existing:
        p.vars["exp_target"] = int(existing.exp_num)
        p.vars["local_slot"] = int(existing.slot)
        return int(existing.exp_num), int(existing.slot)

    exp_num = 1
    while True:
        # Hard ordering: cannot start exp_num>1 until exp_num-1 is complete.
        if exp_num > 1 and not cohort_complete(session, exp_num - 1):
            # Wait for previous to complete (or for a slot to free there)
            p.vars["exp_target"] = int(exp_num - 1)
            p.vars["local_slot"] = 0
            return int(exp_num - 1), 0

        # If current exp has any free slot, allocate the lowest free slot
        if _cohort_has_free_slot(session, exp_num):
            csize = cohort_size(session)
            for s in range(1, csize + 1):
                if not _active_slot_row(root, exp_num, s):
                    CohortSlot.create(
                        subsession=root,
                        exp_num=int(exp_num),
                        slot=int(s),
                        participant_code=p.code,
                        active=True,
                        completed=False,
                    )
                    p.vars["exp_target"] = int(exp_num)
                    p.vars["local_slot"] = int(s)
                    return int(exp_num), int(s)

        # Cohort is full.
        # If it's not complete yet => wait (do NOT skip ahead).
        if not cohort_complete(session, exp_num):
            p.vars["exp_target"] = int(exp_num)
            p.vars["local_slot"] = 0
            return int(exp_num), 0

        # Full and complete => go to next exp
        exp_num += 1


def free_slot_for_participant(session, participant_code: str):
    """
    Called when Prolific marks TIMED-OUT / RETURNED / REJECTED.
    Frees slot so a replacement participant can take it.
    """
    root = _root_subsession(session)
    if not root:
        return
    rows = CohortSlot.filter(subsession=root, participant_code=participant_code, active=True)
    for r in rows:
        r.active = False
        r.completed = False
        try:
            r.save()
        except Exception:
            pass


def mark_participant_complete_in_cohort(player):
    root = _root_subsession(player)
    if not root:
        return
    rows = CohortSlot.filter(subsession=root, participant_code=player.participant.code, active=True)
    for r in rows:
        r.completed = True
        try:
            r.save()
        except Exception:
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
        return True
    if s in {"no", "n", "0", "false", "f"}:
        return False
    return ""


def participant_completed_app(participant, app_name: str) -> bool:
    return bool(participant.vars.get(f"{app_name}_completed", False))


def mark_participant_completed_app(participant, app_name: str):
    participant.vars[f"{app_name}_completed"] = True


def reset_this_app_for_participant(participant):
    """
    Keep it light. Do NOT store big blobs in participant.vars.
    """
    for k in [
        "needed_sentence_keys",
        "missing_sentence_keys",
        "_needed_sentence_keys_set",
        "exp_target",
        "local_slot",
    ]:
        participant.vars.pop(k, None)

    participant.vars.pop(f"{Constants.name_in_url}_completed", None)
    try:
        participant._index_in_pages = 0
    except Exception:
        pass


def sentences_for_line_from_trialrow(tr: TrialRow, exp_target: int):
    """
    If producer_slot==0, use tr.producer_sentences.
    Else create sentence_lookup referencing source_exp=exp_target-1.
    """
    producer_slot = int(tr.producer_slot)
    interpreter_slot = int(tr.interpreter_slot)
    cond = clean_str(tr.condition)

    if producer_slot == 0:
        s = tr.producer_sentences or "[]"
        if isinstance(s, str) and s.strip():
            return s, None
        return "[]", None

    return "[]", {
        "source_exp": int(exp_target) - 1,
        "producer_slot": int(producer_slot),
        "interpreter_slot": int(interpreter_slot),
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
            for extra in pair[len(suffixes) :]:
                v = clean_str(extra) or "None"
                parts.append(v)

        out.append(" ".join([p for p in parts if str(p).strip()]).strip())

    return out


def required_sentence_keys_for_participant(player) -> set:
    needed = set()
    root = _root_subsession(player)
    if not root:
        return needed
    items = ScheduleItem.filter(subsession=root, participant_code=player.participant.code)
    for it in items:
        d = safe_json_loads(it.data, {})
        if not isinstance(d, dict):
            continue
        lookup = d.get("sentence_lookup")
        if not isinstance(lookup, dict):
            continue
        src_exp = safe_int(lookup.get("source_exp"), -1)
        prod = safe_int(lookup.get("producer_slot"), 0)
        interp = safe_int(lookup.get("interpreter_slot"), 0)
        cond = clean_str(lookup.get("condition"))
        if src_exp >= 0 and prod and interp and cond:
            needed.add(sentence_key(src_exp, prod, interp, cond))
    return needed


# ----------------------------------------------------------------------------
# PROLIFIC GATE (TIMED-OUT / RETURNED / REJECTED)
# ----------------------------------------------------------------------------
class ProlificStatusGate(Page):
    @staticmethod
    def is_displayed(player):
        # don't retroactively gate completed participants
        if participant_completed_app(player.participant, Constants.name_in_url):
            return False
        status = clean_str(player.participant.vars.get("prolific_submission_status", ""))
        return status in Constants.BAD_PROLIFIC_STATUSES

    def get(self):
        # free cohort slot + delete schedule so replacement can take over
        try:
            free_slot_for_participant(self.player.session, self.player.participant.code)
            delete_schedule_for_participant(self.player.session, self.player.participant.code)
            reset_this_app_for_participant(self.player.participant)
            self.player.participant.save()
        except Exception:
            pass
        return RedirectResponse(Constants.FALLBACK_URL, status_code=302)


# ----------------------------------------------------------------------------
# NEW: WAIT PAGE WHEN NO SLOT AVAILABLE (slot==0)
# ----------------------------------------------------------------------------
class WaitForSlot(Page):
    template_name = "img_desc/WaitForPrevExperiment.html"

    @staticmethod
    def is_displayed(player):
        exp_target, local_slot = assign_slot_if_needed(player)
        # If no slot allocated yet, wait here (replacement/return frees slots)
        return int(local_slot or 0) == 0

    @staticmethod
    def vars_for_template(player):
        # reuse your template variables; keep them at 0
        return dict(needed=0, missing=0)


# ----------------------------------------------------------------------------
# SESSION CREATION (store Excel rows in DB, not in memory)
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

        # Normalize settings into session.vars (small only)
        clean_settings = {}
        for k, v in (settings or {}).items():
            if isinstance(v, str):
                clean_settings[normalize_key(k)] = clean_str(v)

        s3_raw = clean_settings.get("s3path") or clean_settings.get("s3path_base") or ""
        session.vars["s3path_base"] = fix_s3_url(s3_raw)
        session.vars["extension"] = clean_settings.get("extension") or "png"
        session.vars["prefix"] = clean_settings.get("prefix") or ""
        session.vars["interpreter_title"] = clean_settings.get("interpreter_title") or "Buy medals:"
        session.vars["caseflag"] = _truthy(clean_settings.get("caseflag"))

        session.vars["instructions_url"] = (
            session.config.get("instructions_url")
            or clean_settings.get("instructions_url")
            or "https://google.com"
        )
        session.vars["introduction_text"] = session.config.get("introduction_text") or ""
        session.vars["doc_link"] = session.config.get("doc_link") or ""

        if session.config.get("completion_code"):
            session.vars["completion_code"] = str(session.config["completion_code"])

        session.vars["allowed_values"] = settings.get("allowed_values", []) or []
        session.vars["allowed_regex"] = settings.get("allowed_regex", []) or []

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

        # Persist Excel content into DB (TrialRow + baseline SentenceStore)
        root = _root_subsession(subsession)
        if not root:
            return

        for idx0, r in enumerate(rows):
            exp_num = get_exp_num(r)
            rrn = idx0 + 2

            # baseline sentences Exp=0 into SentenceStore
            if exp_num == 0:
                prod = safe_int(r.get("Producer"), 0)
                interp = safe_int(r.get("Interpreter"), 0)
                cond = clean_str(r.get("Condition"))
                sent = extract_sentences_from_row(r)
                if sent and sent.strip() != "[]":
                    set_sentence(subsession, sentence_key(0, prod, interp, cond), sent)
                continue

            if exp_num <= 0:
                continue

            TrialRow.create(
                subsession=root,
                exp_num=int(exp_num),
                round_in_excel=safe_int(r.get("Round"), 0),
                trial=safe_int(r.get("Trial"), 0),
                producer_slot=safe_int(r.get("Producer"), 0),
                interpreter_slot=safe_int(r.get("Interpreter"), 0),
                condition=clean_str(r.get("Condition")),
                item_nr=clean_str(r.get("Item.Nr")),
                item=clean_str(r.get("Item")),
                producer_sentences=(
                    extract_sentences_from_row(r)
                    if safe_int(r.get("Producer"), 0) == 0
                    else "[]"
                ),
                excel_row_index0=int(idx0),
                excel_row_number_guess=int(rrn),
            )

        logger.info("Session creation completed successfully (DB-backed rows).")

    except Exception as e:
        logger.error(f"ERROR in creating_session: {e}", exc_info=True)
        raise


# ----------------------------------------------------------------------------
# SCHEDULE BUILD (DB-backed)
# ----------------------------------------------------------------------------
def build_schedule_for_participant(player):
    """
    Create 80 ScheduleItem rows (DB) for this participant.
    Guaranteed Exp ordering via assign_slot_if_needed().
    If no slot is available (slot==0), do nothing; they will wait.
    """
    session = player.session
    p = player.participant
    pcode = p.code

    exp_target, local_slot = assign_slot_if_needed(player)
    if int(local_slot or 0) == 0:
        return

    root = _root_subsession(player)
    if not root:
        return

    # fetch all rows for this experiment from DB
    exp_rows = TrialRow.filter(subsession=root, exp_num=int(exp_target))
    if not exp_rows:
        logger.error(f"No TrialRow found for Exp={exp_target} (participant={pcode}).")
        return

    # group by round_in_excel
    rounds_map = {}
    for tr in exp_rows:
        rnum = int(tr.round_in_excel or 0)
        if rnum >= 1:
            rounds_map.setdefault(rnum, []).append(tr)

    round_numbers = sorted(rounds_map.keys())
    if not round_numbers:
        logger.error(f"No valid Round>=1 TrialRow for Exp={exp_target} (participant={pcode}).")
        return

    # snapshot cohort slot mapping exp_target -> participant_code
    slots = CohortSlot.filter(subsession=root, exp_num=int(exp_target), active=True)
    slot_to_code = {int(r.slot): r.participant_code for r in slots}

    otree_round_counter = 1

    for rnum in round_numbers:
        block = rounds_map[rnum]

        p_hits = [tr for tr in block if int(tr.producer_slot) == int(local_slot)][:3]
        i_hits = [tr for tr in block if int(tr.interpreter_slot) == int(local_slot)][:5]

        # PRODUCER
        for tr in p_hits:
            if otree_round_counter > Constants.num_rounds:
                break

            producer_slot = int(tr.producer_slot)
            interpreter_slot = int(tr.interpreter_slot)
            cond = clean_str(tr.condition)
            item_nr = clean_str(tr.item_nr)
            image_raw = clean_str(tr.item)

            picked = image_raw or "d-A-B-BC-3"
            if producer_slot != 0 and not is_valid_real_image(picked):
                picked = image_raw or "d-A-B-BC-3"

            partner_code = slot_to_code.get(interpreter_slot, "") if interpreter_slot else ""
            ps, lookup = sentences_for_line_from_trialrow(tr, int(exp_target))

            d = dict(
                round_number=otree_round_counter,
                role=PRODUCER,
                partner_code=partner_code,
                exp=int(exp_target),
                round_in_excel=int(rnum),
                trial=int(tr.trial or 0),
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
                excel_row_number_guess=int(tr.excel_row_number_guess or 0),
                excel_row_index0=int(tr.excel_row_index0 or 0),
            )

            ScheduleItem.create(
                subsession=root,
                participant_code=pcode,
                round_number=int(otree_round_counter),
                data=json.dumps(d),
            )
            otree_round_counter += 1

        # INTERPRETER
        for tr in i_hits:
            if otree_round_counter > Constants.num_rounds:
                break

            producer_slot = int(tr.producer_slot)
            interpreter_slot = int(tr.interpreter_slot)
            cond = clean_str(tr.condition)
            item_nr = clean_str(tr.item_nr)
            image_raw = clean_str(tr.item)

            picked = image_raw or ("NA_x" if producer_slot == 0 else "d-A-B-BC-3")
            if producer_slot != 0 and not is_valid_real_image(picked):
                picked = image_raw or "d-A-B-BC-3"

            partner_code = slot_to_code.get(producer_slot, "") if producer_slot not in {0, 9} else ""
            ps, lookup = sentences_for_line_from_trialrow(tr, int(exp_target))

            d = dict(
                round_number=otree_round_counter,
                role=INTERPRETER,
                partner_code=partner_code,
                exp=int(exp_target),
                round_in_excel=int(rnum),
                trial=int(tr.trial or 0),
                condition=cond,
                item_nr=item_nr,
                image=picked,
                producer_sentences=ps,
                sentence_lookup=lookup,
                interpreter_rewards="",
                producer_slot=producer_slot,
                interpreter_slot=interpreter_slot,
                excel_row_number_guess=int(tr.excel_row_number_guess or 0),
                excel_row_index0=int(tr.excel_row_index0 or 0),
            )

            ScheduleItem.create(
                subsession=root,
                participant_code=pcode,
                round_number=int(otree_round_counter),
                data=json.dumps(d),
            )
            otree_round_counter += 1

        if otree_round_counter > Constants.num_rounds:
            break

    if otree_round_counter <= Constants.num_rounds:
        logger.warning(
            f"Schedule len mismatch for participant={pcode} exp={exp_target}: "
            f"got={otree_round_counter-1} expected={Constants.num_rounds}"
        )


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
        # ensure slot assignment + schedule exist (safe on refresh)
        assign_slot_if_needed(player)
        ensure_schedule_built(player)

        exp_target = int(player.participant.vars.get("exp_target") or 1)
        if exp_target <= 1:
            return False

        # hard gate: do not proceed until previous cohort is COMPLETE
        if not cohort_complete(player.session, exp_target - 1):
            player.participant.vars["needed_sentence_keys"] = 0
            player.participant.vars["missing_sentence_keys"] = 0
            return True

        # secondary gate: ensure needed sentences exist (if you use lookup across cohorts)
        needed = player.participant.vars.get("_needed_sentence_keys_set")
        if not isinstance(needed, list):
            needed_set = required_sentence_keys_for_participant(player)
            player.participant.vars["_needed_sentence_keys_set"] = list(needed_set)
            needed = list(needed_set)

        if not needed:
            return False

        missing = [k for k in needed if not has_sentence(player, k)]

        player.participant.vars["needed_sentence_keys"] = len(needed)
        player.participant.vars["missing_sentence_keys"] = len(missing)

        return len(missing) > 0

    @staticmethod
    def vars_for_template(player):
        return dict(
            needed=player.participant.vars.get("needed_sentence_keys", 0),
            missing=player.participant.vars.get("missing_sentence_keys", 0),
        )


class Q(Page):
    form_model = "player"

    @staticmethod
    def is_displayed(player):
        if player.round_number > Constants.num_rounds:
            return False

        exp_target, local_slot = assign_slot_if_needed(player)
        if int(local_slot or 0) == 0:
            # WaitForSlot will show instead
            return False

        ensure_schedule_built(player)

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
                    set_sentence(player, k, player.producer_decision)

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
                for d0 in parsed:
                    if not isinstance(d0, dict):
                        continue
                    opt = d0.get("option") or d0.get("choice") or d0.get("label") or d0.get("text") or d0.get("name")
                    if opt is None:
                        continue
                    opt = str(opt).strip()
                    by_label[opt] = normalize_yesno_to_01(d0.get("answer", d0.get("value", d0.get("selected"))))

                if by_label:
                    for opt in choices:
                        labeled.append({"option": opt, "answer": by_label.get(opt, "")})
                else:
                    for i, opt in enumerate(choices):
                        if i >= len(parsed):
                            labeled.append({"option": opt, "answer": ""})
                            continue
                        d0 = parsed[i] if isinstance(parsed[i], dict) else {}
                        labeled.append(
                            {"option": opt, "answer": normalize_yesno_to_01(d0.get("answer", d0.get("value", d0.get("selected"))))}
                        )
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


class Feedback(Page):
    form_model = "player"
    form_fields = ["feedback"]

    @staticmethod
    def is_displayed(player):
        return player.round_number == Constants.num_rounds

    @staticmethod
    def before_next_page(player, timeout_happened):
        mark_participant_completed_app(player.participant, Constants.name_in_url)
        player.participant.save()
        mark_participant_complete_in_cohort(player)


PROLIFIC_COMPLETE_BASE = "https://app.prolific.com/submissions/complete?cc="


class FinalForProlific(Page):
    @staticmethod
    def is_displayed(player):
        return (
            player.session.config.get("for_prolific")
            and player.round_number == Constants.num_rounds
        )

    def get(self):
        cc = (
            self.player.session.vars.get("completion_code")
            or self.player.session.config.get("completion_code")
        )

        if not cc:
            return RedirectResponse(Constants.API_ERR_URL, status_code=302)

        url = PROLIFIC_COMPLETE_BASE + str(cc).strip()
        logger.info(f"[FinalForProlific] redirecting to: {url}")
        return RedirectResponse(url, status_code=302)

    @staticmethod
    def before_next_page(player, timeout_happened):
        if player.round_number == Constants.num_rounds:
            player.in_round(1).completed_experiment = True


# ----------------------------------------------------------------------------
# EXPORT (reads ScheduleItem, not batch_history)
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
    DB-backed export:
      - Group by participant
      - Read schedule from ScheduleItem table
      - Use Player rows only for timing + feedback
    """

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

    def _parse_interp_answers(raw_interp, choices):
        out = {c: "" for c in choices}
        data = safe_json_loads(raw_interp, [])
        if not isinstance(data, list):
            return out
        for d0 in data:
            if not isinstance(d0, dict):
                continue
            opt = str(d0.get("option", "")).strip()
            if opt in out:
                out[opt] = normalize_yesno_to_01(d0.get("answer"))
        return out

    def _resolve_lookup(item, obj_for_db):
        lookup = item.get("sentence_lookup")
        if not isinstance(lookup, dict):
            return ""
        src_exp = safe_int(lookup.get("source_exp"), -1)
        prod = safe_int(lookup.get("producer_slot"), 0)
        interp = safe_int(lookup.get("interpreter_slot"), 0)
        cond = clean_str(lookup.get("condition"))
        k = sentence_key(src_exp, prod, interp, cond)
        return get_sentence(obj_for_db, k) or ""

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

    choice_headers = ["Option_1", "Option_2", "Option_3", "Option_4"]
    if players:
        try:
            choice_headers = _get_choices(players[0].session.vars)
        except Exception:
            pass

    # header
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
        *choice_headers,
        "rewards_raw",
        "seconds",
        "feedback",
        "excel_row_number_guess",
        "excel_row_index0",
    ]

    if not players:
        return

    # group players by participant.code
    players_sorted = sorted(
        players,
        key=lambda p: (
            getattr(getattr(p, "participant", None), "code", ""),
            getattr(p, "round_number", 0) or 0,
        ),
    )

    root = _root_subsession(players_sorted[0])
    if not root:
        return

    current_code = None
    bucket = []

    def _emit_bucket(bucket_players):
        if not bucket_players:
            return
        first = bucket_players[0]
        participant = getattr(first, "participant", None)
        if participant is None:
            return

        session_obj = getattr(first, "session", None)
        session_code = getattr(session_obj, "code", "")
        participant_code = getattr(participant, "code", "")

        prolific_id = participant.vars.get("prolific_id", "")

        demo_obj = {}
        try:
            raw_demo = participant.vars.get("demographics")
            if raw_demo:
                demo_obj = safe_json_loads(raw_demo, {})
            else:
                start_players = [
                    pp for pp in participant.get_players()
                    if hasattr(pp, "survey_data") and getattr(pp, "survey_data", None)
                ]
                if start_players:
                    demo_obj = safe_json_loads(start_players[0].survey_data, {})
        except Exception:
            demo_obj = {}
        demo_cols = [demo_obj.get(k, "") for k in demo_keys]

        timing_map = {}
        feedback_str = ""
        for pp in bucket_players:
            rn = getattr(pp, "round_number", 0) or 0
            if rn:
                timing_map[rn] = getattr(pp, "decision_seconds", 0) or 0
            if rn == Constants.num_rounds and getattr(pp, "feedback", ""):
                feedback_str = getattr(pp, "feedback", "")

        # schedule from DB
        sched_items = ScheduleItem.filter(subsession=root, participant_code=participant_code)
        sched_items.sort(key=lambda it: int(getattr(it, "round_number", 0) or 0))

        obj_for_db = bucket_players[0]

        for it in sched_items:
            item = safe_json_loads(it.data, {})
            if not isinstance(item, dict):
                continue

            rnd = int(item.get("round_number", 0))
            if rnd < 1 or rnd > Constants.num_rounds:
                continue

            my_role = item.get("role", "")
            prod_id = safe_int(item.get("producer_slot"), 0)
            interp_id = safe_int(item.get("interpreter_slot"), 0)
            exp_num = item.get("exp", "")

            raw_sentences = item.get("producer_sentences") or ""
            if not raw_sentences or (isinstance(raw_sentences, str) and raw_sentences.strip() in {"", "[]"}):
                resolved = _resolve_lookup(item, obj_for_db)
                if resolved:
                    raw_sentences = resolved
            sentence_cells = extract_sentence_cells(raw_sentences)

            raw_interp = item.get("interpreter_rewards") or ""
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

    for p in players_sorted:
        part = getattr(p, "participant", None)
        code = getattr(part, "code", "")
        if current_code is None:
            current_code = code
        if code != current_code:
            yield from _emit_bucket(bucket)
            bucket = []
            current_code = code
        bucket.append(p)

    if bucket:
        yield from _emit_bucket(bucket)


page_sequence = [
    ProlificStatusGate,
    WaitForSlot,          
    FaultyCatcher,
    WaitForPrevExperiment,
    Q,
    Feedback,
    FinalForProlific,
]
