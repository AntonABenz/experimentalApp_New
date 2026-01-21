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
    # content helpers used by img_desc templates
    # -------------------------
    def get_image_url(self):
        """
        Task-round image URL (NOT practice intro pages).
        """
        data = self.get_current_batch_data()
        img = clean_str(data.get("image", ""))
        if not img or img.lower() in {"nan", "na", "na_x", "none", "", "x"}:
            return ""
        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        ext = self.session.vars.get("extension") or "png"

        # keep prior behavior: strip "/practice" if present (task images aren't under /practice)
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

            for val, suf in zip(sentence, suffixes):
                val_str = str(val).strip() if val else ""
                if not val_str or val_str.lower() in ["nan", "none"]:
                    val_str = "None"
                parts.append(val_str)
                if suf:
                    parts.append(str(suf))

            if len(sentence) > len(suffixes):
                for extra in sentence[len(suffixes) :]:
                    extra_str = str(extra).strip() if extra else ""
                    if not extra_str or extra_str.lower() in ["nan", "none"]:
                        extra_str = "None"
                    parts.append(extra_str)

            res.append(" ".join([p for p in parts if str(p).strip() != ""]))
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
    """
    Only accept 'd-...' real images.
    Reject placeholders like 'D_...' or 'd_...' (underscore after d).
    """
    img = clean_str(img)
    if not img:
        return False
    low = img.lower()
    if low in {"na_x", "na", "nan", "none", "x"}:
        return False
    if img.startswith("D_") or (img.startswith("d_") and not img.startswith("d-")):
        return False
    if img.startswith("d-") or img.startswith("D-"):
        return True
    return False


def extract_sentences_from_row(r: dict) -> str:
    """
    Excel has Sentence_1_1, Sentence_1_2, ..., Sentence_5_2.
    Convert them into JSON list of pairs.
    """
    pairs = []
    for i in range(1, 6):
        a = clean_str(r.get(f"Sentence_{i}_1"))
        b = clean_str(r.get(f"Sentence_{i}_2"))
        if a or b:
            pairs.append([a, b])
    return json.dumps(pairs)


def fix_s3_url(raw_s3: str) -> str:
    """
    Convert AWS console bucket URL to bucket endpoint URL.
    Also supports already-correct base URLs.
    """
    raw_s3 = clean_str(raw_s3)
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            bucket = raw_s3.split("buckets/")[1].split("?")[0].strip("/")
            return f"https://{bucket}.s3.eu-central-1.amazonaws.com"
        except Exception:
            return raw_s3
    return raw_s3


def _practice_image_url(session, image_name: str) -> str:
    """
    Practice images live under: <s3_base>/practice/<image>.<ext>
    """
    image_name = clean_str(image_name)
    if not image_name:
        return ""
    base = (session.vars.get("s3path_base") or "").rstrip("/")
    ext = session.vars.get("extension") or "png"

    # ensure base is bucket endpoint (not console)
    base = fix_s3_url(base)

    # practice images are inside /practice
    clean_name = image_name.replace(" ", "_")
    if not clean_name.lower().endswith(f".{ext}"):
        clean_name = f"{clean_name}.{ext}"
    return f"{base}/practice/{clean_name}"


def _parse_right_answers(practice_data: dict) -> list:
    """
    Collect right_answer_1.. right_answer_N in order.
    Each value can be:
      - "1" (binary practice)
      - "3; the A" (multi-field, split by ;)
    Returns: list[list[str]]
    """
    answers = []
    keys = []
    for k in practice_data.keys():
        if str(k).lower().startswith("right_answer_"):
            keys.append(k)

    def _num(k):
        m = re.findall(r"\d+", str(k))
        return int(m[0]) if m else 999999

    for k in sorted(keys, key=_num):
        raw = clean_str(practice_data.get(k))
        if not raw:
            continue
        parts = [p.strip() for p in raw.split(";")]
        parts = [p for p in parts if p != ""]
        if parts:
            answers.append(parts)
    return answers


def _detect_exp_column(rows: list[dict]) -> str:
    """
    Exp column may be misnamed (e.g., 'd111').
    Prefer: Exp / Experiment / exp, else fall back to 'd111' if present.
    """
    if not rows:
        return "Exp"

    cols = list(rows[0].keys())

    # 1) strong candidates
    for c in ["Exp", "exp", "EXP", "Experiment", "experiment", "Exp.", "EXPERIMENT"]:
        if c in cols:
            return c

    # 2) your known mistake
    if "d111" in cols:
        return "d111"

    # 3) heuristic: choose a column with small integer values (0..50) that is not Producer/Interpreter/Round/Trial
    banned = {"producer", "interpreter", "round", "trial", "item.nr"}
    for c in cols:
        if normalize_key(c) in banned:
            continue
        v = rows[0].get(c)
        n = safe_int(v, -999)
        if 0 <= n <= 50:
            return c

    # fallback
    return "Exp"


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

        # Import get_data (must exist in reading_xls/get_data.py)
        from reading_xls.get_data import get_data

        excel_payload = get_data(filename)
        raw_data = excel_payload.get("data")
        settings = excel_payload.get("settings") or {}

        # raw_data should be list[dict]
        rows = list(raw_data or [])
        logger.info(f"Total rows loaded: {len(rows)}")

        # ---------------- settings -> session.vars ----------------
        # only normalize string settings here (practice dicts remain dicts)
        clean_settings = {}
        for k, v in (settings or {}).items():
            if isinstance(v, str):
                clean_settings[normalize_key(k)] = clean_str(v)

        # S3 base & extension
        s3_raw = clean_settings.get("s3path") or clean_settings.get("s3path_base") or ""
        session.vars["s3path_base"] = fix_s3_url(s3_raw)
        session.vars["extension"] = clean_settings.get("extension") or "png"

        # misc settings
        session.vars["prefix"] = clean_settings.get("prefix") or ""
        session.vars["interpreter_title"] = clean_settings.get("interpreter_title") or "Buy medals:"
        session.vars["caseflag"] = _truthy(clean_settings.get("caseflag"))
        session.vars["instructions_url"] = clean_settings.get("instructions_url") or "https://google.com"

        # suffixes suffix_1..suffix_10
        suffixes = []
        for i in range(1, 11):
            v = clean_settings.get(f"suffix_{i}")
            if v:
                suffixes.append(v)
        session.vars["suffixes"] = suffixes

        # interpreter choices
        ic = settings.get("interpreter_choices")
        if isinstance(ic, str):
            session.vars["interpreter_choices"] = [x.strip() for x in ic.split(";") if x.strip()]
        elif isinstance(ic, list):
            session.vars["interpreter_choices"] = ic
        else:
            session.vars["interpreter_choices"] = []

        # allowed values/regex arrays (already prepared by get_data.py)
        session.vars["allowed_values"] = settings.get("allowed_values", []) or []
        session.vars["allowed_regexes"] = settings.get("allowed_regex", []) or []

        if session.config.get("completion_code"):
            session.vars["completion_code"] = str(session.config["completion_code"])

        # ---------------- practice sheets -> session.vars['PracticeX'] ----------------
        # get_data() should store dicts under keys like Practice1, Practice2, ...
        for k, v in (settings or {}).items():
            if isinstance(v, dict) and str(k).lower().startswith("practice"):
                practice_data = dict(v)  # copy
                practice_data["title"] = clean_str(practice_data.get("title"))
                practice_data["main_text"] = practice_data.get("main_text", "")

                img_name = clean_str(practice_data.get("image"))
                practice_data["image_url"] = _practice_image_url(session, img_name) if img_name else ""

                # Normalize right answers
                practice_data["right_answers"] = _parse_right_answers(practice_data)

                session.vars[k] = practice_data
                logger.info(
                    f"Loaded {k}: image_url={'yes' if practice_data['image_url'] else 'no'}, "
                    f"right_answers={len(practice_data['right_answers'])}"
                )

        # ---------------- detect Exp column (Exp vs d111) ----------------
        exp_col = _detect_exp_column(rows)
        logger.info(f"Using '{exp_col}' as experiment column")

        # ---------------- determine Excel slot universe ----------------
        slot_ids = set()
        for r in rows:
            p = safe_int(r.get("Producer"), 0)
            i = safe_int(r.get("Interpreter"), 0)
            if p != 0 and p != 9:  # exclude Producer=9 from slot universe
                slot_ids.add(p)
            if i != 0:
                slot_ids.add(i)

        if not slot_ids:
            raise RuntimeError("No Producer/Interpreter IDs found in data rows.")

        K = max(slot_ids)
        players = subsession.get_players()

        logger.info(f"Players={len(players)}, Max slot ID={K}")

        # Map excel slots 1..K to first K participants
        slot_to_pid = {}
        pid_to_slot = {}
        for idx, pl in enumerate(players, start=1):
            if idx <= K:
                slot_to_pid[idx] = pl.id_in_subsession
                pid_to_slot[pl.id_in_subsession] = idx

        # ---------------- valid image pool (only from real producer rows, exclude Producer=9) ----------------
        valid_pool = []
        for r in rows:
            producer_slot = safe_int(r.get("Producer"), 0)
            img = clean_str(r.get("Item"))
            if producer_slot != 0 and producer_slot != 9 and is_valid_real_image(img):
                valid_pool.append(img)

        if not valid_pool:
            logger.warning("No valid images found in data. Using fallback valid image name.")
            valid_pool = ["d-A-B-BC-3"]

        # ---------------- build schedule for each participant ----------------
        from collections import defaultdict

        data_by_pid = defaultdict(list)

        rows_processed = 0
        producer_9_count = 0
        producer_0_count = 0
        normal_pair_count = 0

        for idx, r in enumerate(rows):
            exp_num = safe_int(r.get(exp_col), 0)
            round_in_excel = safe_int(r.get("Round"), 0)
            trial = safe_int(r.get("Trial"), 0)

            condition = clean_str(r.get("Condition"))
            item_nr = clean_str(r.get("Item.Nr"))
            image_raw = clean_str(r.get("Item"))

            producer_slot = safe_int(r.get("Producer"), 0)
            interpreter_slot = safe_int(r.get("Interpreter"), 0)

            if interpreter_slot == 0:
                continue  # unusable row

            sentences_json = extract_sentences_from_row(r)

            interp_pid = slot_to_pid.get(interpreter_slot)
            prod_pid = slot_to_pid.get(producer_slot) if (producer_slot != 0 and producer_slot != 9) else None

            if not interp_pid:
                continue

            sort_key = (exp_num, round_in_excel, trial, idx)
            rows_processed += 1

            # Producer=9 -> interpreter only, random valid image
            if producer_slot == 9:
                producer_9_count += 1
                picked_image = random.choice(valid_pool)
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
                        "image": picked_image,
                        "producer_sentences": sentences_json,
                        "interpreter_rewards": "",
                    }
                )
                continue

            # Producer=0 -> interpreter only, random valid image
            if producer_slot == 0:
                producer_0_count += 1
                picked_image = random.choice(valid_pool)
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
                        "image": picked_image,
                        "producer_sentences": sentences_json,
                        "interpreter_rewards": "",
                    }
                )
                continue

            # Normal producer-interpreter pairing
            if prod_pid:
                normal_pair_count += 1
                prod_image = image_raw if is_valid_real_image(image_raw) else random.choice(valid_pool)

                # producer entry
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
                        "image": prod_image,
                        "producer_sentences": "",
                        "interpreter_rewards": "",
                    }
                )

                # interpreter entry paired
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
                        "image": prod_image,
                        "producer_sentences": sentences_json,
                        "interpreter_rewards": "",
                    }
                )

        logger.info(
            f"Rows processed={rows_processed} "
            f"(Producer=0: {producer_0_count}, Producer=9: {producer_9_count}, Normal pairs: {normal_pair_count})"
        )

        # ---------------- finalize with 3P + 5I pattern repeating ----------------
        empty = []
        for p in players:
            my_items = data_by_pid.get(p.id_in_subsession, [])
            my_items.sort(key=lambda x: x["sort_key"])

            producer_items = [it for it in my_items if it.get("role") == PRODUCER]
            interpreter_items = [it for it in my_items if it.get("role") == INTERPRETER]

            final_history = []
            round_counter = 1
            p_idx = 0
            i_idx = 0

            while round_counter <= Constants.num_rounds:
                # 3 producer rounds
                for _ in range(3):
                    if round_counter > Constants.num_rounds:
                        break
                    if p_idx < len(producer_items):
                        item = producer_items[p_idx].copy()
                        item.pop("sort_key", None)
                        item["round_number"] = round_counter
                        final_history.append(item)
                        p_idx += 1
                        round_counter += 1
                    else:
                        break

                # 5 interpreter rounds
                for _ in range(5):
                    if round_counter > Constants.num_rounds:
                        break
                    if i_idx < len(interpreter_items):
                        item = interpreter_items[i_idx].copy()
                        item.pop("sort_key", None)
                        item["round_number"] = round_counter
                        final_history.append(item)
                        i_idx += 1
                        round_counter += 1
                    else:
                        break

                if p_idx >= len(producer_items) and i_idx >= len(interpreter_items):
                    logger.warning(f"Player {p.id_in_subsession}: ran out of items at round {round_counter}")
                    break

            p.batch_history = json.dumps(final_history)
            p.participant.vars["batch_history"] = p.batch_history
            if not final_history:
                empty.append(p.id_in_subsession)

        if empty:
            logger.warning(f"EMPTY schedules for participants: {empty}")

        logger.info(
            f"Schedule built. players={len(players)} valid_pool={len(valid_pool)} rows={len(rows)} "
            f"exp_col={exp_col}"
        )

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
    """
    One row per round per participant with timing, responses, and demographics.
    Demographics from the 'start' app included in every row.
    """
    yield [
        "session",
        "participant",
        "prolific_id",
        "excel_slot",
        "demographics",
        "round",
        "role",
        "condition",
        "item_nr",
        "image",
        "producer_sentences",
        "interpreter_rewards",
        "seconds",
        "feedback",
        "exp_num",
        "partner_id",
    ]

    from collections import defaultdict

    players_by_participant = defaultdict(list)
    for p in players:
        players_by_participant[p.participant.code].append(p)

    for participant_code, participant_players in players_by_participant.items():
        try:
            first_player = participant_players[0]

            prolific_id = first_player.participant.vars.get("prolific_id", "")
            excel_slot = first_player.id_in_subsession

            demographics = ""
            try:
                start_players = [
                    p for p in first_player.participant.get_players() if hasattr(p, "survey_data") and p.survey_data
                ]
                if start_players:
                    demographics = start_players[0].survey_data or ""
            except Exception as e:
                logger.warning(f"Could not retrieve demographics for {participant_code}: {e}")

            history_json = first_player.participant.vars.get("batch_history", "[]")
            history = json.loads(history_json)

            timing_map = {}
            feedback_map = {}
            for p in participant_players:
                if p.round_number:
                    timing_map[p.round_number] = p.decision_seconds or 0
                    if p.feedback:
                        feedback_map[p.round_number] = p.feedback

            history.sort(key=lambda x: int(x.get("round_number", 0)))

            for item in history:
                rnd = int(item.get("round_number", 0))
                if rnd < 1 or rnd > Constants.num_rounds:
                    continue

                yield [
                    first_player.session.code,
                    participant_code,
                    prolific_id,
                    excel_slot,
                    demographics,
                    rnd,
                    item.get("role", ""),
                    item.get("condition", ""),
                    item.get("item_nr", ""),
                    item.get("image", ""),
                    item.get("producer_sentences", ""),
                    item.get("interpreter_rewards", ""),
                    timing_map.get(rnd, 0),
                    feedback_map.get(rnd, ""),
                    item.get("exp", ""),
                    item.get("partner_id", 0),
                ]
        except Exception as e:
            logger.error(f"Error exporting participant {participant_code}: {e}")
            continue


page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
