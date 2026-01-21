from otree.api import *
import json
import logging
import time
import random
import re
from starlette.responses import RedirectResponse

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
            for item in hist:
                if int(item.get("round_number", 0)) == rnd:
                    item.update(updates)
                    self.batch_history = json.dumps(hist)
                    return
        except Exception:
            pass

    # -------------------------
    # content helpers used by template
    # -------------------------
    def get_image_url(self):
        """
        Main experiment images (NOT practice).
        Uses session.vars['s3path_base'] which should be bucket root (no /practice).
        """
        data = self.get_current_batch_data()
        img = clean_str(data.get("image", ""))
        if not img:
            return ""

        base = (self.session.vars.get("s3path_base") or "").rstrip("/")
        ext = self.session.vars.get("extension") or "png"

        clean_name = img.replace(" ", "_")
        if not clean_name.lower().endswith(f".{ext}"):
            clean_name = f"{clean_name}.{ext}"

        return f"{base}/{clean_name}"

    def get_sentences_data(self):
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
                if not val_str or val_str.lower() in {"nan", "none"}:
                    val_str = "None"
                parts.append(val_str)
                if suf:
                    parts.append(str(suf))

            if len(sentence) > len(suffixes):
                for extra in sentence[len(suffixes):]:
                    extra_str = str(extra).strip() if extra else ""
                    if not extra_str or extra_str.lower() in {"nan", "none"}:
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
    if s.lower() in {"nan", "none", "na", "na_x", "x"}:
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
    Eligible real images:
      - must start with d- or D-
      - reject placeholders like d_... or D_...
    """
    img = clean_str(img)
    if not img:
        return False
    if img.startswith("D_") or (img.startswith("d_") and not img.startswith("d-")):
        return False
    if img.startswith("d-") or img.startswith("D-"):
        return True
    return False


def extract_sentences_from_row(r: dict) -> str:
    pairs = []
    for i in range(1, 6):
        a = clean_str(r.get(f"Sentence_{i}_1"))
        b = clean_str(r.get(f"Sentence_{i}_2"))
        if a or b:
            pairs.append([a, b])
    return json.dumps(pairs)


def pick_exp_value(row: dict) -> int:
    """
    Your Exp column is mistakenly named 'd111'.
    We'll prefer 'd111' first, then fall back to common alternatives.
    """
    for key in ["d111", "Exp", "EXP", "Experiment", "experiment", "Exp."]:
        if key in row:
            v = row.get(key)
            # allow 0, 1, 2, 9 etc.
            if str(v).strip() != "":
                return safe_int(v, 0)
    return 0


def ensure_ext(name: str, ext: str) -> str:
    name = (name or "").strip()
    if not name:
        return ""
    if not name.lower().endswith(f".{ext}"):
        return f"{name}.{ext}"
    return name


def build_practice_image_url(base_root: str, img_name: str, ext: str) -> str:
    """
    Practice images live under:
      <bucket-root>/practice/<file>
    """
    img_name = (img_name or "").strip()
    if not img_name:
        return ""
    img_file = ensure_ext(img_name, ext)
    base_root = (base_root or "").rstrip("/")
    return f"{base_root}/practice/{img_file}"


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
            raise RuntimeError("No filename in session config (set it to your Google Sheet URL).")

        logger.info(f"Starting session creation with filename: {filename}")

        # import loader (must exist now)
        from reading_xls.get_data import get_data

        excel_payload = get_data(filename)
        rows = list(excel_payload.get("data") or [])
        settings = excel_payload.get("settings") or {}

        if not rows:
            raise RuntimeError("No rows loaded from Google Sheet (data/items/trials tab).")

        logger.info(f"Total rows loaded: {len(rows)}")
        logger.info(f"Row[0] keys: {list(rows[0].keys())}")

        # ---------------- settings -> session.vars ----------------
        # only normalize string settings here
        clean_settings = {normalize_key(k): clean_str(v) for k, v in settings.items() if isinstance(v, str)}

        # IMPORTANT: s3path_base should be BUCKET ROOT (NOT /practice)
        s3_root = clean_settings.get("s3path_base") or clean_settings.get("s3path") or ""
        session.vars["s3path_base"] = s3_root.rstrip("/")  # bucket root
        session.vars["extension"] = clean_settings.get("extension") or "png"
        session.vars["prefix"] = clean_settings.get("prefix") or ""
        session.vars["interpreter_title"] = clean_settings.get("interpreter_title") or "Buy medals:"
        session.vars["caseflag"] = _truthy(clean_settings.get("caseflag"))
        session.vars["instructions_url"] = clean_settings.get("instructions_url") or "https://google.com"

        # suffixes
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

        session.vars["allowed_values"] = settings.get("allowed_values", []) or []
        session.vars["allowed_regexes"] = settings.get("allowed_regex", []) or []

        if session.config.get("completion_code"):
            session.vars["completion_code"] = str(session.config["completion_code"])

        # ---------------- load practice configs from settings ----------------
        # get_data.py stored practice sheets as settings['Practice1'], ...
        ext = session.vars["extension"]
        base_root = session.vars["s3path_base"]

        for i in range(1, 21):
            key = f"Practice{i}"
            val = settings.get(key)
            if not isinstance(val, dict):
                continue

            practice_data = val.copy()
            img_name = practice_data.get("image", "").strip() if isinstance(practice_data.get("image"), str) else ""
            if img_name:
                practice_data["image"] = build_practice_image_url(base_root, img_name, ext)
                logger.info(f"Loaded {key} into session.vars with image {practice_data['image']}")
            else:
                logger.warning(f"{key}: practice image missing (expected key 'image' in practice sheet)")
                practice_data["image"] = ""

            session.vars[key] = practice_data

        # ---------------- determine slot universe ----------------
        slot_ids = set()
        for r in rows:
            p = safe_int(r.get("Producer"), 0)
            it = safe_int(r.get("Interpreter"), 0)
            if p != 0 and p != 9:
                slot_ids.add(p)
            if it != 0:
                slot_ids.add(it)

        if not slot_ids:
            raise RuntimeError("No Producer/Interpreter IDs found in data rows.")

        K = max(slot_ids)
        players = subsession.get_players()

        # Map slots 1..K to first K participants
        slot_to_pid = {}
        pid_to_slot = {}
        for idx, pl in enumerate(players, start=1):
            if idx <= K:
                slot_to_pid[idx] = pl.id_in_subsession
                pid_to_slot[pl.id_in_subsession] = idx

        logger.info(f"Excel slot universe: 1..{K}")

        # ---------------- build valid image pool from REAL producer rows ----------------
        valid_pool = []
        for r in rows:
            producer_slot = safe_int(r.get("Producer"), 0)
            img = clean_str(r.get("Item"))
            if producer_slot != 0 and producer_slot != 9 and is_valid_real_image(img):
                valid_pool.append(img)

        if not valid_pool:
            logger.warning("No valid images found in sheet. Using fallback.")
            valid_pool = ["d-A-B-BC-3"]

        # ---------------- build schedule ----------------
        from collections import defaultdict
        data_by_pid = defaultdict(list)

        for idx, r in enumerate(rows):
            exp_num = pick_exp_value(r)          # <-- uses d111
            round_in_excel = safe_int(r.get("Round"), 0)
            trial = safe_int(r.get("Trial"), 0)
            condition = clean_str(r.get("Condition"))
            item_nr = clean_str(r.get("Item.Nr"))
            image_raw = clean_str(r.get("Item"))
            producer_slot = safe_int(r.get("Producer"), 0)
            interpreter_slot = safe_int(r.get("Interpreter"), 0)

            if interpreter_slot == 0:
                continue

            sentences_json = extract_sentences_from_row(r)

            interp_pid = slot_to_pid.get(interpreter_slot)
            prod_pid = slot_to_pid.get(producer_slot) if (producer_slot != 0 and producer_slot != 9) else None
            if not interp_pid:
                continue

            sort_key = (exp_num, round_in_excel, trial, idx)

            if producer_slot in (0, 9):
                picked_image = random.choice(valid_pool)
                data_by_pid[interp_pid].append({
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
                })
            elif prod_pid:
                prod_image = image_raw if is_valid_real_image(image_raw) else random.choice(valid_pool)

                # Producer view
                data_by_pid[prod_pid].append({
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
                })

                # Interpreter view paired with producer
                data_by_pid[interp_pid].append({
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
                })

        # quick visibility into schedules
        for slot in sorted(slot_to_pid.keys()):
            pid = slot_to_pid[slot]
            logger.info(f"Slot {slot} -> pid {pid}: {len(data_by_pid.get(pid, []))} total items")

        # ---------------- finalize pattern 3P + 5I ----------------
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
                    break

            p.batch_history = json.dumps(final_history)
            p.participant.vars["batch_history"] = p.batch_history
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

        return dict(
            d=player.get_current_batch_data(),
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
            return RedirectResponse(Constants.API_ERR_URL, status_code=302)
        return RedirectResponse(STUBURL + str(cc), status_code=302)


page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
