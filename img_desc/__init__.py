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
    batch_history = models.LongStringField(initial="[]")
    inner_role = models.StringField()
    faulty = models.BooleanField(initial=False)
    feedback = models.LongStringField(label="")
    producer_decision = models.LongStringField()
    interpreter_decision = models.LongStringField()
    start_decision_time = models.FloatField(initial=0)
    end_decision_time = models.FloatField(initial=0)
    decision_seconds = models.FloatField(initial=0)

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
        data = self.get_current_batch_data()
        if not data:
            return []
        role = data.get("role", "")
        if role == PRODUCER:
            return []
        raw = data.get("producer_sentences") or "[]"
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
                for extra in sentence[len(suffixes):]:
                    extra_str = str(extra).strip() if extra else ""
                    if not extra_str or extra_str.lower() in ["nan", "none"]:
                        extra_str = "None"
                    parts.append(extra_str)
            res.append(" ".join(parts))
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
    # reject practice placeholders like D_5_... (underscore after d)
    if img.startswith("D_") or (img.startswith("d_") and not img.startswith("d-")):
        return False
    # accept real images like d-A-B-... (hyphen after d)
    return img.startswith("d-") or img.startswith("D-")


def extract_sentences_from_row(r: dict) -> str:
    pairs = []
    for i in range(1, 6):
        a = clean_str(r.get(f"Sentence_{i}_1"))
        b = clean_str(r.get(f"Sentence_{i}_2"))
        if a or b:
            pairs.append([a, b])
    return json.dumps(pairs)


def fix_s3_url(raw_s3: str) -> str:
    raw_s3 = clean_str(raw_s3)
    # convert AWS console bucket URL to bucket endpoint
    if "console.aws.amazon.com" in raw_s3 and "buckets/" in raw_s3:
        try:
            bucket = raw_s3.split("buckets/")[1].split("?")[0].strip("/")
            return f"https://{bucket}.s3.eu-central-1.amazonaws.com"
        except Exception:
            return raw_s3
    return raw_s3


def _pick_exp_value(row: dict) -> int:
    """
    Your sheet sometimes has Exp misnamed as d111.
    Prefer Exp if present, else d111.
    """
    for k in ("Exp", "exp", "EXP", "d111", "D111", "experiment", "Experiment"):
        if k in row and str(row.get(k)).strip() != "":
            return safe_int(row.get(k), 0)
    return 0


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

        # Import the loader
        from reading_xls.get_data import get_data

        excel_payload = get_data(filename)
        rows = list(excel_payload.get("data") or [])
        settings = excel_payload.get("settings") or {}

        if not rows:
            raise RuntimeError("No rows loaded from 'data' tab.")

        logger.info(f"Loaded rows={len(rows)}. settings_keys={len(settings.keys())}")

        # ---------------- settings -> session.vars ----------------
        clean_settings = {normalize_key(k): clean_str(v) for k, v in (settings or {}).items()}

        s3_raw = clean_settings.get("s3path") or clean_settings.get("s3path_base") or ""
        s3_base = fix_s3_url(s3_raw).rstrip("/")

        session.vars["s3path_base"] = s3_base
        session.vars["practice_s3path_base"] = f"{s3_base}/practice"

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
        if isinstance(ic, list):
            session.vars["interpreter_choices"] = ic
        elif isinstance(ic, str):
            session.vars["interpreter_choices"] = [x.strip() for x in ic.split(";") if x.strip()]
        else:
            session.vars["interpreter_choices"] = []

        session.vars["allowed_values"] = settings.get("allowed_values", []) or []
        session.vars["allowed_regexes"] = settings.get("allowed_regex", []) or []

        if session.config.get("completion_code"):
            session.vars["completion_code"] = str(session.config["completion_code"])

        # ---------------- practice pages (Practice1..Practice7) ----------------
        ext = session.vars["extension"]
        practice_base = session.vars["practice_s3path_base"].rstrip("/")

        for i in range(1, 8):
            key = f"Practice{i}"
            enabled = _truthy(settings.get(key, "1"))  # default ON if missing
            if not enabled:
                continue

            conf = settings.get(key)
            if not isinstance(conf, dict):
                # when loaded from sheet, Practice{i} is a dict; if missing, skip
                logger.warning(f"{key} not found as dict in settings (did you name the tab practice_{i}?)")
                continue

            practice_data = dict(conf)

            # attach full image URL if present
            img = clean_str(practice_data.get("image"))
            if img:
                img = img.replace(" ", "_")
                if not img.lower().endswith(f".{ext}"):
                    img = f"{img}.{ext}"
                practice_data["image_url"] = f"{practice_base}/{img}"
            else:
                # practice 6 & 7: allowed to be missing
                practice_data["image_url"] = ""

            session.vars[key] = practice_data
            logger.info(f"Loaded {key} (image_url={practice_data.get('image_url')})")

        # ---------------- determine Excel slot universe ----------------
        slot_ids = set()
        for r in rows:
            p = safe_int(r.get("Producer"), 0)
            i = safe_int(r.get("Interpreter"), 0)
            if p != 0 and p != 9:
                slot_ids.add(p)
            if i != 0:
                slot_ids.add(i)

        if not slot_ids:
            raise RuntimeError("No Producer/Interpreter IDs found in data rows.")

        K = max(slot_ids)
        players = subsession.get_players()

        logger.info(f"Players in session={len(players)}, max slot ID in sheet={K}, slots={sorted(slot_ids)[:10]}...")

        # IMPORTANT: don't silently run with empty schedules
        if len(players) > K:
            raise RuntimeError(
                f"Session has {len(players)} participants but sheet only contains slots 1..{K}. "
                f"Fix by (a) creating a session with {K} participants, or (b) ensure your data tab "
                f"contains Producer/Interpreter IDs up to {len(players)}."
            )

        # Map excel slots 1..K to first K participants
        slot_to_pid = {}
        pid_to_slot = {}
        for idx, pl in enumerate(players, start=1):
            if idx <= K:
                slot_to_pid[idx] = pl.id_in_subsession
                pid_to_slot[pl.id_in_subsession] = idx

        # ---------------- valid image pool ----------------
        valid_pool = []
        for r in rows:
            producer_slot = safe_int(r.get("Producer"), 0)
            img = clean_str(r.get("Item"))
            if producer_slot != 0 and producer_slot != 9 and is_valid_real_image(img):
                valid_pool.append(img)

        if not valid_pool:
            logger.warning("No valid images found; using fallback.")
            valid_pool = ["d-A-B-BC-3"]

        # ---------------- build schedule ----------------
        from collections import defaultdict
        data_by_pid = defaultdict(list)

        for idx, r in enumerate(rows):
            exp_num = _pick_exp_value(r)
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
                # 3 producer
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

                # 5 interpreter
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
            raise RuntimeError(f"Empty schedules for participants: {empty}")

        logger.info(f"Schedule built successfully. players={len(players)} valid_pool={len(valid_pool)} rows={len(rows)}")

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


page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
