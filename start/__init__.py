from otree.api import *
import logging
import re
from pathlib import Path
import pandas as pd  # requires openpyxl in requirements.txt

logger = logging.getLogger("benzapp.start_pages")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def _num_from_name(name: str) -> int:
    """Extract first number from a sheet name, e.g. 'Practice 1' -> 1."""
    m = re.search(r"(\d+)", name or "")
    return int(m.group(1)) if m else 0


def _kv_sheet_to_dict(df) -> dict:
    """
    Interpret a sheet with columns 'name' and 'value' (case insensitive)
    as a {name: value} mapping.
    """
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if not {"name", "value"}.issubset(df.columns):
        return {}

    out = {}
    for _, r in df.iterrows():
        key = str(r.get("name") or "").strip()
        if not key:
            continue
        v_raw = r.get("value")
        v = "" if pd.isna(v_raw) else str(v_raw).strip()
        out[key] = v
    return out


def _load_practice_from_xlsx(xlsx_filename: str):
    """
    Load:
      * meta/settings sheet (s3path, interpreter_choices, interpreter_title, …)
      * practice sheets (Practice 1, Practice_1, practice1, …)

    Practice sheets are expected to have rows:
      - title
      - main_text
      - image  (file name without extension is fine)
      - right_answer_1, right_answer_2, …  (0 or 1)
    """
    # Look in project root, /data, and /start/data
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / xlsx_filename,
        root / "data" / xlsx_filename,
        root / "start" / "data" / xlsx_filename,
    ]
    xlsx_path = next((p for p in candidates if p.exists()), None)
    if not xlsx_path:
        logger.warning(f"XLSX not found: {xlsx_filename} (searched {candidates})")
        return {}, {}

    book = pd.read_excel(xlsx_path, sheet_name=None, dtype=str, engine="openpyxl")

    # ---- meta/settings sheet -------------------------------------------------
    meta = {}
    for sheet_name, df in book.items():
        s_low = str(sheet_name).strip().lower()
        if s_low in {"settings", "meta"}:
            meta.update(_kv_sheet_to_dict(df))

    # ---- practice sheets -----------------------------------------------------
    practice_settings = {}

    for sheet_name, df in book.items():
        name_str = str(sheet_name).strip()
        s_low = name_str.lower()

        # anything that *contains* 'practice' and has a number → treat as practice sheet
        if "practice" not in s_low:
            continue
        n = _num_from_name(s_low)
        if n <= 0:
            continue

        kv = _kv_sheet_to_dict(df)
        if not kv:
            continue

        # Normalised key: 'practice_1', 'practice_2', ...
        key = f"practice_{n}"

        title = kv.get("title", f"Practice {n}")
        main_text = kv.get("main_text", "")

        img = kv.get("image", "") or ""
        if img and "." not in img:
            # allow specifying just 'd-A-B-BC-ABC-Test' → assume .png
            img = f"{img}.png"

        # collect ordered binary right_answer_* values
        ra_pairs = []
        for k, v in kv.items():
            m = re.fullmatch(r"right_answer_(\d+)", k)
            if m:
                idx = int(m.group(1))
                ra_pairs.append((idx, v))
        ra_pairs.sort(key=lambda t: t[0])
        right_answer = [p[1] for p in ra_pairs]

        practice_settings[key] = {
            "title": title,
            "main_text": main_text,
            "image": img,
            "right_answer": right_answer,
        }

    # sensible defaults if some meta missing
    if "interpreter_title" not in meta:
        meta["interpreter_title"] = "Interpretation"

    if "interpreter_choices" not in meta:
        # derive number of choices from first practice’s answers
        first = practice_settings.get("practice_1", {})
        n = max(1, len(first.get("right_answer", [])))
        meta["interpreter_choices"] = ";".join([f"Choice {i}" for i in range(1, n + 1)])

    logger.info(f"Loaded practice tabs: {sorted(practice_settings.keys())}")
    return practice_settings, meta


def _full_image_url(player, img_name: str) -> str:
    """
    Build full image URL for a practice image.
    We prefer the 's3path' value from the Settings sheet.
    Fallback: /static/start/practice/<img_name>
    """
    sess_vars = getattr(player, "session").vars
    sheet_settings = sess_vars.get("sheet_settings", {}) or {}
    base = sheet_settings.get("s3path") or sheet_settings.get("image_path")

    if base:
        base = str(base).rstrip("/")
        # If the base already ends with 'practice', we don’t add it again.
        if base.lower().endswith("/practice"):
            return f"{base}/{img_name}"
        return f"{base}/practice/{img_name}"

    # Local static fallback
    return f"/static/start/practice/{img_name}"


# ---------------------------------------------------------------------
# oTree models
# ---------------------------------------------------------------------


class C(BaseConstants):
    NAME_IN_URL = "start"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


def creating_session(subsession: BaseSubsession):
    self = subsession
    sv = self.session.vars
    logger.info("creating_session in start app")

    cfg = self.session.config
    xlsx_name = (
        cfg.get("practice_xlsx")
        or cfg.get("filename")
        or "disjunctionExpTest8EN_1.xlsx"
    )
    logger.info(f"Excel config value = {xlsx_name}")

    practice_settings, meta = _load_practice_from_xlsx(xlsx_name)

    # store practice sheets
    sv["practice_settings"] = practice_settings or {}

    # store meta/settings (s3path, interpreter_* etc.)
    sv["sheet_settings"] = meta or {}

    # interpreter title
    sv["interpreter_title"] = meta.get("interpreter_title", "Interpretation")

    # interpreter choices (split on ';')
    choices_raw = meta.get("interpreter_choices", "")
    sv["interpreter_choices"] = [
        s.strip() for s in str(choices_raw).split(";") if s.strip()
    ]

    sv.setdefault(
        "desc",
        "Practice data prepared."
        if sv["practice_settings"]
        else "No practice_* sheets found in Excel.",
    )

    logger.info(f"Final interpreter_choices = {sv['interpreter_choices']}")
    logger.info(
        f"Practice1 settings = {sv['practice_settings'].get('practice_1', {})}"
    )


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    survey_data = models.LongStringField(blank=True)


# ---------------------------------------------------------------------
# Base page & practice pages
# ---------------------------------------------------------------------


class _BasePage(Page):
    instructions = False
    instructions_path = "start/includes/instructions.html"

    def get_context_data(self, **context):
        r = super().get_context_data(**context)
        r["instructions_path"] = self.instructions_path
        r["instructions_google_doc"] = self.session.config.get("instructions_path")
        max_idx = getattr(self.participant, "_max_page_index", 1) or 1
        try:
            r["progress"] = f"{int(self._index_in_pages / max_idx * 100):d}"
        except Exception:
            r["progress"] = "0"
        r["instructions"] = self.instructions
        return r


class Consent(_BasePage):
    pass


class Demographics(_BasePage):
    form_model = "player"
    form_fields = ["survey_data"]


class Instructions(_BasePage):
    instructions = True


class _PracticePage(_BasePage):
    """
    Generic practice page. Each concrete subclass only sets practice_id.
    """
    template_name = "start/Practice1.html"  # Vue template shared by all practice pages
    instructions = True
    practice_id = None  # override in subclasses

    @classmethod
    def _get_settings(cls, player: Player) -> dict:
        key = f"practice_{cls.practice_id}"
        s = (player.session.vars.get("practice_settings", {}).get(key, {})).copy()

        # image URL
        img = s.get("image") or ""
        if img:
            s["full_image_path"] = _full_image_url(player, img)
        else:
            s["full_image_path"] = (
                f"https://picsum.photos/600/300?text=Practice+{cls.practice_id}"
            )

        # defaults
        s.setdefault("title", f"Practice {cls.practice_id}")
        s.setdefault("main_text", "")
        s.setdefault("right_answer", [])

        logger.info(f"vars_for_template practice_{cls.practice_id}: {s}")
        return s

    @classmethod
    def vars_for_template(cls, player: Player):
        return dict(settings=cls._get_settings(player))

    @classmethod
    def js_vars(cls, player: Player):
        return dict(settings=cls._get_settings(player))

    @classmethod
    def is_displayed(cls, player: Player):
        pps = player.session.vars.get("user_settings", {}).get("practice_pages", {})
        return pps.get(cls.__name__, True) if pps else True


class Practice1(_PracticePage):
    practice_id = 1


class Practice2(_PracticePage):
    practice_id = 2


class Practice3(_PracticePage):
    practice_id = 3


class Practice4(_PracticePage):
    practice_id = 4


class Practice5(_PracticePage):
    practice_id = 5


class Practice6(_PracticePage):
    practice_id = 6


class Practice7(_PracticePage):
    practice_id = 7


class EndOfIntro(_BasePage):
    pass


page_sequence = [
    Consent,
    Demographics,
    Instructions,
    Practice1,
    Practice2,
    Practice3,
    Practice4,
    Practice5,
    Practice6,
    Practice7,
    EndOfIntro,
]
