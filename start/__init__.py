from otree.api import *
import logging, re
from pathlib import Path

import pandas as pd

logger = logging.getLogger("benzapp.start_pages")

# Optional helper from img_desc app (for static files);
# for S3 we will bypass this.
try:
    from img_desc.utils import get_url_for_image as _img_url_helper
except Exception:
    _img_url_helper = None


def _full_image_url(player, rel_path: str) -> str:
    """
    Fallback URL builder: /static/start/<rel_path>
    Used only if we don't have an S3 base URL.
    """
    if _img_url_helper:
        try:
            return _img_url_helper(player, rel_path)
        except Exception as e:
            logger.warning(f"get_url_for_image failed for {rel_path}: {e}")
    return f"/static/start/{rel_path}"


# ---------- Excel helpers ---------------------------------------------------

def _num_from_name(name: str) -> int:
    m = re.search(r"(\d+)", name or "")
    return int(m.group(1)) if m else 0


def _kv_sheet_to_dict(df) -> dict:
    """
    Expect columns 'name' and 'value' (case-insensitive).
    Returns {name: value} mapping.
    """
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if not {"name", "value"}.issubset(df.columns):
        return {}
    out = {}
    for _, r in df.iterrows():
        k = str(r.get("name") or "").strip()
        if not k:
            continue
        v_raw = r.get("value")
        v = "" if pd.isna(v_raw) else str(v_raw).strip()
        out[k] = v
    return out


def _load_practice_from_xlsx(xlsx_filename: str):
    """
    Load:
      - Settings/meta sheet  → meta dict (includes s3path, interpreter_* ...)
      - Practice_N sheets    → practice_settings['practice_n'] = {...}
    """

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

    meta: dict = {}
    practice_settings: dict = {}

    for sheet_name, df in book.items():
        sname = str(sheet_name).strip()
        s_lower = sname.lower()

        # ---- Settings / Meta sheet (case-insensitive) ----
        if s_lower in ("settings", "meta"):
            logger.info(f"Reading meta from sheet '{sname}'")
            meta.update(_kv_sheet_to_dict(df))
            continue

        # ---- Practice_N sheets ----
        if not s_lower.startswith("practice_"):
            continue

        kv = _kv_sheet_to_dict(df)
        if not kv:
            continue

        title = kv.get("title", f"Practice {_num_from_name(s_lower)}")
        main_text = kv.get("main_text", "")

        img = kv.get("image", "").strip()
        if img and "." not in img:
            img = f"{img}.png"

        # collect ordered right_answer_*
        ra_pairs = []
        for k, v in kv.items():
            m = re.fullmatch(r"right_answer_(\d+)", k)
            if m:
                idx = int(m.group(1))
                ra_pairs.append((idx, v))
        ra_pairs.sort(key=lambda t: t[0])
        right_answer = [p[1] for p in ra_pairs]

        practice_settings[s_lower] = {
            "title": title,
            "main_text": main_text,
            "image": img,
            "right_answer": right_answer,
        }

    # Defaults if meta missing
    if "interpreter_title" not in meta:
        meta["interpreter_title"] = "Interpretation"
    if "interpreter_choices" not in meta:
        first = practice_settings.get("practice_1", {})
        n = max(1, len(first.get("right_answer", [])))
        meta["interpreter_choices"] = ";".join(
            [f"Choice {i}" for i in range(1, n + 1)]
        )

    logger.info(f"Loaded practice tabs from Excel: {sorted(practice_settings.keys())}")
    return practice_settings, meta


# ---------------- oTree models ----------------------------------------------

class C(BaseConstants):
    NAME_IN_URL = "start"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


def creating_session(subsession: BaseSubsession):
    session = subsession.session
    sv = session.vars

    logger.info("creating_session[start]")

    # our SESSION_CONFIG has 'filename'
    xlsx_name = session.config.get("practice_xlsx") or session.config.get("filename")
    logger.info(f"Excel filename from session config = {xlsx_name!r}")

    if xlsx_name:
        ps, meta = _load_practice_from_xlsx(xlsx_name)

        if ps:
            sv["practice_settings"] = ps
        if meta:
            sv["sheet_settings"] = meta

        # interpreter meta
        title = meta.get("interpreter_title")
        choices_raw = meta.get("interpreter_choices")

        if title:
            sv["interpreter_title"] = title
        if choices_raw:
            sv["interpreter_choices"] = [
                s.strip() for s in choices_raw.split(";") if s.strip()
            ]
    else:
        logger.warning("No Excel filename found in session.config")

    # sensible defaults if something went wrong
    sv.setdefault("practice_settings", {})
    sv.setdefault("sheet_settings", {})
    sv.setdefault("interpreter_title", "Interpretation")
    sv.setdefault("interpreter_choices", ["Choice 1", "Choice 2", "Choice 3", "Choice 4"])

    sv.setdefault(
        "desc",
        "Practice data prepared."
        if sv["practice_settings"]
        else "No practice_* sheets found in Excel.",
    )

    logger.info(
        f"Meta from sheet: title={sv['interpreter_title']!r}, "
        f"choices={sv['interpreter_choices']!r}"
    )


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    survey_data = models.LongStringField(blank=True)


# ---------------- Base page & practice pages --------------------------------

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
    template_name = "start/Practice1.html"  # overridden by template names
    instructions = True
    practice_id = None  # override in subclasses (1..7)

    @classmethod
    def _get_settings(cls, player: Player) -> dict:
        key = f"practice_{cls.practice_id}"
        s = (player.session.vars.get("practice_settings", {}).get(key, {})).copy()

        meta = player.session.vars.get("sheet_settings", {}) or {}

        # --- base URL from Settings sheet: s3path / s3_path / image_path / s3_base ---
        base = (
            meta.get("s3path")
            or meta.get("s3_path")
            or meta.get("image_path")
            or meta.get("s3_base")
            or ""
        ).rstrip("/")

        img_url = (s.get("image_url") or "").strip()
        img = (s.get("image") or "").strip()

        # if no extension in 'image', assume .png
        if img and "." not in img:
            img = f"{img}.png"

        # a) explicit full URL overrides everything
        if img_url and (
            img_url.startswith("http://")
            or img_url.startswith("https://")
            or img_url.startswith("s3://")
        ):
            full = img_url
        # b) base from Settings + /practice/ + filename
        elif img and base:
            if "/" in img:
                rel = img
            else:
                rel = f"practice/{img}"
            full = f"{base}/{rel}"
        # c) static fallback
        elif img:
            full = _full_image_url(player, f"practice/{img}")
        else:
            full = f"https://picsum.photos/200/300?text=Practice+{cls.practice_id}"

        s["full_image_path"] = full
        s.setdefault("title", f"Practice {cls.practice_id}")
        s.setdefault("main_text", "")
        s.setdefault("right_answer", [])
        logger.info(f"Practice {cls.practice_id}: image URL = {full}")
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
