from otree.api import *
import logging, re
from pathlib import Path

logger = logging.getLogger("benzapp.start_pages")

# Optional helper from your repo; falls back to /static if not present
try:
    from img_desc.utils import get_url_for_image as _img_url_helper
except Exception:
    _img_url_helper = None


def _full_image_url(player, rel_path: str) -> str:
    if _img_url_helper:
        try:
            return _img_url_helper(player, rel_path)
        except Exception as e:
            logger.warning(f"get_url_for_image failed for {rel_path}: {e}")
    return f"/static/start/{rel_path}"


# ---------- Excel loader (practice_1, practice_2, …) ----------
# Needs: pandas, openpyxl in requirements.txt
import pandas as pd  # noqa: E402


def _num_from_name(name: str) -> int:
    m = re.search(r'(\d+)', name or '')
    return int(m.group(1)) if m else 0


def _kv_sheet_to_dict(df) -> dict:
    """Expect 2 cols named 'name' and 'value' (case-insensitive)."""
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if not {'name', 'value'}.issubset(df.columns):
        return {}
    out = {}
    for _, r in df.iterrows():
        k = str(r.get('name') or '').strip()
        if not k:
            continue
        v_raw = r.get('value')
        v = '' if pd.isna(v_raw) else str(v_raw).strip()
        out[k] = v
    return out


def _load_practice_from_xlsx(xlsx_filename: str):
    """
    Read practice_N tabs (N >= 1) with key/value rows like your screenshot.
    Keys:
      title, main_text, image, right_answer_1, right_answer_2, ...
    Optionally, a 'meta' or 'settings' tab (key/value) with:
      interpreter_title, interpreter_choices (semicolon-separated).
    """
    # Look in repo root, /data, or /start/data
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / xlsx_filename,
        root / 'data' / xlsx_filename,
        root / 'start' / 'data' / xlsx_filename,
    ]
    xlsx_path = next((p for p in candidates if p.exists()), None)
    if not xlsx_path:
        logger.warning(f'XLSX not found: {xlsx_filename} (searched {candidates})')
        return {}, {}

    book = pd.read_excel(xlsx_path, sheet_name=None, dtype=str, engine='openpyxl')

    # --- optional meta ---
    meta = {}
    for tab in ('meta', 'settings'):
        if tab in book:
            meta.update(_kv_sheet_to_dict(book[tab]))

    # --- practice_N sheets ---
    practice_settings = {}
    for sheet_name, df in book.items():
        s = str(sheet_name).strip().lower()
        if not s.startswith('practice_'):
            continue
        kv = _kv_sheet_to_dict(df)
        if not kv:
            continue

        # normalize fields
        title = kv.get('title', f'Practice {_num_from_name(s)}')
        main_text = kv.get('main_text', '')

        img = kv.get('image', '')
        if img and ('.' not in img):  # allow plain 'c' → assume png
            img = f'{img}.png'

        # collect ordered right_answers by suffix number
        ra_pairs = []
        for k, v in kv.items():
            m = re.fullmatch(r'right_answer_(\d+)', k)
            if m:
                idx = int(m.group(1))
                ra_pairs.append((idx, v))
        ra_pairs.sort(key=lambda t: t[0])
        right_answer = [p[1] for p in ra_pairs]

        practice_settings[s] = {
            'title': title,
            'main_text': main_text,
            'image': img,
            'right_answer': right_answer,
        }

    # defaults if meta missing
    if 'interpreter_title' not in meta:
        meta['interpreter_title'] = 'Interpretation'
    if 'interpreter_choices' not in meta:
        # derive number of choices from first practice’s answers
        first = practice_settings.get('practice_1', {})
        n = max(1, len(first.get('right_answer', [])))
        meta['interpreter_choices'] = ';'.join([f'Choice {i}' for i in range(1, n + 1)])

    logger.info(f'Loaded practice tabs: {sorted(practice_settings.keys())}')
    return practice_settings, meta


# ---------------- oTree models ----------------

class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    def creating_session(self):
        sv = self.session.vars

        # 1) Try Excel (explicit) — set this in Session -> Configure settings:
        #    practice_xlsx: "disjunctionExpTest8EN_1.xlsx"
        xlsx_name = self.session.config.get('practice_xlsx')
        if xlsx_name:
            ps, meta = _load_practice_from_xlsx(xlsx_name)
            if ps:
                sv['practice_settings'] = ps
                # interpreter meta
                sv['interpreter_title'] = meta.get('interpreter_title')
                sv['interpreter_choices'] = [
                    s.strip() for s in meta.get('interpreter_choices', '').split(';') if s.strip()
                ]

        # 2) Fallback: if nothing loaded, keep empty structures (your page shows blank)
        sv.setdefault('practice_settings', {})
        sv.setdefault('interpreter_title', 'Interpretation')
        sv.setdefault('interpreter_choices', ['Choice 1', 'Choice 2'])

        # Tiny flag for debug panel
        sv.setdefault(
            'desc',
            'Practice data prepared.' if sv['practice_settings'] else 'No practice_* sheets found in Excel.',
        )


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    survey_data = models.LongStringField(blank=True)


# ---------------- Base page & pages ----------------

class _BasePage(Page):
    instructions = False
    instructions_path = "start/includes/instructions.html"

    def get_context_data(self, **context):
        r = super().get_context_data(**context)
        r['instructions_path'] = self.instructions_path
        r['instructions_google_doc'] = self.session.config.get('instructions_path')
        max_idx = getattr(self.participant, '_max_page_index', 1) or 1
        try:
            r['progress'] = f"{int(self._index_in_pages / max_idx * 100):d}"
        except Exception:
            r['progress'] = "0"
        r['instructions'] = self.instructions
        return r


class Consent(_BasePage):
    pass


class Demographics(_BasePage):
    form_model = 'player'
    form_fields = ['survey_data']


class Instructions(_BasePage):
    instructions = True


class _PracticePage(_BasePage):
    instructions = True
    practice_id = None  # override in subclasses

    @classmethod
    def _get_settings(cls, player: Player) -> dict:
        key = f'practice_{cls.practice_id}'
        s = (player.session.vars.get('practice_settings', {}).get(key, {})).copy()
        # image URL
        img = s.get('image') or ''
        if img:
            s['full_image_path'] = _full_image_url(player, f'practice/{img}')
        else:
            # Provide inline SVG placeholder when no image
            placeholder_svg = (
                '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="400">'
                '<rect width="100%" height="100%" fill="#eee"/>'
                '<text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"'
                ' font-family="Arial, Helvetica, sans-serif" font-size="28" fill="#333">'
                f'Practice {cls.practice_id} Image'
                '</text>'
                '</svg>'
            )
            s['full_image_path'] = 'data:image/svg+xml;utf8,' + placeholder_svg.replace(' ', '%20').replace('<', '%3C').replace('>', '%3E').replace('"', '%22')
        # ensure keys exist
        s.setdefault('title', f'Practice {cls.practice_id}')
        s.setdefault('main_text', '')
        s.setdefault('right_answer', [])
        return s

    @classmethod
    def vars_for_template(cls, player: Player):
        return dict(settings=cls._get_settings(player))

    @classmethod
    def js_vars(cls, player: Player):
        return dict(settings=cls._get_settings(player))

    @classmethod
    def is_displayed(cls, player: Player):
        pps = player.session.vars.get('user_settings', {}).get('practice_pages', {})
        return pps.get(cls.__name__, True) if pps else True


class Practice1(_PracticePage): practice_id = 1
class Practice2(_PracticePage): practice_id = 2
class Practice3(_PracticePage): practice_id = 3
class Practice4(_PracticePage): practice_id = 4
class Practice5(_PracticePage): practice_id = 5
class Practice6(_PracticePage): practice_id = 6
class Practice7(_PracticePage): practice_id = 7


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
