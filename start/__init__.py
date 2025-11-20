from otree.api import *
import logging
import re
from pathlib import Path
import pandas as pd  

logger = logging.getLogger("benzapp.start_pages")

# Optional helper from your repo (used to turn /static rel paths into URLs)
try:
    from img_desc.utils import get_url_for_image as _img_url_helper
except Exception:
    _img_url_helper = None


# ------------------------------ helpers --------------------------------------
def _full_image_url(player, rel_path: str) -> str:
    """Resolve a relative static path to a URL, with optional repo helper."""
    if _img_url_helper:
        try:
            return _img_url_helper(player, rel_path)
        except Exception as e:
            logger.warning(f"get_url_for_image failed for {rel_path}: {e}")
    return f"/static/start/{rel_path}"


def _num_from_name(name: str) -> int:
    m = re.search(r'(\d+)', name or '')
    return int(m.group(1)) if m else 0


def _kv_sheet_to_dict(df) -> dict:
    """Expect 2 columns named 'name' and 'value' (case-insensitive)."""
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
    Loads:
      - Settings/meta tab (key/value): may contain s3_base/s3path, interpreter_* etc.
      - Practice_N tabs (key/value rows): title, main_text, image, image_url, right_answer_1..K
    """
    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / xlsx_filename,                  # exact string as passed
        root / 'data' / xlsx_filename,         # repo/data/<file>
        root / 'start' / 'data' / xlsx_filename,  # start/data/<file>
    ]
    xlsx_path = next((p for p in candidates if p.exists()), None)
    if not xlsx_path:
        logger.warning(f'XLSX not found: {xlsx_filename} (searched {candidates})')
        return {}, {}

    book = pd.read_excel(xlsx_path, sheet_name=None, dtype=str, engine='openpyxl')

    # ---- meta/settings (optional) ----
    meta = {}
    for tab in ('meta', 'settings'):
        if tab in book:
            meta.update(_kv_sheet_to_dict(book[tab]))

    # ---- practice_* tabs ----
    practice_settings = {}
    for sheet_name, df in book.items():
        sname = str(sheet_name).strip().lower()
        if not sname.startswith('practice_'):
            continue
        kv = _kv_sheet_to_dict(df)
        if not kv:
            continue

        title = kv.get('title', f'Practice {_num_from_name(sname)}')
        main_text = kv.get('main_text', '')

        img = kv.get('image', '').strip()
        if img and ('.' not in img):  # allow 'c' -> 'c.png'
            img = f'{img}.png'

        image_url = kv.get('image_url', '').strip()  # full URL (optional)

        # collect right_answer_1..K preserving numeric order
        ra_pairs = []
        for k, v in kv.items():
            m = re.fullmatch(r'right_answer_(\d+)', k)
            if m:
                idx = int(m.group(1))
                ra_pairs.append((idx, v))
        ra_pairs.sort(key=lambda t: t[0])
        right_answer = [p[1] for p in ra_pairs]

        practice_settings[sname] = {
            'title': title,
            'main_text': main_text,
            'image': img,
            'image_url': image_url,
            'right_answer': right_answer,
        }

    # sensible defaults for interpreter fields if not in meta
    if 'interpreter_title' not in meta:
        meta['interpreter_title'] = 'Interpretation'
    if 'interpreter_choices' not in meta:
        first = practice_settings.get('practice_1', {})
        n = max(1, len(first.get('right_answer', [])))
        meta['interpreter_choices'] = ';'.join([f'Choice {i}' for i in range(1, n + 1)])

    logger.info(f'Loaded practice tabs: {sorted(practice_settings.keys())}')
    return practice_settings, meta


# ------------------------------ oTree models ---------------------------------
class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


def creating_session(subsession):
    self = subsession
    sv = self.session.vars
    logger.info('creating_session: start app')

    # Accept either 'practice_xlsx' (explicit) or your existing 'filename' key.
    xlsx_name = (
        self.session.config.get('practice_xlsx')
        or self.session.config.get('filename')
    )
    logger.info(f"Excel config value = {xlsx_name}")

    if xlsx_name:
        ps, meta = _load_practice_from_xlsx(xlsx_name)
        if ps:
            sv['practice_settings'] = ps
            sv['sheet_settings'] = meta or {}
            sv['interpreter_title'] = meta.get('interpreter_title', 'Interpretation')
            sv['interpreter_choices'] = [
                s.strip() for s in (meta.get('interpreter_choices') or '').split(';') if s.strip()
            ]
            logger.info("Practice settings & meta loaded into session.vars")
        else:
            logger.warning("Excel found but no practice_* sheets loaded")
    else:
        logger.warning("No Excel specified in session config")

    # Defaults (safe if Excel missing)
    sv.setdefault('practice_settings', {})
    sv.setdefault('sheet_settings', {})
    sv.setdefault('interpreter_title', 'Interpretation')
    sv.setdefault('interpreter_choices', ['Choice 1', 'Choice 2'])
    sv.setdefault(
        'desc',
        'Practice data prepared.' if sv['practice_settings'] else 'No practice_* sheets found in Excel.',
    )


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    survey_data = models.LongStringField(blank=True)


# ------------------------------ Pages ----------------------------------------
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
    template_name = 'start/Practice1.html'  # overridden dynamically
    instructions = True
    practice_id = None  # override in subclasses

    @classmethod
    def _get_settings(cls, player: Player) -> dict:
        key = f'practice_{cls.practice_id}'
        s = (player.session.vars.get('practice_settings', {}).get(key, {})).copy()

        # Build full image URL:
        # 1) Use full row URL if present
        # 2) Else compose s3_base/s3path + /practice/<image>
        # 3) Else /static/start/practice/<image>
        # 4) Else Picsum placeholder
        meta = player.session.vars.get('sheet_settings', {}) or {}
        s3_base = (meta.get('s3_base') or meta.get('s3path') or '').rstrip('/')

        img_url = (s.get('image_url') or '').strip()
        img_fn = (s.get('image') or '').strip()

        if img_url and (img_url.startswith('http://') or img_url.startswith('https://') or img_url.startswith('s3://')):
            full = img_url
        elif img_fn and s3_base:
            full = f"{s3_base}/practice/{img_fn}"
        elif img_fn:
            full = _full_image_url(player, f'practice/{img_fn}')
        else:
            full = f'https://picsum.photos/200/300?text=Practice+{cls.practice_id}'

        s['full_image_path'] = full

        # Ensure keys
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
