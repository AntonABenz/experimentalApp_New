from otree.api import *
import pandas as pd
import re
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _num_from_name(name: str):
    """Convert 'Practice3' → 3."""
    m = re.search(r'(\d+)', name or '')
    return int(m.group(1)) if m else 0


def _kv_sheet_to_dict(df):
    """Convert 2-column sheet (name/value) to dict."""
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
    if not {'name', 'value'}.issubset(df.columns):
        return {}
    out = {}
    for _, r in df.iterrows():
        key = str(r['name']).strip()
        if key:
            val = '' if pd.isna(r['value']) else str(r['value']).strip()
            out[key] = val
    return out


def _load_practice_from_xlsx(xlsx_name: str):
    """Loads Settings sheet + Practice_1, Practice_2 ... tabs."""

    root = Path(__file__).resolve().parents[1]

    candidates = [
        root / xlsx_name,
        root / 'start' / 'data' / xlsx_name,
        root / 'data' / xlsx_name,
    ]

    xlsx_path = next((p for p in candidates if p.exists()), None)
    if not xlsx_path:
        logger.error(f"Excel file '{xlsx_name}' not found in {candidates}")
        return {}, {}

    book = pd.read_excel(xlsx_path, sheet_name=None, dtype=str, engine='openpyxl')

    # -------------------- Settings sheet --------------------
    meta = {}
    if 'Settings' in book:
        meta = _kv_sheet_to_dict(book['Settings'])

    # Defaults
    if 'interpreter_title' not in meta:
        meta['interpreter_title'] = 'Interpretation'

    if 'interpreter_choices' in meta:
        meta['interpreter_choices'] = [
            s.strip() for s in meta['interpreter_choices'].split(';') if s.strip()
        ]
    else:
        meta['interpreter_choices'] = ['Choice 1', 'Choice 2']

    # S3 base URL
    s3_base = meta.get('s3path_base', '').rstrip('/')
    meta['s3path_base'] = s3_base

    # -------------------- Practice sheets --------------------
    practice = {}

    for sheet_name, df in book.items():
        name_clean = str(sheet_name).strip().lower()
        if not name_clean.startswith('practice'):
            continue

        kv = _kv_sheet_to_dict(df)
        if not kv:
            continue

        pid = _num_from_name(sheet_name)

        # Extract everything
        title = kv.get('title', f'Practice {pid}')
        main_text = kv.get('main_text', '')

        img = kv.get('image') or kv.get('s3path') or ''
        if img and not img.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = img + '.png'

        # Right answers: right_answer_1, _2, _3...
        right_answer = []
        for k, v in kv.items():
            if k.startswith('right_answer_'):
                idx = int(k.split('_')[2])
                right_answer.append((idx, v))

        right_answer.sort(key=lambda x: x[0])
        right_answer = [v for _, v in right_answer]

        practice[f'practice_{pid}'] = dict(
            title=title,
            main_text=main_text,
            image=img,
            right_answer=right_answer,
        )

    logger.info(f"Loaded practice tabs: {list(practice.keys())}")
    return practice, meta


# -------------------------------------------------------------------------
# oTree Models
# -------------------------------------------------------------------------

class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


def creating_session(subsession: BaseSubsession):
    config = subsession.session.config
    xlsx = config.get('practice_xlsx')

    if not xlsx:
        logger.error("No practice_xlsx in SESSION_CONFIG!")
        return

    practice_settings, sheet_settings = _load_practice_from_xlsx(xlsx)

    subsession.session.vars['practice_settings'] = practice_settings
    subsession.session.vars['sheet_settings'] = sheet_settings
    subsession.session.vars['interpreter_title'] = sheet_settings.get('interpreter_title')
    subsession.session.vars['interpreter_choices'] = sheet_settings.get('interpreter_choices')

    logger.info("Practice data fully loaded.")


def _build_image_url(session, filename):
    """Return full S3 URL."""
    base = session.vars.get('sheet_settings', {}).get('s3path_base', '')
    if base:
        return f"{base}/{filename}"
    return f"/static/start/{filename}"


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    survey_data = models.LongStringField(blank=True)


# -------------------------------------------------------------------------
# Pages
# -------------------------------------------------------------------------

class _BasePage(Page):
    instructions = False
    instructions_path = "start/includes/instructions.html"

    def get_context_data(self, **context):
        d = super().get_context_data(**context)
        d['instructions_google_doc'] = self.session.config.get('instructions_path', '')
        return d


class Consent(_BasePage):
    pass


class Demographics(_BasePage):
    form_model = 'player'
    form_fields = ['survey_data']


class Instructions(_BasePage):
    instructions = True


class _PracticePage(_BasePage):
    practice_id = None
    template_name = 'start/Practice1.html'

    @classmethod
    def _settings(cls, player):
        key = f'practice_{cls.practice_id}'
        data = player.session.vars['practice_settings'].get(key, {}).copy()

        img_file = data.get('image', '')
        if img_file:
            data['full_image_path'] = _build_image_url(player.session, img_file)
        else:
            data['full_image_path'] = f"https://picsum.photos/200/300?text=Practice+{cls.practice_id}"

        data.setdefault('right_answer', [])
        return data

    @classmethod
    def vars_for_template(cls, player):
        return dict(settings=cls._settings(player))

    @classmethod
    def js_vars(cls, player):
        return dict(settings=cls._settings(player))


# Auto-generate all practice pages 1–7
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
