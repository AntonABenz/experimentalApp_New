# start/__init__.py
from otree.api import *
import logging
from pathlib import Path
import pandas as pd
import re

logger = logging.getLogger('benzapp.start_pages')

# Optional helper already in your repo; we fall back to /static if missing.
try:
    from img_desc.utils import get_url_for_image as _img_url_helper
except Exception:  # pragma: no cover
    _img_url_helper = None


def _full_image_url(player, rel_path: str) -> str:
    """
    Build a browser-loadable URL for a file under /static.
    We prefer your existing helper if present.
    """
    if _img_url_helper:
        try:
            return _img_url_helper(player, rel_path)
        except Exception as e:
            logger.warning(f'get_url_for_image failed for {rel_path}: {e}')
    return f'/static/start/{rel_path}'


# ---------- XLSX loader (reads practice_1, practice_2, ... sheets) --------------

_num_re = re.compile(r'(\d+)')

def _extract_num_from_sheetname(name: str) -> int:
    m = _num_re.search(name or '')
    return int(m.group(1)) if m else 0


def _load_practice_from_xlsx(filename: str):
    """
    Read practice_{n} sheets that use a 2-column 'name'/'value' layout
    (like your screenshot). Also optionally read a meta/settings sheet
    that contains interpreter_title and interpreter_choices.
    """
    base = Path(__file__).resolve().parents[1]
    candidates = [
        base / filename,
        base / 'data' / filename,
        base / 'start' / 'data' / filename,
    ]
    xlsx_path = next((p for p in candidates if p.exists()), None)
    if not xlsx_path:
        logger.warning(f'XLSX not found: {filename} (looked in {candidates})')
        return {}, {}

    xls = pd.read_excel(xlsx_path, sheet_name=None, dtype=str)
    practice_settings: dict[str, dict] = {}
    meta: dict[str, object] = {}

    # Optional meta/settings sheet with name/value pairs
    for meta_tab in ('meta', 'settings', 'practice', 'data'):
        df = xls.get(meta_tab)
        if df is None:
            continue
        cols = {str(c).strip().lower() for c in df.columns}
        if not {'name', 'value'}.issubset(cols):
            continue
        df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
        kv = {str(r['name']).strip(): ('' if pd.isna(r.get('value')) else str(r['value']).strip())
              for _, r in df.iterrows() if pd.notna(r.get('name'))}
        if 'interpreter_title' in kv:
            meta['interpreter_title'] = kv['interpreter_title']
        if 'interpreter_choices' in kv:
            meta['interpreter_choices'] = [s.strip() for s in kv['interpreter_choices'].split(';') if s.strip()]
        break

    # Practice sheets: practice_1, practice_2, ...
    for sheet_name, df in xls.items():
        if not str(sheet_name).lower().startswith('practice'):
            continue
        cols = {str(c).strip().lower() for c in df.columns}
        if not {'name', 'value'}.issubset(cols):
            continue

        df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})
        kv = {str(r['name']).strip(): ('' if pd.isna(r.get('value')) else str(r['value']).strip())
              for _, r in df.iterrows() if pd.notna(r.get('name'))}

        n = _extract_num_from_sheetname(str(sheet_name))
        settings = {
            'image': kv.get('image', ''),
            'title': kv.get('title', f'Practice {n}'),
            'main_text': kv.get('main_text', ''),
        }
        # right_answer_1..N (strings like "1" or "0")
        ra = []
        i = 1
        while f'right_answer_{i}' in kv:
            v = kv.get(f'right_answer_{i}', '').strip()
            if v != '':
                ra.append(v)
            i += 1
        settings['right_answer'] = ra

        practice_settings[f'practice_{n}'] = settings

    return practice_settings, meta


# --------------------------------- oTree models ---------------------------------

class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    def creating_session(self):
        sv = self.session.vars

        # defaults used by your Vue template
        sv.setdefault('interpreter_title', self.session.config.get('interpreter_title') or 'Interpretation')
        sv.setdefault(
            'interpreter_choices',
            list(self.session.config.get('interpreter_choices') or ['Choice 1', 'Choice 2', 'Choice 3'])
        )

        # Try XLSX if a filename is set in session config
        fname = (self.session.config.get('filename') or '').strip()
        if fname and fname.lower().endswith('.xlsx'):
            practice_settings, meta = _load_practice_from_xlsx(fname)
            if practice_settings:
                sv['practice_settings'] = practice_settings
                if 'interpreter_title' in meta:
                    sv['interpreter_title'] = meta['interpreter_title']
                if 'interpreter_choices' in meta:
                    sv['interpreter_choices'] = meta['interpreter_choices']
            else:
                logger.warning('No practice content found in XLSX; practice_settings is empty.')

        # Human-friendly status text for the debug box you’ve been using
        if sv.get('practice_settings'):
            sv.setdefault('desc', 'Practice data prepared.')
        else:
            sv.setdefault('desc', 'No rows found in sheet_data.')


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # Demographics collected as JSON (hidden field in template)
    survey_data = models.LongStringField(blank=True)


# ------------------------------ Page base class ---------------------------------

class _BasePage(Page):
    instructions = False
    instructions_path = "start/includes/instructions.html"

    def get_context_data(self, **context):
        r = super().get_context_data(**context)
        r['instructions_path'] = self.instructions_path
        r['instructions_google_doc'] = self.session.config.get('instructions_path')
        try:
            max_idx = getattr(self.participant, '_max_page_index', 1) or 1
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


# ------------------------------ Practice pages ----------------------------------

class _PracticePage(_BasePage):
    instructions = True
    practice_id = None  # override in subclasses

    @classmethod
    def _get_settings(cls, player: Player) -> dict:
        s = (player.session.vars.get('practice_settings', {})
             .get(f'practice_{cls.practice_id}', {})).copy()

        img = s.get('image')
        if img:
            s['full_image_path'] = _full_image_url(player, f'practice/{img}')
        else:
            s['full_image_path'] = ''

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
