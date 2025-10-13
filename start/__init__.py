from otree.api import *
import logging, re, json

logger = logging.getLogger('benzapp.start_pages')

# Optional image URL helper from your codebase; fall back to /static/
try:
    from img_desc.utils import get_url_for_image as _img_url_helper
except Exception:  # pragma: no cover
    _img_url_helper = None


def _full_image_url(player, rel_path: str) -> str:
    if _img_url_helper:
        try:
            return _img_url_helper(player, rel_path)
        except Exception as e:
            logger.warning(f'get_url_for_image failed for {rel_path}: {e}')
    return f'/static/start/{rel_path}'


# -------- helpers to read key–value sheets (data, practice_1, …) --------

def _normalize_listish(val):
    """Turn 'A;B;C' or '["A","B"]' or list -> ['A','B',...] (as strings)."""
    if isinstance(val, list):
        return [str(x).strip() for x in val]
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return []
        if s.startswith('['):
            try:
                return [str(x).strip() for x in json.loads(s)]
            except Exception:
                pass
        return [p.strip() for p in s.split(';') if p.strip() != '']
    return []


def _practice_from_kv_rows(rows):
    """
    rows: list of dicts with columns [name, value, ...].
    Supports right_answer_1 … right_answer_N and/or a single 'right_answer'.
    """
    def get(row, key):
        return row.get(key) or row.get(key.capitalize()) or row.get(key.upper())

    kv = {}
    right_pairs = []
    for r in rows or []:
        name = str(get(r, 'name') or '').strip().lower()
        val = get(r, 'value')
        if not name:
            continue
        if name.startswith('right_answer_'):
            m = re.search(r'right_answer_(\d+)', name)
            idx = int(m.group(1)) if m else 10**6
            right_pairs.append((idx, str(val)))
        else:
            kv[name] = val

    right_pairs.sort(key=lambda t: t[0])
    right_answer = [str(v).strip() for _, v in right_pairs]
    if not right_answer and 'right_answer' in kv:
        right_answer = _normalize_listish(kv['right_answer'])

    return {
        'image': (str(kv.get('image') or kv.get('filename') or kv.get('img') or '')).strip(),
        'title': (str(kv.get('title') or '')).strip(),
        'main_text': (str(kv.get('main_text') or kv.get('text') or '')).strip(),
        'right_answer': right_answer,
    }


def _meta_from_data_sheet(rows):
    """Read interpreter_title / interpreter_choices from 'data' tab."""
    def get(row, key):
        return row.get(key) or row.get(key.capitalize()) or row.get(key.upper())

    kv = {}
    for r in rows or []:
        name = str(get(r, 'name') or '').strip().lower()
        val = get(r, 'value')
        if name:
            kv[name] = val

    title = kv.get('interpreter_title') or kv.get('title') or 'Interpretation'
    choices = _normalize_listish(kv.get('interpreter_choices') or kv.get('choices') or [])
    return title, choices


# ---------------- oTree models/pages ----------------

class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    def creating_session(self):
        sv = self.session.vars

        # Your loader should put all tabs into one dict; accept several keys for compatibility
        sheets = sv.get('sheet_data') or sv.get('practices') or sv.get('sheets') or {}

        # ---- meta (title/choices) from 'data' tab (key–value layout) ----
        if 'interpreter_title' not in sv or 'interpreter_choices' not in sv:
            ititle, ichoices = _meta_from_data_sheet(sheets.get('data') or sheets.get('Data') or [])
            sv['interpreter_title'] = ititle
            sv['interpreter_choices'] = ichoices if ichoices else ['Yes', 'No']

        # ---- normalize practice_* tabs into one mapping ----
        practice_settings = {}
        for tab_name, rows in (sheets.items() if isinstance(sheets, dict) else []):
            t = str(tab_name).lower()
            if not t.startswith('practice_'):
                continue

            settings = {}
            # Key–value layout (your current workbook)
            if isinstance(rows, list) and rows and \
               {'name', 'value'}.issubset({str(k).lower() for k in rows[0].keys()}):
                settings = _practice_from_kv_rows(rows)

            # Record-row fallback (kept for backward compatibility)
            elif isinstance(rows, list) and rows:
                row = rows[0]
                settings = {
                    'image': (row.get('filename') or row.get('image') or row.get('img') or '').strip(),
                    'title': (row.get('title') or tab_name).strip(),
                    'main_text': (row.get('main_text') or row.get('text') or '').strip(),
                    'right_answer': _normalize_listish(row.get('right_answer') or row.get('answers')),
                }

            # ensure required keys exist
            settings.setdefault('title', tab_name)
            settings.setdefault('main_text', '')
            settings.setdefault('right_answer', [])
            settings.setdefault('image', '')

            practice_settings[t] = settings

        sv['practice_settings'] = practice_settings

        sv['desc'] = (
            'Practice data prepared from key–value sheets.'
            if practice_settings else
            'No practice_* tabs recognized (expected [name,value] columns).'
        )


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    survey_data = models.LongStringField(blank=True)


# ---------------- shared base page ----------------

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


# ---------------- practice pages ----------------

class _PracticePage(_BasePage):
    instructions = True
    practice_id = None  # set on subclasses

    @classmethod
    def _get_settings(cls, player: Player) -> dict:
        key = f'practice_{cls.practice_id}'
        s = (player.session.vars.get('practice_settings', {}).get(key, {}) or {}).copy()

        img = s.get('image')
        s['full_image_path'] = _full_image_url(player, f'practice/{img}') if img else ''
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
        if pps:
            return pps.get(cls.__name__, True)
        return True


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
