from otree.api import *
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger('benzapp.start_pages')

# Optional helper for building public image URLs (keeps your old behavior if present)
try:
    from img_desc.utils import get_url_for_image as _img_url_helper
except Exception:
    _img_url_helper = None

# Try to import pandas for Excel loading
try:
    import pandas as pd  # requires pandas + openpyxl in requirements.txt
except Exception:
    pd = None


def _full_image_url(player, rel_path: str) -> str:
    """Return a browser-loadable URL for an image path ('practice/foo.png')."""
    if _img_url_helper:
        try:
            return _img_url_helper(player, rel_path)
        except Exception as e:
            logger.warning(f'get_url_for_image failed for {rel_path}: {e}')
    # Fallback; adjust if your static layout differs
    return f'/static/start/{rel_path}'


def _find_local_data_file(name: str | None) -> str | None:
    """Find the Excel/CSV file inside the repo for Heroku dynos."""
    if not name:
        return None
    name = str(name).strip().strip('"').strip("'")
    here = Path(__file__).resolve().parent
    candidates = [
        Path(name),
        here / name,
        here / 'data' / name,
        here.parent / 'data' / name,
        here.parent / name,
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _read_excel_all_sheets(path: str) -> dict[str, "pd.DataFrame"]:
    """Read all sheets from an .xlsx into a dict of DataFrames."""
    if not pd:
        logger.error("pandas not available; cannot load Excel workbook.")
        return {}
    try:
        return pd.read_excel(path, sheet_name=None, engine='openpyxl')
    except Exception as e:
        logger.error(f"Failed to read Excel workbook '{path}': {e}")
        return {}


def _extract_meta(frames: dict) -> tuple[str | None, list[str] | None]:
    """
    Look for a sheet with columns 'key'/'value' to get:
      - interpreter_title
      - interpreter_choices (semicolon-separated)
    """
    ititle, ichoices = None, None
    for name, df in frames.items():
        if not hasattr(df, 'columns'):
            continue
        cols_lc = [str(c).strip().lower() for c in df.columns.tolist()]
        if 'key' in cols_lc and 'value' in cols_lc:
            k_idx, v_idx = cols_lc.index('key'), cols_lc.index('value')
            for _, row in df.iterrows():
                key = str(row.iloc[k_idx]).strip().lower()
                val = row.iloc[v_idx]
                if key == 'interpreter_title' and pd.notna(val):
                    ititle = str(val).strip()
                elif key == 'interpreter_choices' and pd.notna(val):
                    ichoices = [s.strip() for s in str(val).split(';') if s.strip()]
    return ititle, ichoices


def _normalize_practice_row(row: dict, idx: int) -> dict:
    """
    Accepts a dict-like row and normalizes keys:
      filename/image, title, main_text, right_answer (semicolon '1;0;1' or list)
    """
    # flexible column aliases
    def pick(*names):
        for n in names:
            for k, v in row.items():
                if str(k).strip().lower() == n:
                    return v
        return None

    image = pick('filename', 'image', 'img', 'picture', 'file')
    title = pick('title', 'page_title', 'name') or f'Practice {idx}'
    main_text = pick('main_text', 'text', 'html', 'instructions', 'body') or ''

    ra = pick('right_answer', 'answers', 'solution', 'key')
    if isinstance(ra, str):
        right_answer = [s.strip() for s in ra.split(';') if s.strip() != '']
    elif isinstance(ra, (list, tuple)):
        right_answer = [str(x).strip() for x in ra]
    else:
        right_answer = []

    return {
        'image': str(image).strip() if image is not None else '',
        'title': str(title).strip(),
        'main_text': str(main_text),
        'right_answer': right_answer,
    }


def _build_practice_from_workbook(frames: dict) -> dict:
    """
    Collect sheets named like 'Practice', 'Practice 1', 'practice_2', etc.
    Take the first non-empty row of each sheet and normalize it.
    """
    if not frames:
        return {}

    practice = {}
    # sort so 'Practice', 'Practice 1', 'Practice 2' come in numeric order
    def _practice_key(name: str):
        m = re.match(r'(?i)practice[_\s]*(\d+)?', name.strip())
        if m:
            num = m.group(1)
            return (0, int(num)) if num and num.isdigit() else (0, 0)
        return (1, name.lower())

    for sheet_name in sorted(frames.keys(), key=_practice_key):
        if not re.match(r'(?i)^practice', sheet_name.strip()):
            continue
        df = frames[sheet_name]
        if df is None or df.empty:
            continue
        # Use the first non-all-NA row as the source
        row = df.dropna(how='all')
        if row.empty:
            continue
        first = row.iloc[0].to_dict()
        idx_match = re.match(r'(?i)practice[_\s]*(\d+)?', sheet_name.strip())
        idx = int(idx_match.group(1)) if (idx_match and idx_match.group(1) and idx_match.group(1).isdigit()) else len(practice) + 1
        practice[f'practice_{idx}'] = _normalize_practice_row(first, idx)

    return practice


class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    def creating_session(self):
        """
        On session start, load disjunctionExpTest8EN_1.xlsx (multi-worksheet) and
        prepare:
          - session.vars['practice_settings'] = { practice_1: {...}, ... }
          - session.vars['interpreter_title'], session.vars['interpreter_choices']
        """
        sv = self.session.vars

        # If already loaded (e.g., by another app), don't reload.
        if isinstance(sv.get('practice_settings'), dict) and sv['practice_settings']:
            return

        filename = self.session.config.get('filename')
        path = None
        if filename and str(filename).lower().endswith(('.xlsx', '.xls')):
            path = _find_local_data_file(filename)

        frames = {}
        if path:
            frames = _read_excel_all_sheets(path)
            if not frames:
                logger.warning(f"Workbook found but empty or unreadable: {path}")
        else:
            logger.warning(
                f"No Excel file found for 'filename'={filename!r}. "
                f"Place disjunctionExpTest8EN_1.xlsx in the repo and set filename to that exact name."
            )

        # meta (optional)
        ititle, ichoices = _extract_meta(frames)
        sv['interpreter_title'] = ititle or self.session.config.get('interpreter_title') or 'Interpretation'
        sv['interpreter_choices'] = (
            ichoices or self.session.config.get('interpreter_choices') or ['Choice 1', 'Choice 2', 'Choice 3']
        )

        # practice settings from sheets
        practice_settings = _build_practice_from_workbook(frames)
        if not practice_settings:
            logger.warning("No practice sheets detected. Expect sheets named 'Practice', 'Practice 1', ...")
        sv['practice_settings'] = practice_settings

        # Small breadcrumb for your /SessionStartLinks debug view
        sv['desc'] = 'Practice data prepared from Excel.' if practice_settings else 'Excel loaded but no Practice sheets found.'


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # Demographics JSON blob (from your hidden input)
    survey_data = models.LongStringField(blank=True)


# ------------------- Common base page (progress + instructions flag) -------------------

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
    # Template should write JSON into:
    #   <input type="hidden" name="survey_data" id="survey_data">
    form_model = 'player'
    form_fields = ['survey_data']


class Instructions(_BasePage):
    instructions = True


# -------------------------------- Practice pages --------------------------------------

class _PracticePage(_BasePage):
    instructions = True
    practice_id = None  # set in subclasses

    @classmethod
    def _get_settings(cls, player: Player) -> dict:
        s = (player.session.vars.get('practice_settings', {})
             .get(f'practice_{cls.practice_id}', {})).copy()
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
