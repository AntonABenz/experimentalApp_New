from otree.api import *
import logging

logger = logging.getLogger('benzapp.start_pages')

# Try to import your existing helper for image URLs; fall back to a static path.
try:
    from img_desc.utils import get_url_for_image as _img_url_helper
except Exception:  # pragma: no cover
    _img_url_helper = None


def _full_image_url(player, rel_path: str) -> str:
    """
    Return a URL that the browser can load.
    Prefer your existing helper; otherwise fall back to /static/.
    """
    if _img_url_helper:
        try:
            return _img_url_helper(player, rel_path)
        except Exception as e:  # keep the app running even if helper fails
            logger.warning(f'get_url_for_image failed for {rel_path}: {e}')
    # fallback – adjust if your static path differs
    return f'/static/start/{rel_path}'


class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    def creating_session(self):
        """
        Build the structures the practice pages expect.

        This function is defensive: it supports several ways your study may
        already store data, and normalizes everything into:
            session.vars['practice_settings'] = {
                'practice_1': {...}, 'practice_2': {...}, ...
            }
        It also sets:
            session.vars['interpreter_title']
            session.vars['interpreter_choices']
        """
        sv = self.session.vars

        # --- 1) Interpreter title/choices for your Vue table header/rows ----
        if 'interpreter_title' not in sv:
            sv['interpreter_title'] = (
                self.session.config.get('interpreter_title')
                or 'Interpretation'
            )

        if 'interpreter_choices' not in sv:
            # either from config or a safe default list
            choices = self.session.config.get('interpreter_choices')
            if isinstance(choices, (list, tuple)) and choices:
                sv['interpreter_choices'] = list(choices)
            else:
                sv['interpreter_choices'] = ['Choice 1', 'Choice 2', 'Choice 3']

        # --- 2) Normalize practice rows into practice_settings ---------------
        if 'practice_settings' in sv and isinstance(sv['practice_settings'], dict):
            return  # already prepared (perhaps by a loader elsewhere)

        practice_settings = {}

        # Case A: you already have a loader that put rows into session.vars['practices']
        # and a schema into session.vars['sheet_meta']['schema'].
        practices_block = sv.get('practices') or {}
        schema = (sv.get('sheet_meta') or {}).get('schema') or {}

        # Columns we care about (fall back to sensible defaults)
        col_filename = (schema.get('filename') or 'filename').strip()
        col_title = (schema.get('title') or 'title').strip()
        col_main_text = (schema.get('main_text') or 'main_text').strip()
        col_right_answer = (schema.get('right_answer') or 'right_answer').strip()

        if practices_block:
            # Expect shape: {'some_tab': [ {col:val, ...}, ...], ...}
            idx = 1
            for tab in sorted(practices_block):
                rows = practices_block.get(tab) or []
                if not rows:
                    continue
                row = rows[0]  # one screen per tab (like your old project)
                settings = {
                    'image': (row.get(col_filename) or '').strip(),
                    'title': (row.get(col_title) or f'Practice {idx}').strip(),
                    'main_text': (row.get(col_main_text) or '').strip(),
                }
                ra = row.get(col_right_answer)
                # right_answer can be '1;0;1' or a list; normalize to list of strings
                if isinstance(ra, str):
                    settings['right_answer'] = [s.strip() for s in ra.split(';') if s.strip() != '']
                elif isinstance(ra, (list, tuple)):
                    settings['right_answer'] = [str(x) for x in ra]
                else:
                    settings['right_answer'] = []

                practice_settings[f'practice_{idx}'] = settings
                idx += 1

        # Case B: nothing in session.vars['practices']; allow direct config rows
        # session.config['practice_rows'] can be a list of dicts with the same keys.
        if not practice_settings:
            conf_rows = self.session.config.get('practice_rows') or []
            if isinstance(conf_rows, list) and conf_rows:
                for i, row in enumerate(conf_rows, start=1):
                    settings = {
                        'image': (row.get('filename') or row.get('image') or '').strip(),
                        'title': (row.get('title') or f'Practice {i}').strip(),
                        'main_text': (row.get('main_text') or '').strip(),
                    }
                    ra = row.get('right_answer')
                    if isinstance(ra, str):
                        settings['right_answer'] = [s.strip() for s in ra.split(';') if s.strip() != '']
                    elif isinstance(ra, (list, tuple)):
                        settings['right_answer'] = [str(x) for x in ra]
                    else:
                        settings['right_answer'] = []
                    practice_settings[f'practice_{i}'] = settings

        sv['practice_settings'] = practice_settings

        if not practice_settings:
            logger.warning('No practice content found: session.vars["practice_settings"] is empty.')

        # This is handy for the debug panel you’ve been looking at.
        if 'desc' not in sv:
            sv['desc'] = 'Practice data prepared.' if practice_settings else 'No rows found in sheet_data.'


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # Keep demographics as a single JSON blob (hidden input in template)
    survey_data = models.LongStringField(blank=True)


# -------------- Common base page (adds progress + instructions flag) -------------

class _BasePage(Page):
    instructions = False  # subclasses can set to True for instruction/practice screens

    def get_context_data(self, **context):
        r = super().get_context_data(**context)
        r['instructions_google_doc'] = self.session.config.get('instructions_path')
        # progress percentage string ("37", etc.)
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
    """
    Your template should collect answers and put JSON into:
        <input type="hidden" name="survey_data" id="survey_data">
    We rely on vanilla oTree form handling (no custom post()).
    """
    form_model = 'player'
    form_fields = ['survey_data']


class Instructions(_BasePage):
    instructions = True


# ------------------------------ Practice pages -----------------------------------

class _PracticePage(_BasePage):
    instructions = True
    practice_id = None  # set in concrete subclasses

    @classmethod
    def _get_settings(cls, player: Player) -> dict:
        """Return normalized settings for this practice page."""
        s = (player.session.vars.get('practice_settings', {})
             .get(f'practice_{cls.practice_id}', {})).copy()

        img = s.get('image')
        if img:
            # Your template expects js_vars.settings.full_image_path
            s['full_image_path'] = _full_image_url(player, f'practice/{img}')
        else:
            s['full_image_path'] = ''

        # Ensure presence of keys your Vue code reads
        s.setdefault('title', f'Practice {cls.practice_id}')
        s.setdefault('main_text', '')
        s.setdefault('right_answer', [])

        return s

    @classmethod
    def vars_for_template(cls, player: Player):
        """
        Make settings available in the Django template too
        (useful for fallback HTML or simple debug includes).
        """
        return dict(settings=cls._get_settings(player))

    @classmethod
    def js_vars(cls, player: Player):
        """
        Your Vue code reads from window.js_vars.settings, so expose the same dict here.
        """
        return dict(settings=cls._get_settings(player))

    @classmethod
    def is_displayed(cls, player: Player):
        """
        Optional per-page gating via session.vars['user_settings']['practice_pages'].
        """
        pps = player.session.vars.get('user_settings', {}).get('practice_pages', {})
        if pps:
            return pps.get(cls.__name__, True)
        return True


class Practice1(_PracticePage):
    practice_id = 1


class Practice2(_PracticePage):
    practice_id = 2


class Practice3(_PracticePage):
    practice_id = 3


class Practice4(_PracticePage):
    practice_id = 4

    @classmethod
    def js_vars(cls, player: Player):
        """
        Keep your special-case split for right answers if needed.
        (Your Practice1 Vue already does Boolean(parseInt(i)) — so returning the raw strings is fine.)
        This override is shown here if you later need to transform shape for p4 only.
        """
        data = super().js_vars(player)
        return data


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
