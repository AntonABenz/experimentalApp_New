from otree.api import *
import csv
import logging
from pathlib import Path

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
        Load practice screens from a CSV when session.config['filename'] ends with .csv.
        Normalize data into:
          session.vars['practice_settings'] = {'practice_1': {...}, ...}
          session.vars['interpreter_title']
          session.vars['interpreter_choices']
        """

        sv = self.session.vars

        # If something already populated practice_settings, keep it.
        if isinstance(sv.get('practice_settings'), dict) and sv['practice_settings']:
            sv.setdefault('desc', 'Practice data was already set.')
            return

        filename = (self.session.config.get('filename') or '').strip()

        # --------- CSV path resolution ----------
        rows = []
        if filename.endswith('.csv'):
            base_dir = Path(__file__).resolve().parent  # .../start
            candidates = [
                Path(filename),                          # absolute or relative to project root
                base_dir / 'data' / filename,            # start/data/<file>
                base_dir / filename,                     # start/<file>
            ]
            csv_path = next((p for p in candidates if p.exists()), None)
            if not csv_path:
                logger.warning(f'CSV file not found: {filename}. Tried: {", ".join(str(p) for p in candidates)}')
            else:
                try:
                    with csv_path.open(newline='', encoding='utf-8-sig') as f:
                        reader = csv.DictReader(f)
                        rows = [dict(row) for row in reader]
                    logger.info(f'Loaded {len(rows)} rows from CSV: {csv_path}')
                except Exception as e:
                    logger.error(f'Failed reading CSV {csv_path}: {e}')

        # --------- Interpreter header/choices ----------
        # Preferred: from config. Otherwise try to infer from CSV.
        sv['interpreter_title'] = (
            self.session.config.get('interpreter_title')
            or sv.get('interpreter_title')
            or 'Interpretation'
        )

        if 'interpreter_choices' in self.session.config:
            sv['interpreter_choices'] = list(self.session.config['interpreter_choices'])
        else:
            inferred_choices = []
            if rows:
                # Option 1: a column named "interpreter_choices" like "A;B;C"
                raw = (rows[0].get('interpreter_choices') or rows[0].get('choices') or '').strip()
                if raw:
                    inferred_choices = [c.strip() for c in raw.split(';') if c.strip()]

                # Option 2: columns like choice_1, choice_2, ...
                if not inferred_choices:
                    prefix_names = [k for k in rows[0].keys() if k.lower().startswith('choice_')]
                    if prefix_names:
                        # keep stable order by sorting numerically if possible
                        def _choice_key(k):
                            try:
                                return int(k.split('_', 1)[1])
                            except Exception:
                                return 1_000_000
                        for k in sorted(prefix_names, key=_choice_key):
                            val = (rows[0].get(k) or '').strip()
                            if val:
                                inferred_choices.append(val)

            sv['interpreter_choices'] = inferred_choices or ['Choice 1', 'Choice 2', 'Choice 3']

        # --------- Build practice_settings from CSV (robust to schema) ----------
        practice_settings = {}
        if rows:
            # If a "practice_id" column exists, use it. Otherwise, enumerate.
            def _practice_key(r, idx):
                pid = (r.get('practice_id') or r.get('practice') or '').strip()
                if pid.isdigit():
                    return int(pid)
                return idx  # fallback to row order

            # Required columns we look for, with aliases
            def _get(r, *names):
                for n in names:
                    if n in r and r[n] is not None:
                        val = str(r[n]).strip()
                        if val != '':
                            return val
                return ''

            indexed = []
            for i, r in enumerate(rows, start=1):
                # Optional filtering if your CSV mixes practice + other sections:
                section = (r.get('section') or r.get('type') or '').strip().lower()
                if section and section not in ('practice', 'practise', 'practice1', 'intro'):  # accept common labels
                    # Skip non-practice rows; remove this if the CSV contains *only* practice
                    continue

                idx = _practice_key(r, i)
                settings = {
                    'image': _get(r, 'filename', 'image', 'img'),
                    'title': _get(r, 'title', 'screen_title') or f'Practice {idx}',
                    'main_text': _get(r, 'main_text', 'text', 'html', 'content'),
                }

                # right_answer may be "1;0;1" or list-like columns ra_1, ra_2...
                ra = _get(r, 'right_answer', 'answers', 'ra')
                if ra:
                    settings['right_answer'] = [s.strip() for s in ra.split(';') if s.strip() != '']
                else:
                    # Try ra_1, ra_2, ...
                    ra_keys = [k for k in r.keys() if k.lower().startswith('ra_')]
                    vals = []
                    for k in sorted(ra_keys):
                        val = (r.get(k) or '').strip()
                        if val != '':
                            vals.append(val)
                    settings['right_answer'] = vals

                practice_settings[f'practice_{idx}'] = settings

            # Compact to practice_1..N sequence order
            if practice_settings:
                keys_sorted = sorted(
                    practice_settings.keys(),
                    key=lambda k: int(k.split('_')[1]) if k.split('_')[1].isdigit() else 10_000
                )
                practice_settings = {f'practice_{i+1}': practice_settings[k] for i, k in enumerate(keys_sorted)}

        sv['practice_settings'] = practice_settings

        if practice_settings:
            sv['desc'] = f'Practice data loaded from CSV: {filename or "unknown"}.'
        else:
            sv['desc'] = 'No practice rows found in CSV.'
            logger.warning('No practice content found: session.vars["practice_settings"] is empty.')

        # (Optional) helpful note for your debug view
        sv.setdefault('sheet_meta', {})
        sv['sheet_meta']['source'] = 'csv'


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    # Keep demographics as a single JSON blob (hidden input in template)
    survey_data = models.LongStringField(blank=True)


# -------------- Common base page (adds progress + instructions flag) -------------

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
        """Expose settings to the Django template."""
        return dict(settings=cls._get_settings(player))

    @classmethod
    def js_vars(cls, player: Player):
        """Expose the same settings for your Vue component."""
        return dict(settings=cls._get_settings(player))

    @classmethod
    def is_displayed(cls, player: Player):
        """Optional per-page gating via session.vars['user_settings']['practice_pages']."""
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
