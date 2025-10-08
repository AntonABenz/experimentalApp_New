from otree.api import *
from typing import Dict, Any, List

doc = """Main study app (oTree 5). Loads Google Sheet; uses img_desc as fallback."""

class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


class Subsession(BaseSubsession):
    def creating_session(self):
        # Only load once per session
        if 'sheet_data' in self.session.vars:
            return

        from reading_xls.reader import load_all
        cfg = self.session.config
        payload = load_all(cfg.get('filename', 'benz'))

        # Flatten main data to a LIST so pages can safely index rows[0]
        data = payload.get('data') or []
        if isinstance(data, dict):
            tabs = payload.get('tabs') or list(data.keys())
            flat: List[Dict[str, Any]] = []
            for t in tabs:
                flat.extend(data.get(t) or [])
            data = flat

        # Store everything in session.vars
        self.session.vars['sheet_settings'] = payload.get('settings', {})
        self.session.vars['sheet_data'] = data                         # always a LIST
        self.session.vars['practices'] = payload.get('practices', {})  # {tab -> [rows]}
        self.session.vars['sheet_meta'] = dict(
            schema=payload.get('schema', {}),
            tabs=payload.get('tabs', []),
        )


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    prolific_pid = models.StringField(blank=True)
    study_id = models.StringField(blank=True)
    prolific_session_id = models.StringField(blank=True)

    practice_response_text = models.LongStringField(blank=True)
    main_response_text = models.LongStringField(blank=True)


def _capture_prolific(player: Player, params: Dict[str, Any]):
    # Map Prolific params into model fields
    for k, field in (
        ('PROLIFIC_PID', 'prolific_pid'),
        ('STUDY_ID', 'study_id'),
        ('SESSION_ID', 'prolific_session_id'),
    ):
        v = params.get(k) or params.get(k.lower())
        if v:
            setattr(player, field, v)


class Instructions(Page):
    @staticmethod
    def vars_for_template(player: Player):
        return dict(instructions_path=player.session.config.get('instructions_path', ''))


class Consent(Page):
    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        params = dict(
            PROLIFIC_PID=player.participant.vars.get('PROLIFIC_PID'),
            STUDY_ID=player.participant.vars.get('STUDY_ID'),
            SESSION_ID=player.participant.vars.get('SESSION_ID'),
        )
        _capture_prolific(player, params)


class PracticeTask(Page):
    form_model = 'player'
    form_fields = ['practice_response_text']

    @staticmethod
    def is_displayed(player: Player):
        practices = player.session.vars.get('practices') or {}
        # show if any tab has rows
        return any(practices.get(k) for k in practices)

    @staticmethod
    def vars_for_template(player: Player):
        from utils.sheet_utils import image_src

        meta = player.session.vars.get('sheet_meta', {}) or {}
        schema = meta.get('schema', {}) or {}
        practices = player.session.vars.get('practices') or {}

        if not practices:
            return dict(row={}, img='', desc='', practice_tab=None, empty=True)

        # deterministic tab order: use meta['tabs'] if available, else sorted keys
        tabs = meta.get('tabs') or sorted(practices.keys())
        first_tab = next((t for t in tabs if practices.get(t)), None)
        rows = practices.get(first_tab) or []

        row = rows[0] if rows else {}
        # schema column names (fallbacks if schema keys missing)
        desc_key = schema.get('description', 'description')
        # Build outputs
        img = image_src(row, schema) if row else ''
        desc = (row.get(desc_key) or '').strip() if row else ''

        return dict(row=row, img=img, desc=desc, practice_tab=first_tab, empty=not bool(row))


class MainTask(Page):
    form_model = 'player'
    form_fields = ['main_response_text']

    @staticmethod
    def is_displayed(player: Player):
        rows = player.session.vars.get('sheet_data') or []
        return isinstance(rows, list) and len(rows) > 0

    @staticmethod
    def vars_for_template(player: Player):
        from utils.sheet_utils import image_src
        from utils.img_desc import read_desc

        meta = player.session.vars.get('sheet_meta', {}) or {}
        schema = meta.get('schema', {}) or {}
        rows: List[Dict[str, Any]] = player.session.vars.get('sheet_data') or []

        row = rows[0] if rows else {}
        if not row:
            return dict(row={}, img='', desc='No rows found in sheet_data.', empty=True)

        # schema keys with safe fallbacks
        desc_key = schema.get('description', 'description')
        fname_key = schema.get('filename', 'filename')

        img = image_src(row, schema)
        desc = (row.get(desc_key) or '').strip()

        # Fallback to file-based description if sheet text is empty
        if not desc:
            basename = (row.get(fname_key) or '').rsplit('.', 1)[0]
            desc = read_desc(basename)

        return dict(row=row, img=img, desc=desc, empty=False)


class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        from utils.prolific import completion_url
        return dict(
            show_prolific=player.session.config.get('prolific_enabled'),
            completion_link=completion_url(),
        )


page_sequence = [Instructions, Consent, PracticeTask, MainTask, Results]
