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

        # ✳️ store in session.vars (not as ad-hoc attributes)
        self.session.vars['sheet_settings'] = payload['settings']
        self.session.vars['sheet_data'] = payload['data']              # main rows
        self.session.vars['practices'] = payload['practices']          # dict of tabs -> rows
        self.session.vars['sheet_meta'] = dict(schema=payload['schema'], tabs=payload['tabs'])

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    prolific_pid = models.StringField(blank=True)
    study_id = models.StringField(blank=True)
    prolific_session_id = models.StringField(blank=True)

    # if you want to keep both practice & main answers, split them:
    practice_response_text = models.LongStringField(blank=True)
    main_response_text = models.LongStringField(blank=True)

def _capture_prolific(player: Player, params: Dict[str, Any]):
    # map Prolific params into your fields
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
    form_fields = ['practice_response_text']  # was 'response_text'

    @staticmethod
    def is_displayed(player: Player):
        return bool(player.session.vars.get('practices'))

    @staticmethod
    def vars_for_template(player: Player):
        from utils.sheet_utils import image_src
        meta = player.session.vars.get('sheet_meta', {}) or {}
        schema = meta.get('schema', {})
        practices = player.session.vars.get('practices') or {}

        if not practices:
            return dict(row={}, img='', desc='', practice_tab=None)

        first_tab = sorted(practices.keys())[0]
        rows = practices[first_tab]
        row = rows[0] if rows else {}
        img = image_src(row, schema)
        desc = (row.get(schema.get('description')) or '').strip() if schema else ''
        return dict(row=row, img=img, desc=desc, practice_tab=first_tab)

class MainTask(Page):
    form_model = 'player'
    form_fields = ['response_text']

    @staticmethod
    def vars_for_template(player: Player):
        from utils.sheet_utils import image_src
        from utils.img_desc import read_desc

        meta = getattr(player.session, 'sheet_meta', {}) or {}
        schema = meta.get('schema', {}) or {}

        rows = getattr(player.session, 'sheet_data', []) or []
        # Accept both list-of-rows or dict {tab: [rows]}
        if isinstance(rows, dict):
            # Prefer the first tab in meta['tabs'] for deterministic order
            tabs = meta.get('tabs') or list(rows.keys())
            flat = []
            for t in tabs:
                flat.extend(rows.get(t) or [])
            rows = flat

        row = rows[0] if isinstance(rows, list) and rows else {}

        img = image_src(row, schema) if row else ''
        desc = (row.get(schema.get('description')) or '').strip() if row else ''
        if not desc and row:
            basename = (row.get(schema.get('filename')) or '').rsplit('.', 1)[0]
            desc = read_desc(basename)

        # Graceful empty-state so the page renders instead of 500
        if not row:
            return dict(row={}, img='', desc='No rows found in sheet_data.', empty=True)

        return dict(row=row, img=img, desc=desc)

class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        from utils.prolific import completion_url
        return dict(
            show_prolific=player.session.config.get('prolific_enabled'),
            completion_link=completion_url()
        )

page_sequence = [Instructions, Consent, PracticeTask, MainTask, Results]
