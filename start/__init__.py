from otree.api import *
from typing import Dict, Any, List

doc = """Main study app (oTree 5). Loads Google Sheet; uses img_desc text as fallback."""

class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

class Subsession(BaseSubsession):
    def creating_session(self):
        # Only load once per session
        if 'sheet_data' in self.session.vars:
            return

        from reading_xls.reader import load_all  # bring this module from your old repo
        cfg = self.session.config
        payload = load_all(cfg.get('filename', 'benz'))

        # store everything in session.vars
        self.session.vars['sheet_settings'] = payload.get('settings', {})          # dict
        self.session.vars['sheet_data'] = payload.get('data', [])                  # list OR dict
        self.session.vars['practices'] = payload.get('practices', {})              # dict tab -> rows
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
        return bool(player.session.vars.get('practices'))

    @staticmethod
    def vars_for_template(player: Player):
        from utils.sheet_utils import image_src
        meta = player.session.vars.get('sheet_meta', {}) or {}
        schema = meta.get('schema', {}) or {}
        practices = player.session.vars.get('practices') or {}
        settings = player.session.vars.get('sheet_settings', {}) or {}

        if not practices:
            return dict(row={}, img='', desc='', practice_tab=None)

        first_tab = sorted(practices.keys())[0]
        rows = practices.get(first_tab) or []
        row = rows[0] if rows else {}
        img = image_src(row, schema, settings)
        desc = (row.get(schema.get('description')) or '').strip() if row else ''
        return dict(row=row, img=img, desc=desc, practice_tab=first_tab)

class MainTask(Page):
    form_model = 'player'
    form_fields = ['main_response_text']

    @staticmethod
    def vars_for_template(player: Player):
        from utils.sheet_utils import image_src
        from utils.img_desc import read_desc

        meta = player.session.vars.get('sheet_meta', {}) or {}
        schema = meta.get('schema', {}) or {}
        settings = player.session.vars.get('sheet_settings', {}) or {}

        rows = player.session.vars.get('sheet_data')
        # Accept either list-of-rows OR dict {tab: [rows]}
        if isinstance(rows, dict):
            tabs = meta.get('tabs') or list(rows.keys())
            flat: List[Dict[str, Any]] = []
            for t in tabs:
                flat.extend(rows.get(t) or [])
            rows = flat
        if not isinstance(rows, list):
            rows = []

        row = rows[0] if rows else {}
        if not row:
            return dict(row={}, img='', desc='No rows found in sheet_data.', empty=True, progress=dict(current=0, total=0, percent=0))

        img = image_src(row, schema, settings)

        # description (prefer sheet cell; fallback to local text)
        desc = (row.get(schema.get('description')) or '').strip()
        if not desc:
            # Try filename column or 'Item'
            fname_key = schema.get('filename') or 'filename'
            basename = (row.get(fname_key) or row.get('Item') or '').rsplit('.', 1)[0]
            desc = read_desc(basename) if basename else ''

        # simple progress (first item only for now)
        total = len(rows)
        progress = dict(current=1 if total else 0, total=total, percent=int(100 * (1 / total)) if total else 0)

        return dict(row=row, img=img, desc=desc, progress=progress)

class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        from utils.prolific import completion_url
        return dict(
            show_prolific=bool(player.session.config.get('prolific_enabled')),
            completion_link=completion_url(),
        )

page_sequence = [Instructions, Consent, PracticeTask, MainTask, Results]
