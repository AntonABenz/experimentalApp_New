from otree.api import *
from typing import Dict, Any, List

doc = """Main study app (oTree 5). Loads Google Sheet; uses img_desc as fallback."""

class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

class Subsession(BaseSubsession): pass
class Group(BaseGroup): pass

class Player(BasePlayer):
    prolific_pid = models.StringField(blank=True)
    study_id = models.StringField(blank=True)
    session_id = models.StringField(blank=True)
    response_text = models.LongStringField(blank=True)

def creating_session(subsession: Subsession):
    if 'sheet_data' in subsession.session.vars:
        return
    from reading_xls.reader import load_all
    cfg = subsession.session.config
    payload = load_all(cfg.get('filename', 'benz'))
    subsession.session.sheet_settings = payload['settings']
    subsession.session.sheet_data = payload['data']
    subsession.session.practices = payload['practices']
    subsession.session.sheet_meta = dict(schema=payload['schema'], tabs=payload['tabs'])

def _capture_prolific(player: Player, params: Dict[str, Any]):
    for k, field in (('PROLIFIC_PID', 'prolific_pid'),
                     ('STUDY_ID', 'study_id'),
                     ('SESSION_ID', 'session_id')):
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
    form_fields = ['response_text']

    @staticmethod
    def is_displayed(player: Player):
        return bool(player.session.practices)

    @staticmethod
    def vars_for_template(player: Player):
        from utils.sheet_utils import image_src
        meta = player.session.sheet_meta
        schema = meta['schema']
        if not player.session.practices:
            return dict(row={}, img='', desc='', practice_tab=None)
        first_tab = sorted(player.session.practices.keys())[0]
        rows = player.session.practices[first_tab]
        row = rows[0] if rows else {}
        img = image_src(row, schema)
        desc = (row.get(schema['description']) or '').strip()
        return dict(row=row, img=img, desc=desc, practice_tab=first_tab)

class MainTask(Page):
    form_model = 'player'
    form_fields = ['response_text']

    @staticmethod
    def vars_for_template(player: Player):
        from utils.sheet_utils import image_src
        from utils.img_desc import read_desc
        meta = player.session.sheet_meta
        schema = meta['schema']
        rows: List[Dict] = player.session.sheet_data or []
        row = rows[0] if rows else {}
        img = image_src(row, schema)
        # prefer Sheet description; fallback to img_desc/<filename>.txt|md
        desc = (row.get(schema['description']) or '').strip()
        if not desc:
            basename = (row.get(schema['filename']) or '').rsplit('.', 1)[0]
            desc = read_desc(basename)
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

