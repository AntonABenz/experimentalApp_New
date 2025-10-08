from otree.api import *
from typing import Dict, Any, List

doc = """Main study app (oTree 5). Loads Google Sheet; uses img_desc as fallback."""

class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1


# ---------- helpers ----------
def _ensure_sheet_loaded(session):
    if session.vars.get('sheet_data') is not None:
        return
    # load your XLS/Google-Sheet bundle
    from reading_xls.reader import load_all
    payload = load_all(session.config.get('filename', 'benz'))
    session.vars['sheet_settings'] = payload['settings']
    session.vars['sheet_data']   = payload['data']        # list OR dict-of-tabs
    session.vars['practices']    = payload['practices']   # dict tab -> rows
    session.vars['sheet_meta']   = dict(schema=payload['schema'], tabs=payload['tabs'])


def _capture_prolific(player, params: Dict[str, Any]):
    for k, field in (('PROLIFIC_PID', 'prolific_pid'),
                     ('STUDY_ID', 'study_id'),
                     ('SESSION_ID', 'prolific_session_id')):
        v = params.get(k) or params.get(k.lower())
        if v:
            setattr(player, field, v)


# ---------- models ----------
class Subsession(BaseSubsession):
    def creating_session(self):
        _ensure_sheet_loaded(self.session)


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    prolific_pid = models.StringField(blank=True)
    study_id = models.StringField(blank=True)
    prolific_session_id = models.StringField(blank=True)

    # separate fields for practice vs main
    practice_response_text = models.LongStringField(blank=True)
    main_response_text = models.LongStringField(blank=True)


# ---------- pages ----------
class Instructions(Page):
    @staticmethod
    def vars_for_template(player: Player):
        # many templates check for 'instructions', so provide that key
        return dict(instructions=player.session.config.get('instructions_path', ''))


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
        _ensure_sheet_loaded(player.session)
        practices = player.session.vars.get('practices') or {}
        # show if *any* tab has at least one practice row
        return any(practices.get(tab) for tab in practices)

    @staticmethod
    def vars_for_template(player: Player):
        _ensure_sheet_loaded(player.session)
        from utils.sheet_utils import image_src

        meta = player.session.vars.get('sheet_meta') or {}
        schema = meta.get('schema') or {}
        practices = player.session.vars.get('practices') or {}

        if not practices or not any(practices.get(t) for t in practices):
            return dict(empty=True, img='', desc='', row={}, practice_tab=None,
                        instructions=player.session.config.get('instructions_path', ''))

        first_tab = sorted(practices.keys())[0]
        rows = practices.get(first_tab) or []
        row = rows[0] if rows else {}
        img = image_src(row, schema) if row else ''
        desc_key = schema.get('description')
        desc = (row.get(desc_key) or '').strip() if (row and desc_key) else ''

        return dict(empty=False, img=img, desc=desc, row=row, practice_tab=first_tab,
                    # keep templates happy:
                    progress={'current': 0, 'total': 0, 'percent': 0},
                    instructions=player.session.config.get('instructions_path', ''))


class MainTask(Page):
    form_model = 'player'
    form_fields = ['main_response_text']

    @staticmethod
    def vars_for_template(player: Player):
        _ensure_sheet_loaded(player.session)
        from utils.sheet_utils import image_src
        from utils.img_desc import read_desc

        meta = player.session.vars.get('sheet_meta') or {}
        schema = meta.get('schema') or {}
        rows = player.session.vars.get('sheet_data') or []

        # allow dict-of-tabs or flat list
        if isinstance(rows, dict):
            tabs = meta.get('tabs') or list(rows.keys())
            flat: List[Dict] = []
            for t in tabs:
                flat.extend(rows.get(t) or [])
            rows = flat

        row = rows[0] if isinstance(rows, list) and rows else {}
        if not row:
            return dict(empty=True, img='', desc='No rows found in sheet_data.',
                        row={}, progress={'current': 0, 'total': 0, 'percent': 0},
                        instructions=player.session.config.get('instructions_path', ''))

        img = image_src(row, schema) if row else ''
        desc_key = schema.get('description')
        desc = (row.get(desc_key) or '').strip() if desc_key else ''
        if not desc:
            fname_key = schema.get('filename')
            basename = (row.get(fname_key) or '').rsplit('.', 1)[0] if fname_key else ''
            if basename:
                desc = read_desc(basename)

        total = len(rows)
        current = 1  # showing the first item only in this app
        percent = round(100 * current / total) if total else 0

        return dict(empty=False, img=img, desc=desc, row=row,
                    progress={'current': current, 'total': total, 'percent': percent},
                    instructions=player.session.config.get('instructions_path', ''))


class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        from utils.prolific import completion_url
        return dict(
            show_prolific=player.session.config.get('prolific_enabled'),
            completion_link=completion_url(),
            instructions=player.session.config.get('instructions_path', ''),
        )


page_sequence = [Instructions, Consent, PracticeTask, MainTask, Results]
