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
        try:
            from reading_xls.reader import load_all
            cfg = self.session.config
            payload = load_all(cfg.get('filename', 'benz'))
        except Exception:
            payload = dict(settings={}, data=[], practices={}, schema={}, tabs=[])

        self.session.vars['sheet_settings'] = payload.get('settings', {})
        self.session.vars['sheet_data'] = payload.get('data', [])           # main rows (list or dict)
        self.session.vars['practices'] = payload.get('practices', {})       # {tab: [rows]}
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

    # keep separate fields for practice & main answers
    practice_response_text = models.LongStringField(blank=True)
    main_response_text = models.LongStringField(blank=True)


# ---------- helpers ----------

def _capture_prolific(player: Player, params: Dict[str, Any]):
    for k, field in (
        ('PROLIFIC_PID', 'prolific_pid'),
        ('STUDY_ID', 'study_id'),
        ('SESSION_ID', 'prolific_session_id'),
    ):
        v = params.get(k) or params.get(k.lower())
        if v:
            setattr(player, field, v)


def _progress(player: Player) -> int:
    """Best-effort progress %, falls back to 0 if internals not present."""
    try:
        idx = getattr(player.participant, '_index_in_pages', None)
        mx = getattr(player.participant, '_max_page_index', None)
        if idx and mx:
            return int(round(100 * idx / mx))
    except Exception:
        pass
    return 0


def _common_ctx(player: Player) -> Dict[str, Any]:
    return dict(
        instructions=player.session.config.get('instructions_path', ''),
        progress=_progress(player),
    )


# ---------- pages ----------

class Instructions(Page):
    @staticmethod
    def vars_for_template(player: Player):
        ctx = _common_ctx(player)
        ctx['instructions_path'] = player.session.config.get('instructions_path', '')
        return ctx


class Consent(Page):
    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        params = dict(
            PROLIFIC_PID=player.participant.vars.get('PROLIFIC_PID'),
            STUDY_ID=player.participant.vars.get('STUDY_ID'),
            SESSION_ID=player.participant.vars.get('SESSION_ID'),
        )
        _capture_prolific(player, params)

    @staticmethod
    def vars_for_template(player: Player):
        return _common_ctx(player)


class PracticeTask(Page):
    form_model = 'player'
    form_fields = ['practice_response_text']

    @staticmethod
    def is_displayed(player: Player):
        return bool(player.session.vars.get('practices'))

    @staticmethod
    def vars_for_template(player: Player):
        from utils.sheet_utils import image_src

        ctx = _common_ctx(player)
        meta = player.session.vars.get('sheet_meta', {}) or {}
        schema = meta.get('schema', {}) or {}
        practices = player.session.vars.get('practices') or {}

        if not practices:
            ctx.update(dict(row={}, img='', desc='', practice_tab=None))
            return ctx

        first_tab = sorted(practices.keys())[0]
        rows = practices.get(first_tab) or []
        row = rows[0] if rows else {}
        img = image_src(row, schema) if row else ''
        desc = (row.get(schema.get('description')) or '').strip() if (row and schema) else ''
        ctx.update(dict(row=row, img=img, desc=desc, practice_tab=first_tab))
        return ctx


class MainTask(Page):
    form_model = 'player'
    form_fields = ['main_response_text']  # <- was 'response_text'

    @staticmethod
    def vars_for_template(player: Player):
        from utils.sheet_utils import image_src
        from utils.img_desc import read_desc

        ctx = _common_ctx(player)
        meta = player.session.vars.get('sheet_meta', {}) or {}
        schema = meta.get('schema', {}) or {}

        rows = player.session.vars.get('sheet_data', []) or []
        # Accept both list-of-rows or dict {tab: [rows]}
        if isinstance(rows, dict):
            tabs = meta.get('tabs') or list(rows.keys())
            flat: List[Dict[str, Any]] = []
            for t in tabs:
                flat.extend(rows.get(t) or [])
            rows = flat

        row = rows[0] if isinstance(rows, list) and rows else {}

        img = image_src(row, schema) if row else ''
        desc = (row.get(schema.get('description')) or '').strip() if (row and schema) else ''
        if not desc and row:
            basename = (row.get(schema.get('filename')) or '').rsplit('.', 1)[0]
            desc = read_desc(basename)

        if not row:
            ctx.update(dict(row={}, img='', desc='No rows found in sheet_data.', empty=True))
            return ctx

        ctx.update(dict(row=row, img=img, desc=desc))
        return ctx


class Results(Page):
    @staticmethod
    def vars_for_template(player: Player):
        from utils.prolific import completion_url
        ctx = _common_ctx(player)
        ctx.update(dict(
            show_prolific=player.session.config.get('prolific_enabled'),
            completion_link=completion_url(),
        ))
        return ctx


page_sequence = [Instructions, Consent, PracticeTask, MainTask, Results]
