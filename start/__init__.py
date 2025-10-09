from otree.api import *
import otree.api as otree
import logging, json
from json import JSONDecodeError

logger = logging.getLogger("benzapp.start_pages")

class C(BaseConstants):
    NAME_IN_URL = 'start'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

class Subsession(BaseSubsession):
    def creating_session(self):
        # Ensure the keys exist even if your loader didn't set them
        self.session.vars.setdefault('user_settings', {})       # e.g., {'practice_pages': {'Practice1': True, ...}}
        self.session.vars.setdefault('practice_settings', {})   # e.g., {'practice_1': {...}, 'practice_2': {...}}

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    # Prolific fields
    prolific_pid = models.StringField(blank=True)
    study_id = models.StringField(blank=True)
    prolific_session_id = models.StringField(blank=True)

    # Demographics: store the JSON blob from the page
    survey_data = models.LongStringField(blank=True)

def _capture_prolific(player: Player):
    """Copy Prolific query params (saved by oTree) into your Player fields."""
    for k, field in (('PROLIFIC_PID', 'prolific_pid'),
                     ('STUDY_ID', 'study_id'),
                     ('SESSION_ID', 'prolific_session_id')):
        v = player.participant.vars.get(k) or player.participant.vars.get(k.lower())
        if v:
            setattr(player, field, v)

# ----- Base Page that injects context expected by your templates -----
class Page(otree.Page):
    instructions_path = "start/includes/instructions.html"  # if you include this partial
    instructions = False
    template_name = None  # subclasses can set one template for many pages

    def get_context_data(self, **context):
        r = super().get_context_data(**context)
        maxpages = getattr(self.participant, '_max_page_index', 1) or 1
        idx = getattr(self, '_index_in_pages', 1) or 1
        r["instructions_google_doc"] = self.session.config.get("instructions_path")
        r["maxpages"] = maxpages
        r["page_index"] = idx
        r["progress"] = f"{int(idx / maxpages * 100):d}"
        r["instructions"] = self.instructions
        return r

# ----- Pages -----
class Consent(Page):
    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        _capture_prolific(player)

class Demographics(Page):
    form_model = 'player'
    form_fields = ['survey_data']

class Instructions(Page):
    pass

# ----- Practices: one HTML template reused by all practice pages -----
def _practice_enabled(session, cls_name):
    pps = session.vars.get('user_settings', {}).get('practice_pages', {})
    return pps.get(cls_name, True) if pps else True

class _PracticePage(Page):
    instructions = True
    practice_id: int = None
    template_name = 'start/Practice.html'  # one template for all practices

    def is_displayed(self):
        # allow toggling specific practice pages on/off from session.vars
        return _practice_enabled(self.session, self.__class__.__name__)

    def js_vars(self):
        """Expose per-practice settings to JS (image path, answers, etc.)."""
        try:
            settings = (self.session.vars.get("practice_settings", {})
                        .get(f"practice_{self.practice_id}", {})).copy()
            # If you use absolute URLs for images, add/transform here.
            # Example: if your settings contain "image": "foo.jpg", you can
            # construct a full path or leave it as-is if your template can handle it.
            return dict(settings=settings)
        except Exception as e:
            logger.error(f"Cannot build js_vars for practice_{self.practice_id}: {e}")
            return {}

class Practice1(_PracticePage): practice_id = 1
class Practice2(_PracticePage): practice_id = 2
class Practice3(_PracticePage): practice_id = 3
class Practice4(_PracticePage):
    practice_id = 4
    def js_vars(self):
        d = super().js_vars().get('settings', {}).copy()
        ra = d.get('right_answer')
        if ra:
            # Convert ["a;b;c", "x;y"] -> [["a","b","c"], ["x","y"]]
            d["right_answer"] = [[item.strip() for item in s.split(';')] for s in ra]
        return dict(settings=d)
class Practice5(_PracticePage): practice_id = 5
class Practice6(_PracticePage): practice_id = 6
class Practice7(_PracticePage): practice_id = 7

class EndOfIntro(Page):
    pass

page_sequence = [
    Consent,
    Demographics,
    Instructions,
    Practice1, Practice2, Practice3, Practice4, Practice5, Practice6, Practice7,
    EndOfIntro,
]
