from otree.api import *

class C(BaseConstants):
    NAME_IN_URL = 'demographics'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

class Subsession(BaseSubsession): pass
class Group(BaseGroup): pass

class Player(BasePlayer):
    age = models.IntegerField(min=18, max=100, label="Your age")
    gender = models.StringField(
        choices=['Female','Male','Other','Prefer not to say'],
        widget=widgets.RadioSelect,
        label="Gender"
    )

class Demographics(Page):
    form_model = 'player'
    form_fields = ['age','gender']

page_sequence = [Demographics]

