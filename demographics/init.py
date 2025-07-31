from otree.api import *

class C(BaseConstants):
    NAME_IN_URL = 'demographics'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    age = models.IntegerField(min=18, max=100)
    gender = models.StringField(
        choices=['Male', 'Female', 'Other', 'Prefer not to say']
    )
    education = models.StringField(
        choices=[
            'High school',
            'Some college',
            'Bachelor\'s degree',
            'Master\'s degree',
            'PhD',
            'Other'
        ]
    )
    feedback = models.LongStringField(blank=True)
