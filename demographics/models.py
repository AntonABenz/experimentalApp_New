from otree.api import *

class C(BaseConstants):
    NAME_IN_URL = __name__.split('.')[-2]  # auto-derive app name from folder
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    pass
