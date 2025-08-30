from otree.api import *

class Constants(BaseConstants):  # NOT C(BaseConstants)
    name_in_url = 'start'        # lowercase
    players_per_group = None
    num_rounds = 7              # or however many practice pages

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    # Your practice fields here
    practice_1_response = models.StringField(blank=True)
    # Add more fields...

# Add your practice pages here
class PracticePage1(Page):
    form_model = 'player'
    form_fields = ['practice_1_response']

# Add page_sequence
page_sequence = [PracticePage1]  # Add all your pages
