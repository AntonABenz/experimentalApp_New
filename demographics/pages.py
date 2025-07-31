from otree.api import *
from . import *

class Demographics(Page):
    form_model = 'player'
    form_fields = ['age', 'gender', 'education']  # Add your demographic fields

page_sequence = [Demographics]