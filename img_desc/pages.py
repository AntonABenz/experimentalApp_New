from otree.api import *
from . import *

class Instructions(Page):
    def vars_for_template(self):
        return {}

class Producer(Page):
    form_model = 'player'
    form_fields = []  # Add your form fields

class Interpreter(Page):
    form_model = 'player'
    form_fields = []  # Add your form fields

page_sequence = [Instructions, Producer, Interpreter]