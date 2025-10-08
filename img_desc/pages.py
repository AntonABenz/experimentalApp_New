from otree.api import *
from . import models

# --- Producer Page ---
class Producer(Page):
    form_model = 'player'
    form_fields = ['producer_response']
    template_name = 'img_desc/Producer.html'

    def vars_for_template(self):
        stimuli = self.session.vars.get('stimuli', [])
        # Use participant id_in_subsession to index into stimuli
        idx = self.player.id_in_subsession - 1
        current_stimulus = stimuli[idx] if idx < len(stimuli) else {}
        return {
            'stimulus': current_stimulus,
        }


# --- Interpreter Page ---
class Interpreter(Page):
    form_model = 'player'
    form_fields = ['interpreter_choice']
    template_name = 'img_desc/Interpreter.html'

    def vars_for_template(self):
        stimuli = self.session.vars.get('stimuli', [])
        idx = self.player.id_in_subsession - 1
        current_stimulus = stimuli[idx] if idx < len(stimuli) else {}
        return {
            'stimulus': current_stimulus,
        }


# --- Q Page ---
class Q(Page):
    form_model = 'player'
    form_fields = ['q_response']
    template_name = 'img_desc/Q.html'


# --- Feedback Page ---
class Feedback(Page):
    form_model = 'player'
    form_fields = ['feedback']
    template_name = 'img_desc/Feedback.html'


# --- Page sequence ---
page_sequence = [
    Producer,
    Interpreter,
    Q,
    Feedback,
]
