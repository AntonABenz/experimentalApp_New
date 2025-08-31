from otree.api import *

class Constants(BaseConstants):
    name_in_url = 'demographics'
    players_per_group = None
    num_rounds = 1

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    age = models.IntegerField(
        min=18, 
        max=100, 
        label="What is your age?",
        blank=True
    )
    gender = models.StringField(
        choices=['Male', 'Female', 'Other', 'Prefer not to say'],
        label="What is your gender?",
        blank=True
    )
    education = models.StringField(
        choices=[
            'High school',
            'Some college',
            'Bachelor\'s degree',
            'Master\'s degree',
            'PhD',
            'Other'
        ],
        label="What is your highest level of education?",
        blank=True
    )
    feedback = models.LongStringField(
        blank=True,
        label="Any feedback about the study? (Optional)"
    )
    
    # Prolific integration fields - RENAMED TO AVOID CONFLICTS
    prolific_pid = models.StringField(blank=True)
    prolific_study_id = models.StringField(blank=True)      # NOT study_id
    prolific_session_id = models.StringField(blank=True)    # NOT session_id
    
    def set_prolific_data(self):
        """Store Prolific parameters from session"""
        if hasattr(self.participant, 'vars'):
            self.prolific_pid = self.participant.vars.get('PROLIFIC_PID', '')
            self.prolific_study_id = self.participant.vars.get('STUDY_ID', '')
            self.prolific_session_id = self.participant.vars.get('SESSION_ID', '')

# PAGES
class Demographics(Page):
    form_model = 'player'
    form_fields = ['age', 'gender', 'education', 'feedback']
    
    def before_next_page(self):
        # Store Prolific data before completing
        self.player.set_prolific_data()

class ProlificCompletion(Page):
    def vars_for_template(self):
        return {
            'prolific_completion_url': 'https://app.prolific.co/submissions/complete?cc=COMPLETED',
            'prolific_pid': self.player.prolific_pid
        }

# Page sequence
page_sequence = [Demographics, ProlificCompletion]
