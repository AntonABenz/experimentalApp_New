from otree.api import *

class C(BaseConstants):
    NAME_IN_URL = 'img_desc'
    PLAYERS_PER_GROUP = 4  # Producer + 3 Interpreters (or your configuration)
    NUM_ROUNDS = 1  # Or however many rounds you have

class Subsession(BaseSubsession):
    def creating_session(self):
        from utils.google_sheets import load_sheet_data
        filename = self.session.config.get('filename', 'benz')
        sheet_data = load_sheet_data(filename)
        
        if sheet_data:
            self.session.vars['sheet_data'] = sheet_data
            self.session.vars['settings'] = sheet_data.get('settings', {})

class Group(BaseGroup):
    # Add your group-level fields
    condition = models.StringField()
    item = models.StringField()

class Player(BasePlayer):
    # Add your player fields
    role = models.StringField(choices=['Producer', 'Interpreter'])
    producer_response = models.LongStringField(blank=True)
    interpreter_response = models.StringField(blank=True)
    
    # Prolific integration
    prolific_pid = models.StringField(blank=True)
    
    def set_prolific_data(self):
        """Set Prolific data from session"""
        self.prolific_pid = self.participant.vars.get('PROLIFIC_PID', '')
