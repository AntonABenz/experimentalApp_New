from otree.api import *

# Constants - OLD FORMAT
class Constants(BaseConstants):
    name_in_url = 'img_desc'
    players_per_group = 4
    num_rounds = 85

class Subsession(BaseSubsession):
    def creating_session(self):
    # Only load if not already loaded
        if 'sheet_data_loaded' not in self.session.vars:
            try:
                from utils.google_sheets import load_sheet_data
                filename = self.session.config.get('filename', 'testt')
                sheet_data = load_sheet_data(filename)
                
                if sheet_data:
                    self.session.vars['sheet_data'] = sheet_data
                    self.session.vars['settings'] = sheet_data.get('settings', {})
                    self.session.vars['sheet_data_loaded'] = True
            except Exception as e:
                logger.error(f"Failed to load Google Sheets: {e}")
                self.session.vars['sheet_data'] = {'settings': {}, 'data': []}
                self.session.vars['settings'] = {}
                self.session.vars['sheet_data_loaded'] = True

class Group(BaseGroup):
    # Group-level fields for your experiment
    condition = models.StringField(blank=True)
    item = models.StringField(blank=True)
    exp_id = models.StringField(blank=True)
    
    def get_player_by_player_role(self, role_name):
        """Get player by their role"""
        for player in self.get_players():
            if player.player_role == role_name:
                return player
        return None

class Player(BasePlayer):
    # Role assignment
    player_role = models.StringField(
        choices=['Producer', 'Interpreter'], 
        blank=True
    )
    
    # Response fields
    producer_response = models.LongStringField(
        blank=True,
        label="Describe what you see in the image"
    )
    interpreter_response = models.StringField(
        blank=True,
        label="Your interpretation"
    )
    
    # Timing fields
    time_spent_producer = models.FloatField(blank=True)
    time_spent_interpreter = models.FloatField(blank=True)
    
    # Prolific integration fields - RENAMED TO AVOID CONFLICTS
    prolific_pid = models.StringField(blank=True)
    prolific_study_id = models.StringField(blank=True)  # Changed from study_id
    prolific_session_id = models.StringField(blank=True)  # Changed from session_id
    
    def set_prolific_data(self):
        """Store Prolific parameters from session"""
        if hasattr(self.participant, 'vars'):
            self.prolific_pid = self.participant.vars.get('PROLIFIC_PID', '')
            self.prolific_study_id = self.participant.vars.get('STUDY_ID', '')
            self.prolific_session_id = self.participant.vars.get('SESSION_ID', '')
    
    def get_partner(self):
        """Get the other player in the group (for Producer-Interpreter pairs)"""
        if self.player_role == 'Producer':
            return self.group.get_player_by_player_role('Interpreter')
        elif self.player_role == 'Interpreter':
            return self.group.get_player_by_player_role('Producer')
        return None

# PAGES
class Instructions(Page):
    def vars_for_template(self):
        instructions_url = self.session.config.get('instructions_path', '')
        return {
            'instructions_url': instructions_url,
            'player_role': self.player.player_role
        }

class Producer(Page):
    form_model = 'player'
    form_fields = ['producer_response']
    
    def is_displayed(self):
        return self.player.player_role == 'Producer'
    
    def vars_for_template(self):
        # Load experiment data for this round
        sheet_data = self.session.vars.get('sheet_data', {})
        settings = self.session.vars.get('settings', {})
        
        # Get image URL based on round and condition
        s3_base = self.session.config.get('s3_base_url', '')
        
        # Example logic to get image based on round number
        # You'll need to adjust this based on your data structure
        round_num = self.round_number
        image_filename = f"image_{round_num}.jpg"  # Adjust as needed
        image_url = f"{s3_base}/{image_filename}" if s3_base else ''
        
        return {
            'sheet_data': sheet_data,
            'settings': settings,
            'image_url': image_url,
            'player_role': self.player.player_role,
            'round_number': round_num
        }

class WaitForProducer(WaitPage):
    def is_displayed(self):
        return self.player.player_role == 'Interpreter'
    
    body_text = "Waiting for the Producer to complete their description..."

class Interpreter(Page):
    form_model = 'player'
    form_fields = ['interpreter_response']
    
    def is_displayed(self):
        return self.player.player_role == 'Interpreter'
    
    def vars_for_template(self):
        # Get producer's response
        producer = self.group.get_player_by_player_role('Producer')
        producer_response = producer.producer_response if producer else ''
        
        return {
            'producer_response': producer_response,
            'player_role': self.player.player_role
        }

class WaitForAll(WaitPage):
    def after_all_players_arrive(self):
        # Store Prolific data for all players
        for player in self.group.get_players():
            player.set_prolific_data()

class Results(Page):
    def vars_for_template(self):
        # Show results to all players
        producer = self.group.get_player_by_player_role('Producer')
        interpreters = [p for p in self.group.get_players() if p.player_role == 'Interpreter']
        
        return {
            'producer_response': producer.producer_response if producer else '',
            'interpreter_responses': [p.interpreter_response for p in interpreters],
            'player_role': self.player.player_role
        }

# Page sequence
page_sequence = [
    Instructions,
    Producer,
    WaitForProducer,
    Interpreter,
    WaitForAll,
    Results
]
