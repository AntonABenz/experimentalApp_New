from otree.api import *

class Constants(BaseConstants):
    name_in_url = 'start'
    players_per_group = None
    num_rounds = 7

class Subsession(BaseSubsession):
    def creating_session(self):
        try:
            from utils.google_sheets import load_practice_data
            filename = self.session.config.get('filename', 'testt')
            practice_data = load_practice_data(filename)
            self.session.vars['practice_data'] = practice_data
            
            # Also load main settings for interpreter choices
            from utils.google_sheets import load_sheet_data
            sheet_data = load_sheet_data(filename)
            if sheet_data and 'settings' in sheet_data:
                settings = sheet_data['settings']
                self.session.vars['interpreter_choices'] = settings.get('interpreter_choices', '').split(';')
                self.session.vars['interpreter_title'] = settings.get('interpreter_title', 'Choices:')
                self.session.vars['s3_path'] = settings.get('s3path', '')
            else:
                self.session.vars['interpreter_choices'] = ['option1', 'option2', 'option3']
                self.session.vars['interpreter_title'] = 'Make your choice:'
                self.session.vars['s3_path'] = ''
        except Exception as e:
            # Fallback if Google Sheets fails
            self.session.vars['practice_data'] = {}
            self.session.vars['interpreter_choices'] = ['bonbon', 'chocolat', 'gummy bears']
            self.session.vars['interpreter_title'] = 'Buy sweets:'
            self.session.vars['s3_path'] = ''

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    # Practice responses
    practice_1_response = models.StringField(blank=True)
    practice_2_response = models.StringField(blank=True)
    practice_3_response = models.StringField(blank=True)
    practice_4_response = models.StringField(blank=True)
    practice_5_response = models.StringField(blank=True)
    practice_6_response = models.StringField(blank=True)
    practice_7_response = models.StringField(blank=True)

# PRACTICE PAGES
class Practice1(Page):
    form_model = 'player'
    form_fields = ['practice_1_response']
    
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_1', {})
        s3_path = self.session.vars.get('s3_path', '')
        
        # Construct full image path
        image_filename = settings.get('image_path', 'practice1.png')
        if s3_path and not image_filename.startswith('http'):
            full_image_path = f"{s3_path}/practice/{image_filename}"
        else:
            full_image_path = image_filename
            
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': full_image_path,
                    'title': settings.get('title', 'Practice Page 1'),
                    'main_text': settings.get('main_text', 'This is practice page 1'),
                    'right_answer': settings.get('right_answer', '1;0;1').split(';'),
                }
            }
        }

class Practice2(Page):
    form_model = 'player'
    form_fields = ['practice_2_response']
    
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_2', {})
        s3_path = self.session.vars.get('s3_path', '')
        
        image_filename = settings.get('image_path', 'practice2.png')
        if s3_path and not image_filename.startswith('http'):
            full_image_path = f"{s3_path}/practice/{image_filename}"
        else:
            full_image_path = image_filename
            
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': full_image_path,
                    'title': settings.get('title', 'Practice Page 2'),
                    'main_text': settings.get('main_text', 'This is practice page 2'),
                    'right_answer': settings.get('right_answer', '0;1;0').split(';'),
                }
            }
        }

class Practice3(Page):
    form_model = 'player'
    form_fields = ['practice_3_response']
    
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_3', {})
        s3_path = self.session.vars.get('s3_path', '')
        
        image_filename = settings.get('image_path', 'practice3.png')
        if s3_path and not image_filename.startswith('http'):
            full_image_path = f"{s3_path}/practice/{image_filename}"
        else:
            full_image_path = image_filename
            
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': full_image_path,
                    'title': settings.get('title', 'Practice Page 3'),
                    'main_text': settings.get('main_text', 'This is practice page 3'),
                    'right_answer': settings.get('right_answer', '1;1;0').split(';'),
                }
            }
        }

class Practice4(Page):
    form_model = 'player'
    form_fields = ['practice_4_response']
    
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_4', {})
        s3_path = self.session.vars.get('s3_path', '')
        
        image_filename = settings.get('image_path', 'practice4.png')
        if s3_path and not image_filename.startswith('http'):
            full_image_path = f"{s3_path}/practice/{image_filename}"
        else:
            full_image_path = image_filename
            
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': full_image_path,
                    'title': settings.get('title', 'Practice Page 4'),
                    'main_text': settings.get('main_text', 'This is practice page 4'),
                    'right_answer': settings.get('right_answer', '0;0;1').split(';'),
                }
            }
        }

class Practice5(Page):
    form_model = 'player'
    form_fields = ['practice_5_response']
    
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_5', {})
        s3_path = self.session.vars.get('s3_path', '')
        
        image_filename = settings.get('image_path', 'practice5.png')
        if s3_path and not image_filename.startswith('http'):
            full_image_path = f"{s3_path}/practice/{image_filename}"
        else:
            full_image_path = image_filename
            
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': full_image_path,
                    'title': settings.get('title', 'Practice Page 5'),
                    'main_text': settings.get('main_text', 'This is practice page 5'),
                    'right_answer': settings.get('right_answer', '1;0;0').split(';'),
                }
            }
        }

class Practice6(Page):
    form_model = 'player'
    form_fields = ['practice_6_response']
    
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_6', {})
        s3_path = self.session.vars.get('s3_path', '')
        
        image_filename = settings.get('image_path', 'practice6.png')
        if s3_path and not image_filename.startswith('http'):
            full_image_path = f"{s3_path}/practice/{image_filename}"
        else:
            full_image_path = image_filename
            
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': full_image_path,
                    'title': settings.get('title', 'Practice Page 6'),
                    'main_text': settings.get('main_text', 'This is practice page 6'),
                    'right_answer': settings.get('right_answer', '0;1;1').split(';'),
                }
            }
        }

class Practice7(Page):
    form_model = 'player'
    form_fields = ['practice_7_response']
    
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_7', {})
        s3_path = self.session.vars.get('s3_path', '')
        
        image_filename = settings.get('image_path', 'practice7.png')
        if s3_path and not image_filename.startswith('http'):
            full_image_path = f"{s3_path}/practice/{image_filename}"
        else:
            full_image_path = image_filename
            
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': full_image_path,
                    'title': settings.get('title', 'Practice Page 7'),
                    'main_text': settings.get('main_text', 'This is practice page 7'),
                    'right_answer': settings.get('right_answer', '1;1;1').split(';'),
                }
            }
        }

# Page sequence with all 7 practice pages
page_sequence = [Practice1, Practice2, Practice3, Practice4, Practice5, Practice6, Practice7]
