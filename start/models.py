from otree.api import *

class Constants(BaseConstants):
    name_in_url = 'start'
    players_per_group = None
    num_rounds = 7

class Subsession(BaseSubsession):
    pass

class Group(BaseGroup):
    pass

class Player(BasePlayer):
    # Practice responses for each page
    practice_1_response = models.StringField(blank=True)
    practice_2_response = models.StringField(blank=True)
    practice_3_response = models.StringField(blank=True)
    practice_4_response = models.StringField(blank=True)
    practice_5_response = models.StringField(blank=True)
    practice_6_response = models.StringField(blank=True)
    practice_7_response = models.StringField(blank=True)

class Practice1(Page):
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_1', {})
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': settings.get('image_path', '/static/default.jpg'),
                    'title': settings.get('title', 'Practice Page 1'),
                    'main_text': settings.get('main_text', 'Practice instructions'),
                    'right_answer': settings.get('right_answer', ['1', '0', '1']),
                }
            }
        }

class Practice2(Page):
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_2', {})
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': settings.get('image_path', '/static/default.jpg'),
                    'title': settings.get('title', 'Practice Page 2'),
                    'main_text': settings.get('main_text', 'Practice instructions'),
                    'right_answer': settings.get('right_answer', ['1', '0', '1']),
                }
            }
        }

class Practice3(Page):
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_3', {})
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': settings.get('image_path', '/static/default.jpg'),
                    'title': settings.get('title', 'Practice Page 3'),
                    'main_text': settings.get('main_text', 'Practice instructions'),
                    'right_answer': settings.get('right_answer', ['1', '0', '1']),
                }
            }
        }

class Practice4(Page):
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_4', {})
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': settings.get('image_path', '/static/default.jpg'),
                    'title': settings.get('title', 'Practice Page 4'),
                    'main_text': settings.get('main_text', 'Practice instructions'),
                    'right_answer': settings.get('right_answer', ['1', '0', '1']),
                }
            }
        }

class Practice5(Page):
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_5', {})
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': settings.get('image_path', '/static/default.jpg'),
                    'title': settings.get('title', 'Practice Page 5'),
                    'main_text': settings.get('main_text', 'Practice instructions'),
                    'right_answer': settings.get('right_answer', ['1', '0', '1']),
                }
            }
        }

class Practice6(Page):
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_6', {})
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': settings.get('image_path', '/static/default.jpg'),
                    'title': settings.get('title', 'Practice Page 6'),
                    'main_text': settings.get('main_text', 'Practice instructions'),
                    'right_answer': settings.get('right_answer', ['1', '0', '1']),
                }
            }
        }

class Practice7(Page):
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_7', {})
        return {
            'js_vars': {
                'settings': {
                    'full_image_path': settings.get('image_path', '/static/default.jpg'),
                    'title': settings.get('title', 'Practice Page 7'),
                    'main_text': settings.get('main_text', 'Practice instructions'),
                    'right_answer': settings.get('right_answer', ['1', '0', '1']),
                }
            }
        }

# Set your page_sequence to match the class names
page_sequence = [
    Practice1,
    Practice2,
    Practice3,
    Practice4,
    Practice5,
    Practice6,
    Practice7,
]
