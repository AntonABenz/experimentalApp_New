from otree.api import *
from . import *

class Practice1(Page):
    template_name = 'start/Practice1.html'
    form_model = 'player'
    form_fields = ['practice_1_response']

    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_1', {})
        page_sequence_class_names = [
            'Practice1',
            'Practice2',
            'Practice3',
            'Practice4',
            'Practice5',
            'Practice6',
            'Practice7',
        ]
        current_page_name = self.__class__.__name__
        try:
            current_index = page_sequence_class_names.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence_class_names) * 100)
        return {
            'practice_settings': settings,
            'image_url': self.get_practice_image_url(1),
            'page_title': settings.get('title', 'Practice Page 1'),
            'instructions': False,
            'progress': progress,
        }

    def get_practice_image_url(self, page_num):
        s3_base = self.session.config.get('s3_base_url', '')
        return f"{s3_base}/practice/practice_{page_num}.jpg"

class Practice2(Page):
    template_name = 'start/Practice2.html'
    form_model = 'player'
    form_fields = ['practice_2_response']

    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_2', {})
        page_sequence_class_names = [
            'Practice1',
            'Practice2',
            'Practice3',
            'Practice4',
            'Practice5',
            'Practice6',
            'Practice7',
        ]
        current_page_name = self.__class__.__name__
        try:
            current_index = page_sequence_class_names.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence_class_names) * 100)
        return {
            'practice_settings': settings,
            'image_url': self.get_practice_image_url(2),
            'page_title': settings.get('title', 'Practice Page 2'),
            'instructions': False,
            'progress': progress,
        }

    def get_practice_image_url(self, page_num):
        s3_base = self.session.config.get('s3_base_url', '')
        return f"{s3_base}/practice/practice_{page_num}.jpg"

class Practice3(Page):
    template_name = 'start/Practice3.html'
    form_model = 'player'
    form_fields = ['practice_3_response']
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_3', {})
        page_sequence_class_names = [
            'Practice1',
            'Practice2',
            'Practice3',
            'Practice4',
            'Practice5',
            'Practice6',
            'Practice7',
        ]
        current_page_name = self.__class__.__name__
        try:
            current_index = page_sequence_class_names.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence_class_names) * 100)
        return {
            'practice_settings': settings,
            'image_url': self.get_practice_image_url(3),
            'page_title': settings.get('title', 'Practice Page 3'),
            'instructions': False,
            'progress': progress,
        }
    def get_practice_image_url(self, page_num):
        s3_base = self.session.config.get('s3_base_url', '')
        return f"{s3_base}/practice/practice_{page_num}.jpg"

class Practice4(Page):
    template_name = 'start/Practice4.html'
    form_model = 'player'
    form_fields = ['practice_4_response']
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_4', {})
        page_sequence_class_names = [
            'Practice1',
            'Practice2',
            'Practice3',
            'Practice4',
            'Practice5',
            'Practice6',
            'Practice7',
        ]
        current_page_name = self.__class__.__name__
        try:
            current_index = page_sequence_class_names.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence_class_names) * 100)
        return {
            'practice_settings': settings,
            'image_url': self.get_practice_image_url(4),
            'page_title': settings.get('title', 'Practice Page 4'),
            'instructions': False,
            'progress': progress,
        }
    def get_practice_image_url(self, page_num):
        s3_base = self.session.config.get('s3_base_url', '')
        return f"{s3_base}/practice/practice_{page_num}.jpg"

class Practice5(Page):
    template_name = 'start/Practice5.html'
    form_model = 'player'
    form_fields = ['practice_5_response']
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_5', {})
        page_sequence_class_names = [
            'Practice1',
            'Practice2',
            'Practice3',
            'Practice4',
            'Practice5',
            'Practice6',
            'Practice7',
        ]
        current_page_name = self.__class__.__name__
        try:
            current_index = page_sequence_class_names.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence_class_names) * 100)
        return {
            'practice_settings': settings,
            'image_url': self.get_practice_image_url(5),
            'page_title': settings.get('title', 'Practice Page 5'),
            'instructions': False,
            'progress': progress,
        }
    def get_practice_image_url(self, page_num):
        s3_base = self.session.config.get('s3_base_url', '')
        return f"{s3_base}/practice/practice_{page_num}.jpg"

class Practice6(Page):
    template_name = 'start/Practice6.html'
    form_model = 'player'
    form_fields = ['practice_6_response']
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_6', {})
        page_sequence_class_names = [
            'Practice1',
            'Practice2',
            'Practice3',
            'Practice4',
            'Practice5',
            'Practice6',
            'Practice7',
        ]
        current_page_name = self.__class__.__name__
        try:
            current_index = page_sequence_class_names.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence_class_names) * 100)
        return {
            'practice_settings': settings,
            'image_url': self.get_practice_image_url(6),
            'page_title': settings.get('title', 'Practice Page 6'),
            'instructions': False,
            'progress': progress,
        }
    def get_practice_image_url(self, page_num):
        s3_base = self.session.config.get('s3_base_url', '')
        return f"{s3_base}/practice/practice_{page_num}.jpg"

class Practice7(Page):
    template_name = 'start/Practice7.html'
    form_model = 'player'
    form_fields = ['practice_7_response']
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_7', {})
        page_sequence_class_names = [
            'Practice1',
            'Practice2',
            'Practice3',
            'Practice4',
            'Practice5',
            'Practice6',
            'Practice7',
        ]
        current_page_name = self.__class__.__name__
        try:
            current_index = page_sequence_class_names.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence_class_names) * 100)
        return {
            'practice_settings': settings,
            'image_url': self.get_practice_image_url(7),
            'page_title': settings.get('title', 'Practice Page 7'),
            'instructions': False,
            'progress': progress,
        }
    def get_practice_image_url(self, page_num):
        s3_base = self.session.config.get('s3_base_url', '')
        return f"{s3_base}/practice/practice_{page_num}.jpg"

# Updated page_sequence
page_sequence = [
    Practice1,
    Practice2,
    Practice3,
    Practice4,
    Practice5,
    Practice6,
    Practice7,
]
