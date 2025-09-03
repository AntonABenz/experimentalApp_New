from otree.api import *
from . import *

class PracticePage1(Page):
    template_name = 'start/Practice1.html'
    form_model = 'player'
    form_fields = ['practice_1_response']

    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_1', {})

        page_sequence = self.session.config['page_sequence']
        current_page_name = self._current_page_name
        try:
            current_index = page_sequence.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence) * 100)

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

class PracticePage2(Page):
    template_name = 'start/Practice2.html'
    form_model = 'player'
    form_fields = ['practice_2_response']

    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_2', {})

        page_sequence = self.session.config['page_sequence']
        current_page_name = self._current_page_name
        try:
            current_index = page_sequence.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence) * 100)

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

class PracticePage3(Page):
    template_name = 'start/Practice3.html'
    form_model = 'player'
    form_fields = ['practice_3_response']

    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_3', {})

        page_sequence = self.session.config['page_sequence']
        current_page_name = self._current_page_name
        try:
            current_index = page_sequence.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence) * 100)

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

class PracticePage4(Page):
    template_name = 'start/Practice4.html'
    form_model = 'player'
    form_fields = ['practice_4_response']

    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_4', {})

        page_sequence = self.session.config['page_sequence']
        current_page_name = self._current_page_name
        try:
            current_index = page_sequence.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence) * 100)

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

class PracticePage5(Page):
    template_name = 'start/Practice5.html'
    form_model = 'player'
    form_fields = ['practice_5_response']

    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_5', {})

        page_sequence = self.session.config['page_sequence']
        current_page_name = self._current_page_name
        try:
            current_index = page_sequence.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence) * 100)

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

class PracticePage6(Page):
    template_name = 'start/Practice6.html'
    form_model = 'player'
    form_fields = ['practice_6_response']

    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_6', {})

        page_sequence = self.session.config['page_sequence']
        current_page_name = self._current_page_name
        try:
            current_index = page_sequence.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence) * 100)

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

class PracticePage7(Page):
    template_name = 'start/Practice7.html'
    form_model = 'player'
    form_fields = ['practice_7_response']

    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get('Practice_7', {})

        page_sequence = self.session.config['page_sequence']
        current_page_name = self._current_page_name
        try:
            current_index = page_sequence.index(current_page_name)
        except ValueError:
            current_index = 0
        progress = int((current_index + 1) / len(page_sequence) * 100)

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

page_sequence = [
    PracticePage1,
    PracticePage2,
    PracticePage3,
    PracticePage4,
    PracticePage5,
    PracticePage6,
    PracticePage7,
]
