from otree.api import *
from . import models

# --- Introduction Pages ---

class Consent(Page):
    template_name = 'start/Consent.html'


class Instructions(Page):
    template_name = 'start/Instructions.html'


class Demographics(Page):
    form_model = 'player'
    form_fields = ['age', 'gender']
    template_name = 'start/Demographics.html'


# --- Practice Pages ---

class PracticeBase(Page):
    form_model = 'player'

    def _get_practice_vars(self, page_num):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get(f'Practice_{page_num}', {})
        progress = int(page_num / 7 * 100)
        s3_base = self.session.config.get('s3_base_url', '')
        return {
            'practice_settings': settings,
            'image_url': f"{s3_base}/practice/practice_{page_num}.jpg",
            'page_title': settings.get('title', f'Practice Page {page_num}'),
            'instructions': False,
            'progress': progress,
        }


class Practice1(PracticeBase):
    form_fields = ['practice_1_response']
    template_name = 'start/Practice1.html'

    def vars_for_template(self):
        return self._get_practice_vars(1)


class Practice2(PracticeBase):
    form_fields = ['practice_2_response']
    template_name = 'start/Practice2.html'

    def vars_for_template(self):
        return self._get_practice_vars(2)


class Practice3(PracticeBase):
    form_fields = ['practice_3_response']
    template_name = 'start/Practice3.html'

    def vars_for_template(self):
        return self._get_practice_vars(3)


class Practice4(PracticeBase):
    form_fields = ['practice_4_response']
    template_name = 'start/Practice4.html'

    def vars_for_template(self):
        return self._get_practice_vars(4)


class Practice5(PracticeBase):
    form_fields = ['practice_5_response']
    template_name = 'start/Practice5.html'

    def vars_for_template(self):
        return self._get_practice_vars(5)


class Practice6(PracticeBase):
    form_fields = ['practice_6_response']
    template_name = 'start/Practice6.html'

    def vars_for_template(self):
        return self._get_practice_vars(6)


class Practice7(PracticeBase):
    form_fields = ['practice_7_response']
    template_name = 'start/Practice7.html'

    def vars_for_template(self):
        return self._get_practice_vars(7)


# --- Page sequence ---
page_sequence = [
    Consent,
    Instructions,
    Demographics,
    Practice1,
    Practice2,
    Practice3,
    Practice4,
    Practice5,
    Practice6,
    Practice7,
]
