from otree.api import *

# --- Introduction Pages ---

class Consent(Page):
    template_name = 'start/Consent.html'

class Introduction(Page):
    template_name = 'start/Introduction.html'

class Instructions(Page):
    template_name = 'start/Instructions.html'
    

class Demographics(Page):
     def post(self):
        print(self.request.POST)
        raw_data = self.request.POST.get('survey_data')
        try:
            json_data= json.loads(raw_data)
            print(json_data)
            self.player.survey_data = json.dumps(json_data)
        except JSONDecodeError:
            logger.warning('No  demographic data')
        except Exception as e:
            logger.error(f"Error while saving demographic data: {e}")
        return super().post()


# --- Practice Pages ---

class Practice1(Page):
    template_name = 'start/Practice1.html'
    form_model = 'player'
    form_fields = ['practice_1_response']

    def vars_for_template(self):
        return self._get_practice_vars(1)

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


class Practice2(Practice1):
    template_name = 'start/Practice2.html'
    form_fields = ['practice_2_response']

    def vars_for_template(self):
        return self._get_practice_vars(2)


class Practice3(Practice1):
    template_name = 'start/Practice3.html'
    form_fields = ['practice_3_response']

    def vars_for_template(self):
        return self._get_practice_vars(3)


class Practice4(Practice1):
    template_name = 'start/Practice4.html'
    form_fields = ['practice_4_response']

    def vars_for_template(self):
        return self._get_practice_vars(4)


class Practice5(Practice1):
    template_name = 'start/Practice5.html'
    form_fields = ['practice_5_response']

    def vars_for_template(self):
        return self._get_practice_vars(5)


class Practice6(Practice1):
    template_name = 'start/Practice6.html'
    form_fields = ['practice_6_response']

    def vars_for_template(self):
        return self._get_practice_vars(6)


class Practice7(Practice1):
    template_name = 'start/Practice7.html'
    form_fields = ['practice_7_response']

    def vars_for_template(self):
        return self._get_practice_vars(7)


# --- Full sequence ---
page_sequence = [
    Consent,
    Introduction,
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
