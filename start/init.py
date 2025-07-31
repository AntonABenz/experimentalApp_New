from otree.api import *
from . import *

class BasePracticePage(Page):
    page_num = None  # Must be set in subclasses
    form_model = 'player'
    
    def get_form_fields(self):
        return [f'practice_{self.page_num}_response']
    
    def get_settings_key(self):
        return f'Practice_{self.page_num}'
    
    def get_image_url(self):
        s3_base = self.session.config.get('s3_base_url', '')
        return f"{s3_base}/practice/practice_{self.page_num}.jpg"
    
    def vars_for_template(self):
        practice_data = self.session.vars.get('practice_data', {})
        settings = practice_data.get(self.get_settings_key(), {})
        
        return {
            'practice_settings': settings,
            'image_url': self.get_image_url(),
            'page_title': settings.get('title', f'Practice Page {self.page_num}')
        }

# Dynamically generate PracticePage classes for pages 1-7
page_classes = []
for i in range(1, 8):
    class_name = f'PracticePage{i}'
    cls = type(
        class_name,
        (BasePracticePage,),
        {'page_num': i}
    )
    globals()[class_name] = cls
    page_classes.append(cls)

page_sequence = page_classes
