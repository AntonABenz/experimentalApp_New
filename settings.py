from os import environ

def as_bool(v, default=False):
    if v is None:
        return default
    return v in ('1', 'true', 'True', True)

SESSION_CONFIGS = [
    dict(
        name='full_study',
        display_name='Full study',
        app_sequence=['start', 'img_desc'],
        num_demo_participants=4,
        filename='disjunctionExpTest8EN_1.xlsx',
    ),
    dict(
        name='practice_pages',
        display_name='Practice pages only',
        app_sequence=['start'],
        num_demo_participants=4,
        filename='disjunctionExpTest8EN_1.xlsx',
    ),
]

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00,
    participation_fee=0.00,
)

PARTICIPANT_FIELDS = ['prolific_pid', 'study_id', 'prolific_session_id']
SESSION_FIELDS = ['practice_settings', 'sheet_settings', 'interpreter_title', 'interpreter_choices']

LANGUAGE_CODE = 'en'
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

ADMIN_USERNAME = environ.get('OTREE_ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD', 'admin')
SECRET_KEY = environ.get('OTREE_SECRET_KEY', 'dev-secret')

OTREE_PRODUCTION = as_bool(environ.get('OTREE_PRODUCTION', '0'), False)

INSTALLED_APPS = ['otree']
