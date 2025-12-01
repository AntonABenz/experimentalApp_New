from os import environ

def as_bool(v, default=False):
    if v is None:
        return default
    return v in ('1', 'true', 'True', True)


SESSION_CONFIGS = [
    # Full pipeline: practice (start) + main experiment (img_desc)
    dict(
        name='full_study',
        display_name='Full study',
        app_sequence=['start', 'img_desc'],
        num_demo_participants=8,
        filename='disjunctionExpTest8EN_1.xlsx',

        # used by img_desc + start.Consent
        for_prolific=True,
        expand_slots=True,
    ),

    # Practice only (no Prolific / main experiment)
    dict(
        name='practice_pages',
        display_name='Practice pages only',
        app_sequence=['start'],
        num_demo_participants=4,
        filename='disjunctionExpTest8EN_1.xlsx',

        for_prolific=False,
        expand_slots=False,
    ),
]


SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00,
    participation_fee=0.00,
)


# -------------------------------------------------------------------
# VARS KEYS USED IN YOUR CODE
# -------------------------------------------------------------------
# img_desc + start use participant.vars for Prolific + a completion flag
PARTICIPANT_FIELDS = [
    'prolific_id',          
    'study_id',
    'prolific_session_id',  
    'full_study_completed',
]

# All keys you store in session.vars in start + img_desc
SESSION_FIELDS = [
    # from start/__init__.py
    'practice_settings',
    'sheet_settings',
    'interpreter_title',
    'interpreter_choices',
    'suffixes',
    'allowed_values',
    'EndOfIntroText',

    # from img_desc 
    'active_batch',
    'user_data',
    'num_rounds',
    'practice_pages',
    'user_settings',
    's3path',
    'extension',
    'prefix',
    'allowed_regex',
    'caseflag',
    'interpreter_input_type',
    'interpreter_input_choices',
    'max_users',
    'batch_size',
]


LANGUAGE_CODE = 'en'
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True  
ADMIN_USERNAME = environ.get('OTREE_ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD', 'admin')
SECRET_KEY = environ.get('OTREE_SECRET_KEY', 'dev-secret')

OTREE_PRODUCTION = as_bool(environ.get('OTREE_PRODUCTION', '0'), False)

INSTALLED_APPS = ['otree']
