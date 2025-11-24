from os import environ

def as_bool(v, default=False):
    if v is None:
        return default
    return v in ('1', 'true', 'True', True)


# ------------------------------------------------------------
# IMPORTANT:
# We explicitly set BOTH:
#   practice_xlsx = 'disjunctionExpTest8EN_1.xlsx'
#   filename      = 'disjunctionExpTest8EN_1.xlsx'
# so that start/initialization can load the correct Excel file.
# ------------------------------------------------------------
EXCEL_FILE = environ.get('BENZ_SPREADSHEET_NAME', 'disjunctionExpTest8EN_1.xlsx')


SESSION_CONFIGS = [
    dict(
        name='full_study',
        display_name='Full study',
        app_sequence=['start', 'img_desc'],
        num_demo_participants=4,

        # 🔥 These must point to the real file in start/data/
        practice_xlsx=EXCEL_FILE,
        filename=EXCEL_FILE,

        instructions_path=environ.get('BENZ_INSTRUCTIONS_URL', ''),
        expand_slots=as_bool(environ.get('BENZ_EXPAND_SLOTS', '1'), True),
        prolific_enabled=as_bool(environ.get('BENZ_PROLIFIC_ENABLED', '1'), True),
    ),

    dict(
        name='practice_pages',
        display_name='Practice pages',
        app_sequence=['start'],
        num_demo_participants=2,

        # 🔥 Must NOT load “benz”
        practice_xlsx=EXCEL_FILE,
        filename=EXCEL_FILE,

        instructions_path=environ.get('BENZ_INSTRUCTIONS_URL', ''),
        expand_slots=False,
        prolific_enabled=False,
    ),
]


SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00,
    participation_fee=0.00,
    doc="",
)

PARTICIPANT_FIELDS = ['prolific_pid', 'study_id', 'prolific_session_id']
SESSION_FIELDS = ['sheet_settings', 'sheet_data', 'practices', 'sheet_meta']

LANGUAGE_CODE = 'en'
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

ADMIN_USERNAME = environ.get('OTREE_ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD', 'changeme')
SECRET_KEY = environ.get('OTREE_SECRET_KEY', 'dev-secret')

OTREE_PRODUCTION = as_bool(environ.get('OTREE_PRODUCTION', '0'), False)

# These match your XLSX Settings sheet
SHEET_TAB_SETTINGS = environ.get('BENZ_SHEET_TAB_SETTINGS', 'Settings')
SHEET_TAB_DATA = environ.get('BENZ_SHEET_TAB_DATA', 'Data')
SHEET_TAB_PREFIX_PRACTICE = environ.get('BENZ_SHEET_TAB_PREFIX_PRACTICE', 'Practice')

# IMPORTANT: use 's3path'
SHEET_COLS = {
    'image_url': environ.get('BENZ_COL_IMAGE_URL', 's3path'),
    'filename': environ.get('BENZ_COL_FILENAME', 'filename'),
    'extension': environ.get('BENZ_COL_EXTENSION', 'extension'),
    'prompt': environ.get('BENZ_COL_PROMPT', 'prompt'),
    'description': environ.get('BENZ_COL_DESCRIPTION', 'description'),
    'practice': environ.get('BENZ_COL_PRACTICE', 'is_practice'),
}

DEMO_PAGE_INTRO_HTML = ""
INSTALLED_APPS = ['otree']
