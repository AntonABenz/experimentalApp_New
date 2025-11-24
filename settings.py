from os import environ
from pathlib import Path

def as_bool(v, default=False):
    if v is None:
        return default
    return v in ('1', 'true', 'True', True)


# --------------------------------------------------------------------
# MAIN EXCEL FILE
# --------------------------------------------------------------------

# Default filename inside start/data/
DEFAULT_EXCEL_FILENAME = "disjunctionExpTest8EN_1.xlsx"

# Full path oTree should load
EXCEL_FILE = f"start/data/{DEFAULT_EXCEL_FILENAME}"

# Allow override from environment (Heroku)
EXCEL_FILE = environ.get("BENZ_SPREADSHEET_NAME", EXCEL_FILE)


# --------------------------------------------------------------------
# SESSION CONFIGS
# --------------------------------------------------------------------

SESSION_CONFIGS = [
    dict(
        name='full_study',
        display_name='Full study',
        app_sequence=['start', 'img_desc'],
        num_demo_participants=4,

        # IMPORTANT: Feed correct Excel file into start/init.py
        practice_xlsx=EXCEL_FILE,
        filename=EXCEL_FILE,

        instructions_path=environ.get('BENZ_INSTRUCTIONS_URL', ''),
        expand_slots=as_bool(environ.get('BENZ_EXPAND_SLOTS', '1'), True),
        prolific_enabled=as_bool(environ.get('BENZ_PROLIFIC_ENABLED', '1'), True),
    ),
    dict(
        name='practice_pages',
        display_name='Practice pages only',
        app_sequence=['start'],
        num_demo_participants=4,

        practice_xlsx=EXCEL_FILE,
        filename=EXCEL_FILE,

        expand_slots=False,
        prolific_enabled=False,
    ),
]


SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00,
    participation_fee=0.00,
    doc="",
)

PARTICIPANT_FIELDS = []
SESSION_FIELDS = ['practice_settings', 'sheet_settings', 'interpreter_title', 'interpreter_choices']

LANGUAGE_CODE = 'en'
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = False

ADMIN_USERNAME = environ.get("OTREE_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = environ.get("OTREE_ADMIN_PASSWORD", "changeme")

SECRET_KEY = environ.get("OTREE_SECRET_KEY", "dev-secret")

OTREE_PRODUCTION = as_bool(environ.get('OTREE_PRODUCTION', '0'), False)

INSTALLED_APPS = ['otree']
