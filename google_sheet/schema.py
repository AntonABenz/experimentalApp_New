import os
from typing import Dict

def get_schema_from_env() -> Dict[str, str]:
    return {
        'image_url':   os.environ.get('BENZ_COL_IMAGE_URL',   's3path'),
        'filename':    os.environ.get('BENZ_COL_FILENAME',    'filename'),
        'extension':   os.environ.get('BENZ_COL_EXTENSION',   'extension'),
        'prompt':      os.environ.get('BENZ_COL_PROMPT',      'prompt'),
        'description': os.environ.get('BENZ_COL_DESCRIPTION', 'description'),
        'practice':    os.environ.get('BENZ_COL_PRACTICE',    'is_practice'),
    }

def get_tabs_from_env():
    return dict(
        settings=os.environ.get('BENZ_SHEET_TAB_SETTINGS', 'Settings'),
        data=os.environ.get('BENZ_SHEET_TAB_DATA', 'Data'),
        practice_prefix=os.environ.get('BENZ_SHEET_TAB_PREFIX_PRACTICE', 'Practice'),
    )
