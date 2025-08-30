import os
from os import environ
import dj_database_url
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent


# oTree settings
SESSION_CONFIGS = [
    dict(
        name='full_study',
        display_name="Full Study",
        app_sequence=['start', 'img_desc', 'demographics'],
        num_demo_participants=4,
        # Custom parameters
        filename=environ.get('GOOGLE_SHEET_NAME', 'benz'),
        instructions_path=environ.get('INSTRUCTIONS_URL', ''),
        s3_base_url=environ.get('S3_BASE_URL', ''),
        expand_slots=True,
    ),
    dict(
        name='practice_only',
        display_name="Practice Pages Only",
        app_sequence=['start'],
        num_demo_participants=1,
        filename=environ.get('GOOGLE_SHEET_NAME', 'benz'),
    ),
    dict(
        name='main_study_only',
        display_name="Main Study Only", 
        app_sequence=['img_desc', 'demographics'],
        num_demo_participants=4,
        filename=environ.get('GOOGLE_SHEET_NAME', 'benz'),
    ),
]

# Participant fields for Prolific integration
PARTICIPANT_FIELDS = [
    'prolific_pid',
    'study_id', 
    'session_id',
    'batch_id',
    'condition',
]

SESSION_FIELDS = [
    'config_dict',
    'google_sheet_data',
    'practice_data',
]

# Database
if environ.get('DATABASE_URL'):
    DATABASES = {
        'default': dj_database_url.parse(environ.get('DATABASE_URL'))
    }
else:
    DATABASES = {
        'default': {
            'ENGINE': 'django.db.backends.sqlite3',
            'NAME': BASE_DIR / 'db.sqlite3',
        }
    }

# Google Sheets configuration
GOOGLE_SHEETS_CONFIG = {
    'client_email': environ.get('GOOGLE_CLIENT_EMAIL'),
    'client_id': environ.get('GOOGLE_CLIENT_ID'),
    'private_key': environ.get('GOOGLE_PRIVATE_KEY', '').replace('\\n', '\n'),
    'private_key_id': environ.get('GOOGLE_PRIVATE_KEY_ID'),
}

# Security
SECRET_KEY = environ.get('SECRET_KEY', 'your-dev-secret-key-change-in-production')
DEBUG = environ.get('DEBUG', 'False').lower() == 'true'
ALLOWED_HOSTS = ['*'] if environ.get('OTREE_PRODUCTION') == '1' else ['localhost', '127.0.0.1']

# Middleware - IMPORTANT: Includes our custom Prolific middleware
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'otree.middleware.CheckBotMiddleware',
    'middleware.prolific_middleware.ProlificMiddleware',  # Custom Prolific handling
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

# Static files
STATIC_URL = '/static/'
STATIC_ROOT = BASE_DIR / 'staticfiles'
STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage'

# Templates
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / '_templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
                'otree.context_processors.otree',
            ],
        },
    },
]

# Internationalization
LANGUAGE_CODE = 'en'
TIME_ZONE = 'UTC'
USE_I18N = True
USE_TZ = True

# oTree specific
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = False
POINTS_CUSTOM_NAME = ''

# Prolific integration
PROLIFIC_COMPLETION_URL = 'https://app.prolific.co/submissions/complete'

# Admin
OTREE_ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD', 'admin')
OTREE_PRODUCTION = environ.get('OTREE_PRODUCTION', '') == '1'
OTREE_AUTH_LEVEL = environ.get('OTREE_AUTH_LEVEL', 'STUDY')

# Logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
}
