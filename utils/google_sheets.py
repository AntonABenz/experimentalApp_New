import gspread
from oauth2client.service_account import ServiceAccountCredentials
from os import environ
import logging
import time

logger = logging.getLogger(__name__)

# Simple in-memory cache
_sheets_cache = {}
_cache_timeout = 300  # 5 minutes

def get_google_sheet_client():
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    
    client_email = environ.get('GOOGLE_CLIENT_EMAIL')
    client_id = environ.get('GOOGLE_CLIENT_ID')
    private_key = environ.get('GOOGLE_PRIVATE_KEY', '').replace('\\n', '\n')
    private_key_id = environ.get('GOOGLE_PRIVATE_KEY_ID')
    
    if not all([client_email, client_id, private_key, private_key_id]):
        return None
    
    creds_dict = {
        'type': 'service_account',
        'project_id': 'versatile-nomad-423508-e4',
        'private_key_id': environ.get('GOOGLE_PRIVATE_KEY_ID'),
        'private_key': environ.get('GOOGLE_PRIVATE_KEY').replace('\\\\n', '\\n'),
        'client_email': environ.get('GOOGLE_CLIENT_EMAIL'),
        'client_id': environ.get('GOOGLE_CLIENT_ID'),
        'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
        'token_uri': 'https://oauth2.googleapis.com/token',
        
    }
    
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        return gspread.authorize(creds)
    except Exception as e:
        logger.error(f"Failed to authenticate with Google Sheets: {e}")
        return None

def load_sheet_data(filename):
    """Load with caching to reduce API calls"""
    cache_key = f"sheet_data_{filename}"
    current_time = time.time()
    
    # Check cache first
    if cache_key in _sheets_cache:
        cached_data, timestamp = _sheets_cache[cache_key]
        if current_time - timestamp < _cache_timeout:
            logger.info(f"Using cached data for {filename}")
            return cached_data
    
    # Load fresh data
    client = get_google_sheet_client()
    if not client:
        return {'settings': {}, 'data': []}
        
    try:
        sheet = client.open(filename)
        
        # Load settings with retry
        settings_data = {}
        try:
            settings_sheet = sheet.worksheet('settings')
            for row in settings_sheet.get_all_records():
                parameter = row.get('parameter', '')
                value = row.get('value', '')
                if parameter:
                    settings_data[parameter] = value
        except Exception as e:
            logger.warning(f"Could not load Settings sheet: {e}")
        
        # Load data with retry
        main_data = []
        try:
            data_sheet = sheet.worksheet('Data')
            main_data = data_sheet.get_all_records()
        except Exception as e:
            logger.warning(f"Could not load Data sheet: {e}")
        
        result = {'settings': settings_data, 'data': main_data}
        
        # Cache the result
        _sheets_cache[cache_key] = (result, current_time)
        
        return result
    except Exception as e:
        logger.error(f"Failed to load sheet data from {filename}: {e}")
        return {'settings': {}, 'data': []}

def load_practice_data(filename):
    """Load practice data with caching"""
    cache_key = f"practice_data_{filename}"
    current_time = time.time()
    
    # Check cache first
    if cache_key in _sheets_cache:
        cached_data, timestamp = _sheets_cache[cache_key]
        if current_time - timestamp < _cache_timeout:
            return cached_data
    
    client = get_google_sheet_client()
    if not client:
        return {}
        
    try:
        sheet = client.open(filename)
        practice_data = {}
        
        for i in range(1, 8):
            try:
                worksheet = sheet.worksheet(f'Practice_{i}')
                records = worksheet.get_all_records()
                if records:
                    # Convert list of records to single dict
                    practice_dict = {}
                    for record in records:
                        practice_dict[record.get('parameter', '')] = record.get('value', '')
                    practice_data[f'Practice_{i}'] = practice_dict
                else:
                    practice_data[f'Practice_{i}'] = {}
            except Exception as e:
                logger.warning(f"Could not load Practice_{i}: {e}")
                practice_data[f'Practice_{i}'] = {}
        
        # Cache the result
        _sheets_cache[cache_key] = (practice_data, current_time)
        
        return practice_data
    except Exception as e:
        logger.error(f"Failed to load practice data from {filename}: {e}")
        return {}
