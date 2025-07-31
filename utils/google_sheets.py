"""
Google Sheets integration utilities
Extracted from the original benzproj code
"""
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from django.conf import settings
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def get_google_sheet_client():
    """Get authenticated Google Sheets client"""
    scope = [
        'https://spreadsheets.google.com/feeds',
        'https://www.googleapis.com/auth/drive'
    ]
    
    config = settings.GOOGLE_SHEETS_CONFIG
    creds_dict = {
        "type": "service_account",
        "project_id": "your-project-id",  # Update this
        "private_key_id": config['private_key_id'],
        "private_key": config['private_key'],
        "client_email": config['client_email'],
        "client_id": config['client_id'],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{config['client_email']}"
    }
    
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        return gspread.authorize(creds)
    except Exception as e:
        logger.error(f"Failed to authenticate with Google Sheets: {e}")
        return None

def load_sheet_data(filename):
    """Load main experiment data from Google Sheets"""
    client = get_google_sheet_client()
    if not client:
        return None
        
    try:
        sheet = client.open(filename)
        
        # Load settings
        settings_sheet = sheet.worksheet('Settings')
        settings_data = {}
        for row in settings_sheet.get_all_records():
            settings_data[row.get('parameter', '')] = row.get('value', '')
        
        # Load main data
        data_sheet = sheet.worksheet('Data')
        main_data = data_sheet.get_all_records()
        
        return {
            'settings': settings_data,
            'data': main_data
        }
    except Exception as e:
        logger.error(f"Failed to load sheet data: {e}")
        return None

def load_practice_data(filename):
    """Load practice page settings from Google Sheets"""
    client = get_google_sheet_client()
    if not client:
        return {}
        
    try:
        sheet = client.open(filename)
        practice_data = {}
        
        for i in range(1, 8):  # Practice_1 through Practice_7
            try:
                worksheet = sheet.worksheet(f'Practice_{i}')
                records = worksheet.get_all_records()
                if records:
                    practice_data[f'Practice_{i}'] = records[0]
            except:
                practice_data[f'Practice_{i}'] = {}
                
        return practice_data
    except Exception as e:
        logger.error(f"Failed to load practice data: {e}")
        return {}
