"""
Google Sheets integration utilities
Updated to load full service account JSON from environment variable
"""
import os
import json
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
    
    creds_json = os.environ.get('GOOGLE_SHEETS_CREDS_JSON')
    if not creds_json:
        logger.error("GOOGLE_SHEETS_CREDS_JSON environment variable not set")
        return None
    
    try:
        creds_dict = json.loads(creds_json)
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        return gspread.authorize(creds)
    except Exception as e:
        logger.error(f"Failed to authenticate with Google Sheets: {repr(e)}", exc_info=True)
        return None

def load_sheet_data(filename):
    """Load main experiment data from Google Sheets"""
    client = get_google_sheet_client()
    if not client:
        return None
        
    try:
        sheet = client.open_by_key('1WAjT5DrDAangn55J23Ruot7QvdnMWjGSwbXcRfVxWp0')
        
        # Load settings
        settings_sheet = sheet.worksheet('settings')
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
        logger.error(f"GoogleSheetsUtils - Failed to load sheet data: {repr(e)}", exc_info=True)
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
            except Exception as e:
                logger.warning(f"Practice sheet Practice_{i} not found or load failed: {repr(e)}", exc_info=True)
                practice_data[f'Practice_{i}'] = {}
                
        return practice_data
    except Exception as e:
        logger.error(f"Failed to load practice data: {repr(e)}", exc_info=True)
        return {}
