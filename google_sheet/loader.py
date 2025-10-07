# common/google_loader.py
import csv
import requests
from io import StringIO

def load_stimuli_from_gsheet(sheet_url):
    """
    Loads a Google Sheet that is published as CSV and returns a list of dictionaries.
    Each dict = one trial/stimulus.
    Example sheet URL:
      https://docs.google.com/spreadsheets/d/<sheet_id>/export?format=csv
    """
    try:
        response = requests.get(sheet_url)
        response.raise_for_status()
        csv_text = response.text
        reader = csv.DictReader(StringIO(csv_text))
        return [row for row in reader]
    except Exception as e:
        print("Error loading sheet:", e)
        return []
