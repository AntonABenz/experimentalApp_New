import csv
import io
import requests

def load_practice_data(filename):
    """
    Loads practice sheet data from Google Sheets (CSV export URL).
    Expected: sheet named 'Practice' in the workbook.
    """
    try:
        csv_url = f"https://docs.google.com/spreadsheets/d/{filename}/gviz/tq?tqx=out:csv&sheet=Practice"
        response = requests.get(csv_url)
        response.raise_for_status()
        decoded_content = response.content.decode('utf-8')
        reader = csv.DictReader(io.StringIO(decoded_content))
        practice_data = {f"Practice_{i+1}": row for i, row in enumerate(reader)}
        return practice_data
    except Exception as e:
        print(f"Error loading practice data: {e}")
        return {}


def load_sheet_data(filename):
    """
    Loads main experiment data (e.g., stimuli, conditions) from Google Sheets.
    Expected: sheet named 'Main' or 'Sheet1'.
    """
    try:
        csv_url = f"https://docs.google.com/spreadsheets/d/{filename}/gviz/tq?tqx=out:csv&sheet=Main"
        response = requests.get(csv_url)
        response.raise_for_status()
        decoded_content = response.content.decode('utf-8')
        reader = csv.DictReader(io.StringIO(decoded_content))
        data = [row for row in reader]

        # You can add special parsing logic here if you want to separate "settings"
        sheet_data = {
            'data': data,
            'settings': data[0] if data else {}
        }
        return sheet_data
    except Exception as e:
        print(f"Error loading sheet data: {e}")
        return {'data': [], 'settings': {}}
