
import os, json, gspread
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

def _creds():
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON") or os.environ.get("GOOGLE_CREDENTIALS")
    if not raw:
        raise RuntimeError("Set GOOGLE_SERVICE_ACCOUNT_JSON in Heroku config vars.")
    info = json.loads(raw)
    return Credentials.from_service_account_info(info, scopes=SCOPES)

def load_sheet_into_session(session, filename=None):
    """Load all tabs; stash raw data + build a 'practices' block."""
    filename = filename or session.config.get("filename")
    if not filename:
        return

    gc = gspread.authorize(_creds())
    sh = gc.open(filename)

    sheet_data = {}
    for ws in sh.worksheets():
        sheet_data[ws.title] = ws.get_all_records()

    session.vars["sheet_data"] = sheet_data
    session.vars["sheet_meta"] = {"schema": (sheet_data[next(iter(sheet_data))][0] if sheet_data and sheet_data[next(iter(sheet_data))] else {})}
    session.vars["desc"] = f"Loaded {len(sh.worksheets())} tabs from Google Sheets."

    # Build the structure your Practice pages expect:
    practices = {}
    for tab_name, rows in sheet_data.items():
        if tab_name.lower().startswith("practice") and rows:
            practices[tab_name] = rows  # keep all rows; your start app uses the first one
    session.vars["practices"] = practices

