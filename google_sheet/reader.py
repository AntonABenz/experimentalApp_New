import os, json, tempfile
from typing import Dict, List
import gspread
from google.oauth2.service_account import Credentials
from .schema import get_schema_from_env, get_tabs_from_env

SCOPES = ['https://www.googleapis.com/auth/spreadsheets.readonly']

def _creds():
    blob = os.environ.get('BENZ_GOOGLE_CREDENTIALS_JSON')
    if blob:
        data = json.loads(blob)
        tf = tempfile.NamedTemporaryFile(delete=False, suffix='.json')
        tf.write(json.dumps(data).encode('utf-8')); tf.flush()
        return Credentials.from_service_account_file(tf.name, scopes=SCOPES)
    path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if not path:
        raise RuntimeError('Provide BENZ_GOOGLE_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS')
    return Credentials.from_service_account_file(path, scopes=SCOPES)

def _ws_map_by_title(sh) -> Dict[str, List[Dict]]:
    return {ws.title: ws.get_all_records() for ws in sh.worksheets()}

def load_all(spreadsheet_name: str):
    client = gspread.authorize(_creds())
    sh = client.open(spreadsheet_name)
    ws = _ws_map_by_title(sh)

    tabs = get_tabs_from_env()
    schema = get_schema_from_env()

    settings = ws.get(tabs['settings'], [])
    data = ws.get(tabs['data'], [])
    practices = {k: v for k, v in ws.items() if k.lower().startswith(tabs['practice_prefix'].lower())}

    def normalize(rows):
        base = {v: '' for v in schema.values()}
        return [{**base, **r} for r in rows]

    return dict(
        settings=normalize(settings),
        data=normalize(data),
        practices={k: normalize(v) for k, v in practices.items()},
        schema=schema,
        tabs=tabs,
    )
