import pandas as pd
import logging
import re
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("benzapp.get_data")

def _is_gsheet_url(s: str) -> bool:
    return isinstance(s, str) and "docs.google.com/spreadsheets/d/" in s

def _gsheet_export_xlsx_url(url: str) -> str:
    """
    Converts:
      https://docs.google.com/spreadsheets/d/<ID>/edit?gid=...#gid=...
    into:
      https://docs.google.com/spreadsheets/d/<ID>/export?format=xlsx
    """
    m = re.search(r"/spreadsheets/d/([^/]+)", url)
    if not m:
        raise ValueError("Could not parse Google Sheet ID from URL")
    sheet_id = m.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"

def _download_gsheet_xlsx(url: str, cache_seconds: int = 60) -> Path:
    """
    Downloads the Google Sheet as XLSX into /tmp with a short TTL cache.
    Works on Heroku.
    """
    export_url = _gsheet_export_xlsx_url(url)
    tmp_path = Path("/tmp") / "study.xlsx"
    stamp_path = Path("/tmp") / "study.xlsx.stamp"

    # simple TTL cache
    try:
        if tmp_path.exists() and stamp_path.exists():
            age = time.time() - float(stamp_path.read_text().strip() or "0")
            if age < cache_seconds:
                logger.info(f"Using cached Google Sheet XLSX at {tmp_path} (age={age:.1f}s)")
                return tmp_path
    except Exception:
        pass

    logger.info(f"Downloading Google Sheet XLSX from export URL: {export_url}")
    urllib.request.urlretrieve(export_url, tmp_path)
    stamp_path.write_text(str(time.time()))
    logger.info(f"Downloaded Google Sheet to {tmp_path}")
    return tmp_path

def _load_excel(filename: str) -> Path:
    # 1) Google Sheet URL path
    if _is_gsheet_url(filename):
        return _download_gsheet_xlsx(filename, cache_seconds=60)

    # 2) local file path fallback
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(filename),
        root / "start" / "data" / filename,
        root / "data" / filename,
        Path("start/data") / filename,
    ]

    for p in candidates:
        if p.exists():
            logger.info(f"Using local excel file: {p}")
            return p

    raise FileNotFoundError(f"Excel file '{filename}' not found.")

def _allowed_value_converter(v: str):
    if not v: return []
    return [item.strip() for item in str(v).split(";") if item.strip()]

def get_data(filename: str):
    """
    Main function to load Excel data and settings.
    Returns dict with 'data' and 'settings' keys.
    """
    xlsx_path = _load_excel(filename)
    
    # Load Workbook
    try:
        xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    except Exception as e:
        logger.error(f"Critical error opening Excel: {e}")
        raise e
    
    # 1. READ SETTINGS
    settings_sheet = next((s for s in xls.sheet_names if "setting" in s.lower()), None)
    settings_dict = {}
    if settings_sheet:
        df = xls.parse(settings_sheet, header=None, dtype=str, keep_default_na=False)
        if len(df.columns) >= 2:
            for _, row in df.iterrows():
                k, v = str(row[0]).strip(), str(row[1]).strip()
                if k: settings_dict[k] = v
    
    # Process lists
    settings_dict["suffixes"] = [
        str(settings_dict.get(f"suffix_{i}", "")).strip() 
        for i in range(1, 11) if settings_dict.get(f"suffix_{i}")
    ]
    settings_dict["allowed_regex"] = [
        str(settings_dict.get(f"allowed_regex_{i}", "")).strip() 
        for i in range(1, 21) if settings_dict.get(f"allowed_regex_{i}")
    ]
    settings_dict["allowed_values"] = [
        _allowed_value_converter(settings_dict.get(f"allowed_values_{i}", "")) 
        for i in range(1, 21) if settings_dict.get(f"allowed_values_{i}")
    ]
    
    if "interpreter_choices" in settings_dict:
        settings_dict["interpreter_choices"] = _allowed_value_converter(settings_dict["interpreter_choices"])
    
    # 2. READ DATA
    data_sheet = next((s for s in xls.sheet_names if s.lower() in ["data", "items", "trials"]), None)
    if not data_sheet:
        raise ValueError("No 'data' sheet found in Excel file.")
    
    df = xls.parse(data_sheet, dtype=str, keep_default_na=False, na_filter=False)
    
    # Clean whitespace
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Clean Item filenames (spaces to underscores)
    if "Item" in df.columns:
        df["Item"] = df["Item"].apply(lambda x: x.replace(" ", "_") if x else "")
    
    return dict(
        data=df.to_dict(orient="records"),
        settings=settings_dict
    )
