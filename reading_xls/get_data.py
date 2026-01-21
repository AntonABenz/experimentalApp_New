import re
import time
import logging
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
