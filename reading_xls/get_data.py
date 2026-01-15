import pandas as pd
import re
import logging
import hashlib
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("benzapp.get_data")

SETTINGS_WS = "settings"
DATA_WS = "data"
PRACTICE_WS_PREFIX = "practice_"

def _fingerprint_file(p: Path) -> dict:
    """Generates a unique hash of the file content."""
    stat = p.stat()
    sha = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha.update(chunk)
    
    # Return a summary dict
    return {
        "path": str(p.resolve()),
        "size_bytes": stat.st_size,
        "sha256_hash": sha.hexdigest()[:10] + "...", # Show first 10 chars
        "full_hash": sha.hexdigest()
    }

def _load_excel(filename: str) -> Path:
    # Look in multiple locations
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(filename),
        root / "start" / "data" / filename,
        root / "data" / filename,
        Path("start/data") / filename,
    ]

    logger.info(f"🔍 Searching for Excel file: {filename}")
    
    for p in candidates:
        if p.exists():
            fp = _fingerprint_file(p)
            logger.info(f"✅ FOUND EXCEL FILE!")
            logger.info(f"   📂 Path: {fp['path']}")
            logger.info(f"   ⚖️ Size: {fp['size_bytes']} bytes")
            logger.info(f"   🔑 Fingerprint (Hash): {fp['sha256_hash']}")
            return p

    logger.error(f"❌ Excel file not found. Checked: {[str(c) for c in candidates]}")
    raise FileNotFoundError(f"Excel file '{filename}' not found.")

def _allowed_value_converter(v: str):
    if not v: return []
    return [item.strip() for item in str(v).split(";") if item.strip()]

def get_data(filename: str):
    """
    Robustly loads Excel data and settings.
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
        settings_df = xls.parse(settings_sheet, header=None, dtype=str, keep_default_na=False)
        if len(settings_df.columns) >= 2:
            for _, row in settings_df.iterrows():
                k = str(row[0]).strip()
                v = str(row[1]).strip()
                if k: settings_dict[k] = v

    # Process lists
    settings_dict["suffixes"] = [str(settings_dict.get(f"suffix_{i}", "")).strip() for i in range(1, 11) if settings_dict.get(f"suffix_{i}")]
    settings_dict["allowed_regex"] = [str(settings_dict.get(f"allowed_regex_{i}", "")).strip() for i in range(1, 21) if settings_dict.get(f"allowed_regex_{i}")]
    settings_dict["allowed_values"] = [_allowed_value_converter(settings_dict.get(f"allowed_values_{i}", "")) for i in range(1, 21) if settings_dict.get(f"allowed_values_{i}")]
    
    if "interpreter_choices" in settings_dict:
        settings_dict["interpreter_choices"] = _allowed_value_converter(settings_dict["interpreter_choices"])

    # 2. READ DATA
    data_sheet = next((s for s in xls.sheet_names if s.lower() in ["data", "items", "trials"]), None)
    if not data_sheet:
        raise ValueError("No 'data' sheet found in Excel file.")

    # Load data as strings
    df = xls.parse(data_sheet, dtype=str, keep_default_na=False, na_filter=False)
    
    # Clean whitespace
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Clean Item filenames (spaces to underscores)
    if "Item" in df.columns:
        df["Item"] = df["Item"].apply(lambda x: x.replace(" ", "_") if x else "")

        # Optional: Log specific item check
        ghost_check = df[df['Item'].str.contains("ABC-3", na=False)]
        if not ghost_check.empty:
            logger.warning(f"⚠️ GHOST ITEM FOUND in Loaded Data: {ghost_check['Item'].values[0]}")
        else:
            logger.info("✅ No ghost items (ABC-3) found in this file.")

    # Return structure matching what __init__.py expects
    return dict(
        data=df.to_dict(orient="records"),
        settings=settings_dict
    )
