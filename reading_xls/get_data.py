import pandas as pd
import re
import logging
from pathlib import Path
import numpy as np

logger = logging.getLogger("benzapp.get_data")

SETTINGS_WS = "settings"
DATA_WS = "data"
PRACTICE_WS_PREFIX = "practice_"

def _load_excel(filename: str) -> Path:
    # Look in multiple locations for the file
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(filename),
        root / "start" / "data" / filename,
        root / "data" / filename,
    ]

    for p in candidates:
        if p.exists():
            logger.info(f"✅ FOUND EXCEL: {p.resolve()}")
            return p

    raise FileNotFoundError(f"Excel file '{filename}' not found. Checked: {[str(c) for c in candidates]}")

def _allowed_value_converter(v: str):
    if not v: return []
    return [item.strip() for item in str(v).split(";") if item.strip()]

def get_data(filename: str):
    xlsx_path = _load_excel(filename)
    
    # Load Workbook
    xls = pd.ExcelFile(xlsx_path, engine="openpyxl")

    # 1. READ SETTINGS
    # Find settings sheet (case insensitive)
    settings_sheet = next((s for s in xls.sheet_names if "setting" in s.lower()), None)
    if not settings_sheet:
        raise ValueError("No 'settings' sheet found in Excel file")

    settings_df = xls.parse(settings_sheet, header=None, dtype=str, keep_default_na=False)
    # Convert 2-column format to dict
    settings_dict = {}
    if len(settings_df.columns) >= 2:
        for _, row in settings_df.iterrows():
            k = str(row[0]).strip()
            v = str(row[1]).strip()
            if k: settings_dict[k] = v

    # Process lists and regex
    settings_dict["suffixes"] = [v for k, v in settings_dict.items() if k.startswith("suffix_")]
    settings_dict["allowed_regex"] = [v for k, v in settings_dict.items() if k.startswith("allowed_regex_")]
    settings_dict["allowed_values"] = [
        _allowed_value_converter(v) for k, v in settings_dict.items() if k.startswith("allowed_values_")
    ]
    if "interpreter_choices" in settings_dict:
        settings_dict["interpreter_choices"] = _allowed_value_converter(settings_dict["interpreter_choices"])

    # 2. READ DATA
    # Find data sheet
    data_sheet = next((s for s in xls.sheet_names if s.lower() in ["data", "items", "trials"]), None)
    if not data_sheet:
        raise ValueError("No 'data' sheet found in Excel file")

    df = xls.parse(data_sheet, dtype=str, keep_default_na=False, na_filter=False)
    
    # Clean whitespace
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Basic Image Cleanup (Spaces to underscores)
    if "Item" in df.columns:
        df["Item"] = df["Item"].apply(lambda x: x.replace(" ", "_") if x else "")

    # Convert to list of dicts
    data_records = df.to_dict(orient="records")

    return dict(
        data=data_records,
        settings=settings_dict
    )
