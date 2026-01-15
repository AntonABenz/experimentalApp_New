import pandas as pd
import logging
from pathlib import Path

logger = logging.getLogger("benzapp.get_data")

def _load_excel(filename: str) -> Path:
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(filename),
        root / "start" / "data" / filename,
        root / "data" / filename,
        Path("start/data") / filename,
    ]

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(f"Excel file '{filename}' not found.")

def _allowed_value_converter(v: str):
    if not v: return []
    return [item.strip() for item in str(v).split(";") if item.strip()]

def get_data(filename: str):
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
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Clean Item filenames (spaces to underscores)
    if "Item" in df.columns:
        df["Item"] = df["Item"].apply(lambda x: x.replace(" ", "_") if x else "")

    return dict(
        data=df.to_dict(orient="records"),
        settings=settings_dict
    )
