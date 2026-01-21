import pandas as pd
import logging
import re
import time
import urllib.request
from pathlib import Path
import io

logger = logging.getLogger("benzapp.get_data")

def _is_gsheet_url(s: str) -> bool:
    return isinstance(s, str) and "docs.google.com/spreadsheets/d/" in s

def _gsheet_export_xlsx_url(url: str) -> str:
    m = re.search(r"/spreadsheets/d/([^/]+)", url)
    if not m:
        raise ValueError("Could not parse Google Sheet ID from URL")
    sheet_id = m.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"

def _download_gsheet_xlsx(url: str, cache_seconds: int = 60) -> Path:
    export_url = _gsheet_export_xlsx_url(url)
    tmp_path = Path("/tmp") / "study.xlsx"
    stamp_path = Path("/tmp") / "study.xlsx.stamp"

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
    if _is_gsheet_url(filename):
        return _download_gsheet_xlsx(filename, cache_seconds=60)

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
    xlsx_path = _load_excel(filename)
    
    try:
        xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    except Exception as e:
        logger.error(f"Critical error opening Excel: {e}")
        raise e
    
    # -------------------------------------------------------
    # 1. READ GLOBAL SETTINGS
    # -------------------------------------------------------
    settings_dict = {}
    settings_sheet = next((s for s in xls.sheet_names if "setting" in s.lower()), None)
    
    if settings_sheet:
        df = xls.parse(settings_sheet, header=None, dtype=str, keep_default_na=False)
        if len(df.columns) >= 2:
            for _, row in df.iterrows():
                k, v = str(row[0]).strip(), str(row[1]).strip()
                if k: settings_dict[k] = v

    # -------------------------------------------------------
    # 2. READ DATA & PRACTICE SHEETS
    # -------------------------------------------------------
    all_data_frames = []
    
    for sheet_name in xls.sheet_names:
        lower_name = sheet_name.lower()
        
        # A) Load Trial Data
        if lower_name in ["data", "items", "trials"]:
            logger.info(f"Reading Data sheet: {sheet_name}")
            try:
                df = xls.parse(sheet_name, dtype=str, keep_default_na=False, na_filter=False)
                # Clean
                try: df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
                except AttributeError: df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
                
                if "Item" in df.columns:
                    df["Item"] = df["Item"].apply(lambda x: x.replace(" ", "_") if x else "")
                all_data_frames.append(df)
            except Exception as e:
                logger.warning(f"Error reading data sheet {sheet_name}: {e}")

        # B) Load Practice Configuration
        elif lower_name.startswith("practice"):
            logger.info(f"Reading Practice Config sheet: {sheet_name}")
            try:
                # Expecting format: name, value, comment
                df = xls.parse(sheet_name, dtype=str, keep_default_na=False, na_filter=False)
                practice_conf = {}
                
                cols = [c.lower() for c in df.columns]
                if 'name' in cols and 'value' in cols:
                    for _, row in df.iterrows():
                        # Find actual column names
                        name_col = next(c for c in df.columns if c.lower() == 'name')
                        val_col = next(c for c in df.columns if c.lower() == 'value')
                        
                        k = str(row[name_col]).strip()
                        v = str(row[val_col]).strip()
                        if k: practice_conf[k] = v
                    
                    # --- KEY FIX: NORMALIZE KEY TO 'PracticeX' ---
                    # Converts "practice_1" -> "Practice1"
                    # Converts "Practice 2" -> "Practice2"
                    nums = re.findall(r'\d+', sheet_name)
                    if nums:
                        clean_key = f"Practice{nums[0]}"
                        settings_dict[clean_key] = practice_conf
                        logger.info(f"Loaded {sheet_name} as settings key '{clean_key}'")
                    else:
                        # Fallback for "PracticeTest" etc.
                        clean_key = sheet_name.strip().replace(" ", "").replace("_", "")
                        # Capitalize first letter
                        clean_key = clean_key[0].upper() + clean_key[1:]
                        settings_dict[clean_key] = practice_conf
                        
                else:
                    logger.warning(f"Sheet {sheet_name} missing 'name'/'value' cols")
            except Exception as e:
                logger.warning(f"Error reading practice sheet {sheet_name}: {e}")

    if not all_data_frames:
        raise ValueError("No 'Data', 'Items', or 'Trials' sheet found in Excel.")

    combined_df = pd.concat(all_data_frames, ignore_index=True)
    
    # -------------------------------------------------------
    # 3. PROCESS SETTINGS LISTS
    # -------------------------------------------------------
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

    return dict(
        data=combined_df.to_dict(orient="records"),
        settings=settings_dict
    )
