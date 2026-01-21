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
    except Exception: pass
    logger.info(f"Downloading Google Sheet XLSX from export URL: {export_url}")
    urllib.request.urlretrieve(export_url, tmp_path)
    stamp_path.write_text(str(time.time()))
    return tmp_path

def _load_excel(filename: str) -> Path:
    if _is_gsheet_url(filename):
        return _download_gsheet_xlsx(filename, cache_seconds=60)
    root = Path(__file__).resolve().parents[1]
    candidates = [Path(filename), root/"start"/"data"/filename, root/"data"/filename, Path("start/data")/filename]
    for p in candidates:
        if p.exists(): return p
    raise FileNotFoundError(f"Excel file '{filename}' not found.")

def _clean_val(v):
    if pd.isna(v): return ""
    return str(v).strip()

def _allowed_value_converter(v):
    s = _clean_val(v)
    return [item.strip() for item in s.split(";") if item.strip()]

def get_data(filename: str):
    xlsx_path = _load_excel(filename)
    try: xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    except Exception as e:
        logger.error(f"Critical error opening Excel: {e}")
        raise e

    # 1. READ SETTINGS
    settings_dict = {}
    settings_sheet = next((s for s in xls.sheet_names if "setting" in s.lower()), None)
    if settings_sheet:
        df = xls.parse(settings_sheet, header=None, dtype=str, keep_default_na=False)
        if len(df.columns) >= 2:
            for _, row in df.iterrows():
                k, v = _clean_val(row[0]), _clean_val(row[1])
                if k: settings_dict[k] = v

    # 2. READ DATA & PRACTICE
    all_data_frames = []
    for sheet_name in xls.sheet_names:
        lower_name = sheet_name.lower()
        
        # A) Trial Data
        if lower_name in ["data", "items", "trials"]:
            logger.info(f"Reading Data sheet: {sheet_name}")
            df = xls.parse(sheet_name, dtype=str, keep_default_na=False)
            # Clean all cells
            df = df.map(_clean_val) # Pandas 2.0+
            if "Item" in df.columns:
                df["Item"] = df["Item"].apply(lambda x: x.replace(" ", "_"))
            all_data_frames.append(df)

        # B) Practice Configuration
        elif lower_name.startswith("practice"):
            logger.info(f"Reading Practice Config sheet: {sheet_name}")
            df = xls.parse(sheet_name, dtype=str, keep_default_na=False)
            cols_lower = [str(c).lower() for c in df.columns]
            
            if "name" in cols_lower and "value" in cols_lower:
                practice_conf = {}
                name_idx = cols_lower.index("name")
                val_idx = cols_lower.index("value")
                
                for _, row in df.iterrows():
                    k = _clean_val(row.iloc[name_idx])
                    v = _clean_val(row.iloc[val_idx])
                    if k: practice_conf[k] = v
                
                # Normalize Key (Practice 1 -> Practice1)
                nums = re.findall(r"\d+", sheet_name)
                if nums: clean_key = f"Practice{nums[0]}"
                else: clean_key = sheet_name.strip().replace(" ", "").replace("_", "").capitalize()
                
                settings_dict[clean_key] = practice_conf
                logger.info(f"Loaded {sheet_name} as '{clean_key}' with {len(practice_conf)} items")

    if not all_data_frames:
        raise ValueError("No Data sheet found")

    combined_df = pd.concat(all_data_frames, ignore_index=True)

    # 3. Process Lists
    settings_dict["suffixes"] = [_clean_val(settings_dict.get(f"suffix_{i}")) for i in range(1, 11) if settings_dict.get(f"suffix_{i}")]
    settings_dict["allowed_regex"] = [_clean_val(settings_dict.get(f"allowed_regex_{i}")) for i in range(1, 21) if settings_dict.get(f"allowed_regex_{i}")]
    settings_dict["allowed_values"] = [_allowed_value_converter(settings_dict.get(f"allowed_values_{i}")) for i in range(1, 21) if settings_dict.get(f"allowed_values_{i}")]
    if "interpreter_choices" in settings_dict:
        settings_dict["interpreter_choices"] = _allowed_value_converter(settings_dict["interpreter_choices"])

    return dict(data=combined_df.to_dict(orient="records"), settings=settings_dict)
