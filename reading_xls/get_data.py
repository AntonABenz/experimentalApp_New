# reading_xls/get_data.py
import re
import time
import logging
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger("benzapp.get_data")


def _is_gsheet_url(s: str) -> bool:
    return isinstance(s, str) and "docs.google.com/spreadsheets/d/" in s


def _gsheet_export_xlsx_url(url: str) -> str:
    """
    Any link like:
      https://docs.google.com/spreadsheets/d/<ID>/edit?gid=...#gid=...
    -> https://docs.google.com/spreadsheets/d/<ID>/export?format=xlsx
    (Export returns the whole workbook with ALL tabs, not just the gid tab.)
    """
    m = re.search(r"/spreadsheets/d/([^/]+)", url)
    if not m:
        raise ValueError("Could not parse Google Sheet ID from URL")
    sheet_id = m.group(1)
    return f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"


def _download_gsheet_xlsx(url: str, cache_seconds: int = 60) -> Path:
    export_url = _gsheet_export_xlsx_url(url)
    tmp_path = Path("/tmp") / "study.xlsx"
    stamp_path = Path("/tmp") / "study.xlsx.stamp"

    # TTL cache to reduce repeated downloads on Heroku
    try:
        if tmp_path.exists() and stamp_path.exists():
            age = time.time() - float(stamp_path.read_text().strip() or "0")
            if age < cache_seconds:
                logger.info(f"Using cached Google Sheet XLSX at {tmp_path} (age={age:.1f}s)")
                return tmp_path
    except Exception:
        pass

    logger.info(f"Downloading Google Sheet XLSX from: {export_url}")
    urllib.request.urlretrieve(export_url, tmp_path)
    stamp_path.write_text(str(time.time()))
    logger.info(f"Downloaded Google Sheet to {tmp_path}")
    return tmp_path


def _load_excel(filename: str) -> Path:
    # 1) Google Sheet URL
    if _is_gsheet_url(filename):
        return _download_gsheet_xlsx(filename, cache_seconds=60)

    # 2) Local file path fallback
    root = Path(__file__).resolve().parents[1]
    candidates = [
        Path(filename),
        root / "start" / "data" / filename,
        root / "data" / filename,
        Path("start/data") / filename,
    ]

    for p in candidates:
        if p.exists():
            logger.info(f"Using local Excel file: {p}")
            return p

    raise FileNotFoundError(f"Excel file '{filename}' not found.")


def _strip_df(df: pd.DataFrame) -> pd.DataFrame:
    # Safe stripping for both pandas versions
    try:
        return df.map(lambda x: x.strip() if isinstance(x, str) else x)
    except Exception:
        return df.applymap(lambda x: x.strip() if isinstance(x, str) else x)


def _kv_sheet_to_dict(df: pd.DataFrame) -> dict:
    """
    Supports:
    - 2-col key/value sheets
    - 3-col sheets like name/value/comment (we take first two)
    """
    if df is None or df.empty:
        return {}
    df = df.dropna(how="all")
    if df.shape[1] < 2:
        return {}
    out = {}
    for _, row in df.iterrows():
        k = str(row.iloc[0]).strip() if row.iloc[0] is not None else ""
        v = row.iloc[1]
        if k:
            out[k] = "" if v is None else str(v).strip()
    return out


def _allowed_value_converter(v: str):
    if not v:
        return []
    return [item.strip() for item in str(v).split(";") if item.strip()]


def get_data(filename: str):
    """
    Reads a workbook that contains:
      - settings tab: "settings" (name/value/comment OR 2 columns)
      - data tab: "data" (trial table)
      - practice tabs: "practice_1".."practice_7" (name/value/comment)
    Returns:
      {
        "data": <list-of-dicts>,
        "settings": <dict>,
      }
    Practice tabs are embedded into settings as dicts:
      settings["Practice1"] = {...}
      settings["Practice2"] = {...}
      ...
    """
    xlsx_path = _load_excel(filename)

    try:
        xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    except Exception as e:
        logger.error(f"Critical error opening Excel: {e}")
        raise

    sheet_names = xls.sheet_names
    lower_to_real = {s.lower(): s for s in sheet_names}

    # ---- settings
    settings_dict = {}

    # Prefer exact tab named "settings" (your Google Sheet tab)
    settings_tab = lower_to_real.get("settings")
    # Fallback: any sheet containing "setting"
    if not settings_tab:
        settings_tab = next((s for s in sheet_names if "setting" in s.lower()), None)

    if settings_tab:
        df = xls.parse(settings_tab, header=None, dtype=str, keep_default_na=False, na_filter=False)
        df = _strip_df(df)
        settings_dict.update(_kv_sheet_to_dict(df))
        logger.info(f"Loaded settings from tab '{settings_tab}', keys={len(settings_dict)}")
    else:
        logger.warning("No settings tab found (expected 'settings').")

    # ---- data
    data_tab = lower_to_real.get("data") or next((s for s in sheet_names if s.lower() in {"data", "items", "trials"}), None)
    if not data_tab:
        raise ValueError("No 'data' sheet found (expected tab named 'data').")

    df_data = xls.parse(data_tab, dtype=str, keep_default_na=False, na_filter=False)
    df_data = _strip_df(df_data)

    if "Item" in df_data.columns:
        df_data["Item"] = df_data["Item"].apply(lambda x: x.replace(" ", "_") if x else "")

    data_records = df_data.to_dict(orient="records")
    logger.info(f"Loaded data from tab '{data_tab}', rows={len(data_records)}")

    # ---- practice tabs (practice_1..practice_7)
    for s in sheet_names:
        ls = s.lower()
        if ls.startswith("practice"):
            dfp = xls.parse(s, dtype=str, keep_default_na=False, na_filter=False)
            dfp = _strip_df(dfp)

            # Convert name/value/comment into dict
            # If it has headers name/value/comment, we can use those columns.
            cols_lower = [str(c).lower().strip() for c in dfp.columns]
            if "name" in cols_lower and "value" in cols_lower:
                name_col = next(c for c in dfp.columns if str(c).lower().strip() == "name")
                val_col = next(c for c in dfp.columns if str(c).lower().strip() == "value")
                practice_conf = {}
                for _, row in dfp.iterrows():
                    k = str(row.get(name_col, "")).strip()
                    v = str(row.get(val_col, "")).strip()
                    if k:
                        practice_conf[k] = v
            else:
                # fallback: treat as 2-col kv
                practice_conf = _kv_sheet_to_dict(dfp)

            nums = re.findall(r"\d+", ls)
            if nums:
                key = f"Practice{nums[0]}"
            else:
                # PracticeTest -> PracticeTest
                key = s.strip().replace(" ", "").replace("_", "")
                key = key[0].upper() + key[1:]

            settings_dict[key] = practice_conf
            logger.info(f"Loaded practice tab '{s}' into settings['{key}']")

    # ---- lists from settings
    settings_dict["suffixes"] = [
        str(settings_dict.get(f"suffix_{i}", "")).strip()
        for i in range(1, 11)
        if settings_dict.get(f"suffix_{i}")
    ]
    settings_dict["allowed_regex"] = [
        str(settings_dict.get(f"allowed_regex_{i}", "")).strip()
        for i in range(1, 21)
        if settings_dict.get(f"allowed_regex_{i}")
    ]
    settings_dict["allowed_values"] = [
        _allowed_value_converter(settings_dict.get(f"allowed_values_{i}", ""))
        for i in range(1, 21)
        if settings_dict.get(f"allowed_values_{i}")
    ]

    if "interpreter_choices" in settings_dict:
        settings_dict["interpreter_choices"] = _allowed_value_converter(settings_dict["interpreter_choices"])

    return {"data": data_records, "settings": settings_dict}
