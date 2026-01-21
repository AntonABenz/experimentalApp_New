import pandas as pd
import logging
import re
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("benzapp.get_data")


# -----------------------------
# Google Sheet download helpers
# -----------------------------
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

    # TTL cache
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
    """
    You said there is no Excel in git anymore.
    So we ONLY support Google Sheet URLs here.
    """
    if _is_gsheet_url(filename):
        return _download_gsheet_xlsx(filename, cache_seconds=60)

    raise FileNotFoundError(
        "No local Excel support enabled. Please set session config 'filename' to a Google Sheet URL."
    )


# -----------------------------
# Parsing helpers
# -----------------------------
def _allowed_value_converter(v: str):
    if not v:
        return []
    return [item.strip() for item in str(v).split(";") if item.strip()]


def get_data(filename: str):
    """
    Returns:
      dict(data=<list of dict rows>, settings=<dict>)
    """
    xlsx_path = _load_excel(filename)

    try:
        xls = pd.ExcelFile(xlsx_path, engine="openpyxl")
    except Exception as e:
        logger.error(f"Critical error opening Excel: {e}")
        raise

    # -------------------------------------------------------
    # 1) READ SETTINGS SHEET
    # -------------------------------------------------------
    settings_dict = {}
    settings_sheet = next((s for s in xls.sheet_names if "setting" in s.lower()), None)

    if settings_sheet:
        df = xls.parse(settings_sheet, header=None, dtype=str, keep_default_na=False, na_filter=False)
        if len(df.columns) >= 2:
            for _, row in df.iterrows():
                k = str(row[0]).strip()
                v = str(row[1]).strip()
                if k:
                    settings_dict[k] = v
    else:
        logger.warning("No settings sheet found (sheet name containing 'setting').")

    # -------------------------------------------------------
    # 2) READ DATA SHEET(S) + PRACTICE SHEETS
    # -------------------------------------------------------
    all_data_frames = []

    for sheet_name in xls.sheet_names:
        lower_name = sheet_name.lower()

        # A) Trial data
        if lower_name in ["data", "items", "trials"]:
            logger.info(f"Reading Data sheet: {sheet_name}")
            df = xls.parse(sheet_name, dtype=str, keep_default_na=False, na_filter=False)

            # strip strings
            try:
                df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
            except AttributeError:
                df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

            # normalize Item names: spaces -> underscore
            if "Item" in df.columns:
                df["Item"] = df["Item"].apply(lambda x: x.replace(" ", "_") if x else "")

            all_data_frames.append(df)

        # B) Practice sheets: practice_1, practice_2, ...
        elif lower_name.startswith("practice"):
            logger.info(f"Reading Practice Config sheet: {sheet_name}")
            df = xls.parse(sheet_name, dtype=str, keep_default_na=False, na_filter=False)

            # expecting columns including "name" and "value"
            cols_lower = [c.lower() for c in df.columns]
            if "name" not in cols_lower or "value" not in cols_lower:
                logger.warning(f"Sheet {sheet_name} missing 'name'/'value' columns")
                continue

            name_col = next(c for c in df.columns if c.lower() == "name")
            val_col = next(c for c in df.columns if c.lower() == "value")

            practice_conf = {}
            for _, row in df.iterrows():
                k = str(row[name_col]).strip()
                v = str(row[val_col]).strip()
                if k:
                    practice_conf[k] = v

            # normalize key to PracticeX
            nums = re.findall(r"\d+", sheet_name)
            if nums:
                clean_key = f"Practice{nums[0]}"
            else:
                clean_key = sheet_name.strip().replace(" ", "").replace("_", "")
                clean_key = clean_key[0].upper() + clean_key[1:]

            settings_dict[clean_key] = practice_conf
            logger.info(f"Loaded {sheet_name} as settings key '{clean_key}'")

    if not all_data_frames:
        raise ValueError("No 'Data', 'Items', or 'Trials' sheet found in the Google Sheet.")

    combined_df = pd.concat(all_data_frames, ignore_index=True)

    # -------------------------------------------------------
    # 3) PROCESS SETTINGS LISTS
    # -------------------------------------------------------
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

    return dict(
        data=combined_df.to_dict(orient="records"),
        settings=settings_dict,
    )
