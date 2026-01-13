import json
import re
from pathlib import Path
from typing import Dict, Any

import pandas as pd

from .convert import convert

import logging
logger = logging.getLogger("benzapp.get_data")

SETTINGS_WS = "settings"
DATA_WS = "data"
PRACTICE_WS_PREFIX = "practice_"


def _load_excel(filename: str) -> Path:
    """
    Find the Excel file on disk.
    We try a few sensible locations:
      - filename as given
      - <project_root>/start/data/filename
      - <project_root>/data/filename
    """
    root = Path(__file__).resolve().parents[1]  # project root (folder containing apps)
    candidates = [
        Path(filename),
        root / "start" / "data" / filename,
        root / "data" / filename,
    ]

    for p in candidates:
        if p.exists():
            logger.info(f"Using Excel file: {p}")
            return p

    raise FileNotFoundError(
        f"Could not locate Excel file for filename='{filename}'. "
        f"Tried: {', '.join(str(p) for p in candidates)}"
    )


def _build_practice_settings(xl: pd.ExcelFile) -> Dict[str, Dict[str, Any]]:
    """
    For each sheet whose name matches 'practice_\\d+',
    read it as a 2-column (name, value) table and group
    keys with suffix '_<number>' into lists.
    """
    practice_sheets = [
        s for s in xl.sheet_names if re.match(f"{PRACTICE_WS_PREFIX}\\d+", s, re.IGNORECASE)
    ]

    result_practice_dict: Dict[str, Dict[str, Any]] = {}

    pattern = re.compile(r"_(\d+)$")

    for sheet_name in practice_sheets:
        # FIX: keep_default_na=False ensures "None" stays "None"
        df = xl.parse(sheet_name, dtype=str, keep_default_na=False)
        
        # expect columns "name" and "value" (case-insensitive)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if not {"name", "value"}.issubset(df.columns):
            logger.warning(
                f"Practice sheet '{sheet_name}' does not have columns 'name' and 'value'. Skipping."
            )
            continue

        practice_dict: Dict[str, Any] = {}
        for _, row in df.iterrows():
            key = row["name"]
            value = row["value"]
            if not key:
                continue

            m = pattern.search(key)
            if m:
                base_key = pattern.sub("", key)
                practice_dict.setdefault(base_key, []).append(value)
            else:
                practice_dict[key] = value

        result_practice_dict[sheet_name] = practice_dict

    return result_practice_dict


def _allowed_value_converter(v: str):
    return [item.strip() for item in v.split(";") if item.strip()]


def _validate_regex_patterns(regexs):
    valid_regexs = []
    for regex in regexs:
        try:
            if not regex.startswith("^"):
                regex = "^" + regex
            if not regex.endswith("$"):
                regex = regex + "$"
            re.compile(regex)
            valid_regexs.append(regex)
        except re.error:
            logger.error(f"Invalid regex pattern: {regex}")
            raise
    return valid_regexs


def get_data(filename: str):
    """
    Local Excel-based replacement for the old Google Sheets get_data().
    Returns:
      {
        "data": <DataFrame with columns including 'sentences' etc>,
        "settings": <dict>,
        "practice_settings": <dict-of-dicts>,
      }
    """

    xlsx_path = _load_excel(filename)
    xl = pd.ExcelFile(xlsx_path)

    # ----- SETTINGS -----
    if SETTINGS_WS not in xl.sheet_names:
        raise Exception(
            f"Settings/Data spreadsheet should contain a worksheet named '{SETTINGS_WS}'"
        )

    # FIX: keep_default_na=False to prevent text strings from becoming NaN
    settings_df = xl.parse(SETTINGS_WS, header=None, dtype=str, keep_default_na=False)
    
    # Expect 2-column key/value
    settings_dict = (
        settings_df.set_index(settings_df.columns[0])
        .to_dict()
        .get(settings_df.columns[1], {})
    )

    # add suffixes list from keys suffix_1, suffix_2, ...
    settings_dict["suffixes"] = [
        value
        for key, value in settings_dict.items()
        if re.fullmatch(r"suffix_\d+", str(key))
    ]

    # allowed regex
    regexs = [
        value
        for key, value in settings_dict.items()
        if re.fullmatch(r"allowed_regex_\d+", str(key))
    ]
    settings_dict["allowed_regex"] = _validate_regex_patterns(regexs)

    # allowed values list-of-lists
    settings_dict["allowed_values"] = [
        _allowed_value_converter(value)
        for key, value in settings_dict.items()
        if re.fullmatch(r"allowed_values_\d+", str(key))
    ]

    # interpreter choices
    if "interpreter_choices" in settings_dict:
        settings_dict["interpreter_choices"] = _allowed_value_converter(
            settings_dict["interpreter_choices"]
        )

    if "interpreter_input_choices" in settings_dict:
        settings_dict["interpreter_input_choices"] = _allowed_value_converter(
            settings_dict["interpreter_input_choices"]
        )

    # practice pages flags: Practice1, Practice2, ...
    settings_dict["practice_pages"] = {
        key: bool(int(value)) if str(value).isdigit() else False
        for key, value in settings_dict.items()
        if re.fullmatch(r"Practice\d+", str(key))
    }

    # ----- DATA -----
    if DATA_WS not in xl.sheet_names:
        raise Exception(
            f"Settings/Data spreadsheet should contain a worksheet named '{DATA_WS}'"
        )

    # FIX 1: keep_default_na=False ensures "None" is read as string, not NaN
    df = xl.parse(DATA_WS, dtype={"Condition": str}, keep_default_na=False)

    # FIX 2: Drop rows where Producer is 0 or "0"
    # This removes the rows causing image errors (NA_x, D_5_4_4, etc.)
    if "Producer" in df.columns:
        df = df[df["Producer"] != 0]
        df = df[df["Producer"] != "0"]

    conv_data = convert(df)

    # ----- PRACTICE -----
    practice_settings = _build_practice_settings(xl)

    return dict(
        data=conv_data,
        settings=settings_dict,
        practice_settings=practice_settings,
    )


def long_data(filename: str):
    """
    Optional helper: if you have an 'alt_data' sheet like in the old code.
    """
    xlsx_path = _load_excel(filename)
    xl = pd.ExcelFile(xlsx_path)
    ALT_DATA_WS = "alt_data"
    if ALT_DATA_WS not in xl.sheet_names:
        raise Exception(f"No sheet named '{ALT_DATA_WS}' in {xlsx_path}")
    
    # FIX: keep_default_na=False here as well
    df = xl.parse(ALT_DATA_WS, dtype={"Condition": str}, keep_default_na=False)
    
    # Apply the same Producer filter here if this sheet is used similarly
    if "Producer" in df.columns:
        df = df[df["Producer"] != 0]
        df = df[df["Producer"] != "0"]

    conv_data = convert(df)
    return conv_data
