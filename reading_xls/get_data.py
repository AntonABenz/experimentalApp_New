import json
import re
import random  # <--- Added for random selection
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
    root = Path(__file__).resolve().parents[1]
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
    practice_sheets = [
        s for s in xl.sheet_names if re.match(f"{PRACTICE_WS_PREFIX}\\d+", s, re.IGNORECASE)
    ]

    result_practice_dict: Dict[str, Dict[str, Any]] = {}
    pattern = re.compile(r"_(\d+)$")

    for sheet_name in practice_sheets:
        df = xl.parse(sheet_name, dtype=str, keep_default_na=False)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if not {"name", "value"}.issubset(df.columns):
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
    xlsx_path = _load_excel(filename)
    xl = pd.ExcelFile(xlsx_path)

    # ----- SETTINGS -----
    if SETTINGS_WS not in xl.sheet_names:
        raise Exception(f"Worksheet '{SETTINGS_WS}' not found")

    settings_df = xl.parse(SETTINGS_WS, header=None, dtype=str, keep_default_na=False)
    settings_dict = (
        settings_df.set_index(settings_df.columns[0])
        .to_dict()
        .get(settings_df.columns[1], {})
    )

    settings_dict["suffixes"] = [
        value for key, value in settings_dict.items() if re.fullmatch(r"suffix_\d+", str(key))
    ]

    regexs = [
        value for key, value in settings_dict.items() if re.fullmatch(r"allowed_regex_\d+", str(key))
    ]
    settings_dict["allowed_regex"] = _validate_regex_patterns(regexs)

    settings_dict["allowed_values"] = [
        _allowed_value_converter(value)
        for key, value in settings_dict.items()
        if re.fullmatch(r"allowed_values_\d+", str(key))
    ]

    if "interpreter_choices" in settings_dict:
        settings_dict["interpreter_choices"] = _allowed_value_converter(
            settings_dict["interpreter_choices"]
        )

    if "interpreter_input_choices" in settings_dict:
        settings_dict["interpreter_input_choices"] = _allowed_value_converter(
            settings_dict["interpreter_input_choices"]
        )

    settings_dict["practice_pages"] = {
        key: bool(int(value)) if str(value).isdigit() else False
        for key, value in settings_dict.items()
        if re.fullmatch(r"Practice\d+", str(key))
    }

    # ----- DATA -----
    if DATA_WS not in xl.sheet_names:
        raise Exception(f"Worksheet '{DATA_WS}' not found")

    df = xl.parse(DATA_WS, dtype={"Condition": str}, keep_default_na=False)

    # --- FIX START: Randomly repair broken items ---
    if "Item" in df.columns:
        # 1. Collect all "valid" images (starting with 'd-' and not 'NA')
        # We assume any item starting with 'd-' is a legit image file.
        valid_pool = [
            str(x).strip() for x in df["Item"].unique() 
            if str(x).startswith("d-") and "NA" not in str(x)
        ]
        
        # Fallback if the pool is somehow empty
        if not valid_pool:
            valid_pool = ["d-A-B-BC-3"] 

        def repair_randomly(row):
            item = str(row.get("Item", "")).strip()
            prod = str(row.get("Producer", ""))
            
            # Identify broken rows
            is_broken = (
                item == "NA_x" or 
                item.startswith("D_") or 
                prod == "0" or 
                prod == "0.0"
            )
            
            if is_broken:
                # Pick a random VALID image from the pool
                return random.choice(valid_pool)
            
            return item

        # Apply the random repair
        df["Item"] = df.apply(repair_randomly, axis=1)
    # --- FIX END ---

    conv_data = convert(df)

    # ----- PRACTICE -----
    practice_settings = _build_practice_settings(xl)

    return dict(
        data=conv_data,
        settings=settings_dict,
        practice_settings=practice_settings,
    )


def long_data(filename: str):
    xlsx_path = _load_excel(filename)
    xl = pd.ExcelFile(xlsx_path)
    ALT_DATA_WS = "alt_data"
    if ALT_DATA_WS not in xl.sheet_names:
        raise Exception(f"No sheet named '{ALT_DATA_WS}' in {xlsx_path}")
    
    df = xl.parse(ALT_DATA_WS, dtype={"Condition": str}, keep_default_na=False)
    
    # Apply same random repair logic here if needed
    if "Item" in df.columns:
        valid_pool = [
            str(x).strip() for x in df["Item"].unique() 
            if str(x).startswith("d-") and "NA" not in str(x)
        ]
        if not valid_pool: valid_pool = ["d-A-B-BC-3"]

        def repair_randomly(row):
            item = str(row.get("Item", "")).strip()
            prod = str(row.get("Producer", ""))
            is_broken = (item == "NA_x" or item.startswith("D_") or prod == "0" or prod == "0.0")
            if is_broken:
                return random.choice(valid_pool)
            return item

        df["Item"] = df.apply(repair_randomly, axis=1)

    conv_data = convert(df)
    return conv_data
