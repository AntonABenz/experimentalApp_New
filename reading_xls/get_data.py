import re
import random  # (kept only if you still need it elsewhere; not used now)
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
        df = xl.parse(sheet_name, dtype=str, keep_default_na=False, na_filter=False)
        df.columns = [str(c).strip().lower() for c in df.columns]
        if not {"name", "value"}.issubset(df.columns):
            continue

        practice_dict: Dict[str, Any] = {}
        for _, row in df.iterrows():
            key = str(row.get("name", "")).strip()
            value = str(row.get("value", "")).strip()
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
    return [item.strip() for item in str(v).split(";") if item.strip()]


def _validate_regex_patterns(regexs):
    valid_regexs = []
    for regex in regexs:
        regex = str(regex or "").strip()
        if not regex:
            continue
        try:
            # keep original semantics but ensure anchors exist
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

    settings_df = xl.parse(SETTINGS_WS, header=None, dtype=str, keep_default_na=False, na_filter=False)
    settings_dict = (
        settings_df.set_index(settings_df.columns[0])
        .to_dict()
        .get(settings_df.columns[1], {})
    )

    # normalize keys to str
    settings_dict = {str(k).strip(): ("" if v is None else str(v).strip()) for k, v in settings_dict.items()}

    settings_dict["suffixes"] = [
        value for key, value in settings_dict.items()
        if re.fullmatch(r"suffix_\d+", str(key).strip())
    ]

    regexs = [
        value for key, value in settings_dict.items()
        if re.fullmatch(r"allowed_regex_\d+", str(key).strip())
    ]
    settings_dict["allowed_regex"] = _validate_regex_patterns(regexs)

    settings_dict["allowed_values"] = [
        _allowed_value_converter(value)
        for key, value in settings_dict.items()
        if re.fullmatch(r"allowed_values_\d+", str(key).strip())
    ]

    if "interpreter_choices" in settings_dict:
        settings_dict["interpreter_choices"] = _allowed_value_converter(settings_dict["interpreter_choices"])

    if "interpreter_input_choices" in settings_dict:
        settings_dict["interpreter_input_choices"] = _allowed_value_converter(settings_dict["interpreter_input_choices"])

    settings_dict["practice_pages"] = {
        key: bool(int(value)) if str(value).isdigit() else False
        for key, value in settings_dict.items()
        if re.fullmatch(r"Practice\d+", str(key))
    }

    # ----- DATA -----
    if DATA_WS not in xl.sheet_names:
        raise Exception(f"Worksheet '{DATA_WS}' not found")

    # IMPORTANT:
    # - dtype=str forces everything into strings
    # - keep_default_na=False + na_filter=False prevents pandas treating "None"/"NA" as missing
    df = xl.parse(
        DATA_WS,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )

    # Strip whitespace everywhere, keep empty cells as ""
    df = df.applymap(lambda x: str(x).strip() if x is not None else "")

    # Deterministic image cleanup ONLY
    if "Item" in df.columns:
        fallback_image = "d-A-B-BC-3"

        def fix_item(v: str) -> str:
            v = str(v or "").strip()
            if v == "NA_x":
                return fallback_image
            if v.lower() in {"", "nan"}:
                return ""
            # optional: normalize spaces
            return v.replace(" ", "_")

        df["Item"] = df["Item"].map(fix_item)

    # Convert to your internal format
    conv_data = convert(df)

    # ----- PRACTICE -----
    practice_settings = _build_practice_settings(xl)

    return dict(
        data=conv_data,
        settings=settings_dict,
        practice_settings=practice_settings,
    )
