# google_sheet/loader.py
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


# ---- Defaults (kept in sync with settings.py fallbacks) -----------------------
TAB_SETTINGS = os.environ.get("BENZ_SHEET_TAB_SETTINGS", "Settings")
TAB_DATA = os.environ.get("BENZ_SHEET_TAB_DATA", "Data")
TAB_PREFIX_PRACTICE = os.environ.get("BENZ_SHEET_TAB_PREFIX_PRACTICE", "Practice")

COL_IMAGE_URL = os.environ.get("BENZ_COL_IMAGE_URL", "s3path")
COL_FILENAME = os.environ.get("BENZ_COL_FILENAME", "filename")
COL_EXTENSION = os.environ.get("BENZ_COL_EXTENSION", "extension")
COL_PROMPT = os.environ.get("BENZ_COL_PROMPT", "prompt")
COL_DESCRIPTION = os.environ.get("BENZ_COL_DESCRIPTION", "description")
COL_PRACTICE = os.environ.get("BENZ_COL_PRACTICE", "is_practice")

REQUIRED_DATA_COLS = {
    COL_IMAGE_URL, COL_FILENAME, COL_EXTENSION, COL_PROMPT, COL_DESCRIPTION
}


# ---- Public API ----------------------------------------------------------------
def load_for_session(session) -> Dict[str, Any]:
    """
    Reads the spreadsheet indicated by session.config['filename'] and returns a dict:
      {
        'sheet_settings': Dict[str, Any],
        'sheet_data': List[Dict[str, Any]],
        'practices': Dict[str, List[Dict[str, Any]]],
        'sheet_meta': Dict[str, Any],
      }
    NOTE: This function DOES NOT mutate session.vars; your app should assign it.
    """
    filename = session.config.get("filename")
    if not filename:
        raise ValueError("Session config missing 'filename'")

    # Heuristic: if it ends with .xlsx, treat as local Excel file
    if filename.lower().endswith(".xlsx"):
        # Resolve relative paths against repo root and /start/data
        base_dir = Path(__file__).resolve().parent.parent  # project root (folder containing google_sheet/)
        candidate_paths = [
            Path(filename),
            base_dir / "start" / "data" / filename,
            base_dir / filename,
        ]
        exc = None
        for p in candidate_paths:
            try:
                if p.exists():
                    return load_from_path(p)
            except Exception as e:
                exc = e
        # If we got here, we didn't find a working path
        raise FileNotFoundError(
            f"Could not locate Excel file for filename='{filename}'. "
            f"Tried: {', '.join(str(p) for p in candidate_paths)}"
        ) from exc

    # Otherwise, treat as a Google Sheet name/ID/URL you handle elsewhere.
    # (Implement your preferred GSheets backend here if needed.)
    raise ValueError(
        "Non-.xlsx 'filename' detected. Add a Google Sheets loader or set "
        "BENZ_SPREADSHEET_NAME to a .xlsx file name."
    )


def load_from_path(xlsx_path: Path) -> Dict[str, Any]:
    """
    Load Settings, Data, and Practice_* tabs from a local Excel file.
    Returns a dict suitable for assigning into session.vars.
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Excel file not found: {xlsx_path}")

    # Read sheets
    xl = pd.ExcelFile(xlsx_path)
    sheet_names = set(xl.sheet_names)

    # ---- Settings
    settings_dict: Dict[str, Any] = {}
    if TAB_SETTINGS in sheet_names:
        # Expect a 2-column key/value table; we’ll coerce into dict
        settings_df = xl.parse(TAB_SETTINGS).dropna(how="all")
        if settings_df.shape[1] >= 2:
            # Take first 2 columns as key/value
            kv = settings_df.iloc[:, :2].dropna()
            for k, v in kv.values.tolist():
                if pd.isna(k):
                    continue
                settings_dict[str(k).strip()] = _maybe_cast(v)
        else:
            # If it's a single-row settings sheet, coerce via to_dict
            settings_dict = settings_df.to_dict(orient="list")

    # ---- Data
    if TAB_DATA not in sheet_names:
        raise ValueError(f"'{TAB_DATA}' sheet is required in {xlsx_path}")
    data_df = xl.parse(TAB_DATA).dropna(how="all")
    _require_columns(data_df, REQUIRED_DATA_COLS, where=TAB_DATA)
    data_records: List[Dict[str, Any]] = _records(data_df)

    # ---- Practice_* tabs
    practices: Dict[str, List[Dict[str, Any]]] = {}
    practice_tabs = sorted([s for s in sheet_names if s.startswith(TAB_PREFIX_PRACTICE)])
    for tab in practice_tabs:
        df = xl.parse(tab).dropna(how="all")
        # Use the intersection to be permissive:
        needed = REQUIRED_DATA_COLS.intersection(set(df.columns))
        if needed:
            _require_columns(df, needed, where=tab)
        practices[tab] = _records(df)

    # ---- Meta
    meta = {
        "source_path": str(xlsx_path),
        "tabs": list(sheet_names),
        "settings_tab": TAB_SETTINGS,
        "data_tab": TAB_DATA,
        "practice_prefix": TAB_PREFIX_PRACTICE,
        "columns": {
            "image_url": COL_IMAGE_URL,
            "filename": COL_FILENAME,
            "extension": COL_EXTENSION,
            "prompt": COL_PROMPT,
            "description": COL_DESCRIPTION,
            "practice": COL_PRACTICE,
        },
        "row_counts": {
            "data": len(data_records),
            "practice_total": sum(len(v) for v in practices.values()),
        },
    }

    return {
        "sheet_settings": settings_dict,
        "sheet_data": data_records,
        "practices": practices,
        "sheet_meta": meta,
    }


# ---- Helpers -------------------------------------------------------------------
def _records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    # Convert NaN to None and strip strings
    cleaned = df.where(pd.notnull(df), None)
    recs: List[Dict[str, Any]] = []
    for r in cleaned.to_dict(orient="records"):
        recs.append({k: _strip(v) for k, v in r.items()})
    return recs


def _require_columns(df: pd.DataFrame, required: set[str], where: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sheet '{where}': {missing}")


def _strip(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip()
    return v


def _maybe_cast(v: Any) -> Any:
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in {"true", "false"}:
            return s.lower() == "true"
        try:
            if "." in s:
                return float(s)
            return int(s)
        except Exception:
            return s
    return v
