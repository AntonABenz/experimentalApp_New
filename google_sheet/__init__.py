# google_sheet/__init__.py
from .loader import load_for_session, load_from_path  # re-export for convenience

__all__ = ["load_for_session", "load_from_path"]
