import os
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DESC_DIR = os.path.normpath(os.path.join(BASE_DIR, '..', 'img_desc'))

@lru_cache(maxsize=2048)
def read_desc(basename: str) -> str:
    if not basename:
        return ''
    for ext in ('.txt', '.md'):
        p = os.path.join(IMG_DESC_DIR, basename + ext)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return f.read().strip()
    return ''
