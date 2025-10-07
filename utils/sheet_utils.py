from typing import Dict

def image_src(row: Dict, schema: Dict[str, str]) -> str:
    url_col = schema['image_url']
    base = (row.get(url_col) or '').strip()
    fn = (row.get(schema['filename']) or '').strip()
    ext = (row.get(schema['extension']) or '').strip() or 'png'

    if base:
        if base.lower().startswith('http'):
            if fn:
                sep = '' if base.endswith('/') else '/'
                return f"{base}{sep}{fn}.{ext}"
            return base
        if fn:
            sep = '' if base.endswith('/') else '/'
            return f"{base}{sep}{fn}.{ext}"
        return base

    if fn:
        return f"/static/images/{fn}.{ext}"
    return ""
