from typing import Optional

def completion_url(completion_code: Optional[str] = None) -> str:
    base = "https://app.prolific.com/submissions/complete"
    return f"{base}?cc={completion_code}" if completion_code else base

def maybe_expand_slots(enabled: bool, batch_done: bool) -> None:
    if enabled and batch_done:
        # TODO: add your API call here if you do slot expansion.
        pass
