from os import environ


def as_bool(v, default=False):
    if v is None:
        return default
    return v in ("1", "true", "True", True)


# Provided by you
INSTRUCTIONS_URL = "https://docs.google.com/document/d/1PBL73XGMwln6uTFBXW2UaGoeb5-pM5OClvXXUe6YCP0/edit?tab=t.0"
INTRODUCTION_URL = "https://docs.google.com/document/d/1frtr8zzT1KehperGjaNJNKgEh9fpZvHc0OvPuUgPblo/edit?tab=t.0"


SESSION_CONFIGS = [
    dict(
        name="full_study",
        display_name="Full study",
        app_sequence=["start", "img_desc"],
        num_demo_participants=8,
        filename="https://docs.google.com/spreadsheets/d/1IQiLSLk9LFt8EK9fr5l_Z8sx8-p9VotoHmlXP5s7mRs/edit?gid=1525060338#gid=1525060338",

        # used by img_desc + start.Consent
        for_prolific=True,
        expand_slots=True,

        # cohort setup (1–4 => Exp1, 5–8 => Exp2, etc.)
        cohort_size=4,

        # NEW: doc links (stored in session.vars in creating_session)
        instructions_url=INSTRUCTIONS_URL,
        introduction_url=INTRODUCTION_URL,
    ),

    dict(
        name="practice_pages",
        display_name="Practice pages only",
        app_sequence=["start"],
        num_demo_participants=4,
        filename="disjunctionExpTest8EN_1.xlsx",
        for_prolific=False,
        expand_slots=False,

        # optional for practice too
        instructions_url=INSTRUCTIONS_URL,
        introduction_url=INTRODUCTION_URL,
    ),
]


SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00,
    participation_fee=0.00,
)


PARTICIPANT_FIELDS = [
    "prolific_id",
    "study_id",
    "prolific_session_id",
    "full_study_completed",
]


SESSION_FIELDS = [
    # from start/__init__.py
    "practice_settings",
    "sheet_settings",
    "interpreter_title",
    "interpreter_choices",
    "suffixes",
    "allowed_values",
    "EndOfIntroText",

    # from img_desc
    "active_batch",
    "user_data",
    "num_rounds",
    "practice_pages",
    "user_settings",
    "s3path",
    "extension",
    "prefix",
    "allowed_regex",
    "caseflag",
    "interpreter_input_type",
    "interpreter_input_choices",
    "max_users",
    "batch_size",

    # NEW: keep these accessible in session.vars
    "cohort_size",
    "instructions_url",
    "introduction_url",
]


LANGUAGE_CODE = "en"
REAL_WORLD_CURRENCY_CODE = "USD"
USE_POINTS = True

ADMIN_USERNAME = environ.get("OTREE_ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = environ.get("OTREE_ADMIN_PASSWORD", "admin")
SECRET_KEY = environ.get("OTREE_SECRET_KEY", "dev-secret")

OTREE_PRODUCTION = as_bool(environ.get("OTREE_PRODUCTION", "0"), False)

INSTALLED_APPS = ["otree"]
