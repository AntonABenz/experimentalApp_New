# asgi.py
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse, RedirectResponse, PlainTextResponse

from otree.asgi import app as otree_app
from prolific_webhook import prolific_webhook_view

# NEW
from otree.models import Session  # Django ORM

async def ping(request):
    return JSONResponse({"ok": True})

async def entry(request):
    """
    Prolific landing endpoint.
    Redirects to /join/<session_code>?<original_querystring>
    IMPORTANT: You must create at least one session in admin (full_study) beforehand.
    """
    qs = str(request.url.query or "")
    suffix = f"?{qs}" if qs else ""

    # pick the latest real session for config_name='full_study'
    s = (
        Session.objects
        .filter(is_demo=False, config_name="full_study")
        .order_by("-id")
        .first()
    )

    if not s:
        return PlainTextResponse("No active session found. Create a 'full_study' session in admin.", status_code=500)

    return RedirectResponse(f"/join/{s.code}{suffix}", status_code=302)

app = Starlette(
    routes=[
        Route("/prolific/ping/", ping, methods=["GET"]),
        Route("/prolific/webhook/", prolific_webhook_view, methods=["POST", "GET", "HEAD"]),
        # NEW:
        Route("/entry", entry, methods=["GET"]),
        Route("/entry/", entry, methods=["GET"]),
        Mount("/", app=otree_app),
    ]
)
