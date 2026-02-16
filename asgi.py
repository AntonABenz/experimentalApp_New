# asgi.py
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse

from otree.asgi import app as otree_app
from prolific_webhook import prolific_webhook_view

async def ping(request):
    return JSONResponse({"ok": True})

app = Starlette(
    routes=[
        Route("/prolific/ping/", ping, methods=["GET"]),
        Route("/prolific/webhook/", prolific_webhook_view, methods=["POST", "GET", "HEAD"]),
        Mount("/", app=otree_app),
    ]
)
