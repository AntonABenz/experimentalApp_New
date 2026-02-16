# asgi.py
from starlette.applications import Starlette
from starlette.routing import Mount, Route

from otree.asgi import app as otree_app
from prolific_webhook import prolific_webhook_view  # must be Starlette-compatible

app = Starlette(
    routes=[
        Route("/prolific/webhook/", prolific_webhook_view, methods=["POST", "GET", "HEAD"]),
        # optional debug endpoint
        Route("/prolific/ping/", lambda request: {"ok": True}, methods=["GET"]),
        Mount("/", app=otree_app),
    ]
)
