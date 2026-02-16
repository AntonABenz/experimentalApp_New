# urls.py
from otree.urls import get_urlpatterns
from starlette.routing import Route
from starlette.responses import PlainTextResponse

from prolific_webhook import prolific_webhook_view

def ping(request):
    return PlainTextResponse("ok", status_code=200)

urlpatterns = get_urlpatterns() + [
    Route("/prolific/ping/", endpoint=ping, methods=["GET"]),
    Route("/prolific/webhook/", endpoint=prolific_webhook_view, methods=["POST", "GET"]),
]
