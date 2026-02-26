# urls.py (project root)
from otree.urls import get_urlpatterns
from starlette.routing import Route

from prolific_webhook import prolific_webhook_view

urlpatterns = get_urlpatterns()

urlpatterns += [
    # Starlette routing: must start with "/"
    Route("/prolific/webhook", prolific_webhook_view, methods=["GET", "POST"]),
    Route("/prolific/webhook/", prolific_webhook_view, methods=["GET", "POST"]),
]
