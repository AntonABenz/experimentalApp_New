# urls.py  (project root)
from otree.urls import get_urlpatterns
from starlette.routing import Route

from prolific_webhook import prolific_webhook_view

routes = get_urlpatterns()

# Add webhook endpoint (allow GET for quick sanity check)
routes.append(
    Route(
        "/prolific/webhook/",
        endpoint=prolific_webhook_view,
        methods=["GET", "POST"],
        name="prolific_webhook",
    )
)
