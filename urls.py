# urls.py (project root)

from otree.urls import get_urlpatterns
from starlette.routing import Route

from prolific_webhook import prolific_webhook_view

urlpatterns = get_urlpatterns()

# Add BOTH versions (with and without trailing slash),
# because Starlette routing is strict about slashes.
urlpatterns += [
    Route("/prolific/webhook", prolific_webhook_view, methods=["GET", "POST"]),
    Route("/prolific/webhook/", prolific_webhook_view, methods=["GET", "POST"]),
]
