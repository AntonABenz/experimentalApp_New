import otree.urls
from starlette.routing import Route

from prolific_webhook import prolific_webhook_view

urlpatterns = otree.urls.get_urlpatterns() + [
    Route("/prolific/webhook/", prolific_webhook_view, methods=["POST", "GET"]),
]
