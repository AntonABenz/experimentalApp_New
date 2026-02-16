from django.urls import path
import otree.urls

from prolific_webhook import prolific_webhook_view

urlpatterns = otree.urls.get_urlpatterns() + [
    path("prolific/webhook/", prolific_webhook_view, name="prolific_webhook"),
]
