from django.urls import path
from otree.urls import urlpatterns
from prolific_webhook import prolific_webhook_view

urlpatterns += [
    path("prolific/webhook/", prolific_webhook_view, name="prolific_webhook"),
]
