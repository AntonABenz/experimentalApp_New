from django.urls import path, include
from prolific_webhook import prolific_webhook

urlpatterns = [
    path("prolific/webhook/", prolific_webhook),
    path("", include("otree.urls")),
]
