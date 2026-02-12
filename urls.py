from django.urls import path
from prolific_webhook import prolific_webhook

urlpatterns = [
    path("prolific/webhook/", prolific_webhook),
]
