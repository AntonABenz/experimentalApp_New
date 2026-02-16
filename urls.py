from django.urls import path
from prolific_webhook import prolific_webhook_view

urlpatterns = [
    # Prolific will call this URL
    path("prolific/webhook/", prolific_webhook_view, name="prolific_webhook"),
]
