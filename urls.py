# urls.py (project root)

from django.urls import path
from otree.urls import get_urlpatterns

from prolific_webhook import prolific_webhook_view

urlpatterns = get_urlpatterns()

# Django paths must NOT start with "/"
urlpatterns += [
    path("prolific/webhook/", prolific_webhook_view),
    path("prolific/webhook", prolific_webhook_view),  
]
