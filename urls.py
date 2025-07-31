from django.urls import path, include
from django.views.generic import RedirectView

# Custom Prolific URL handling (alternative to middleware)
urlpatterns = [
    # Standard oTree URLs
    path('', include('otree.urls')),
    
    # Custom Prolific endpoint (optional)
    path('p/<str:participant_code>/', RedirectView.as_view(
        url='/InitiateParticipant/%(participant_code)s?%(query_string)s'
    ), name='prolific_redirect'),
]
