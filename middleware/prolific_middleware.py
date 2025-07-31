"""
Custom middleware to handle Prolific parameters
This REPLACES the functionality from otree_new fork
"""

class ProlificMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Handle InitiateParticipant requests with Prolific parameters
        if 'InitiateParticipant' in request.path:
            # Capture all URL parameters (what otree_new did)
            prolific_params = {
                'PROLIFIC_PID': request.GET.get('PROLIFIC_PID'),
                'STUDY_ID': request.GET.get('STUDY_ID'), 
                'SESSION_ID': request.GET.get('SESSION_ID'),
            }
            
            # Store in session for later retrieval
            request.session.update({k: v for k, v in prolific_params.items() if v})
            
        response = self.get_response(request)
        return response
