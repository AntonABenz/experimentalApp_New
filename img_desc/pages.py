from otree.api import Currency as c, currency_range
from ._builtin import WaitPage
from start.pages import Page
from .models import Constants, PRODUCER, INTERPRETER
from django.shortcuts import redirect
import json
from pprint import pprint
import logging
from django.forms.models import model_to_dict

logger = logging.getLogger("benzapp.pages")


class MyPage(Page):
    def is_displayed(self):
        return self.round_number <= self.session.vars["num_rounds"]


class FaultyCatcher(Page):
    def is_displayed(self):
        return self.player.faulty

    def get(self):
        if self.player.faulty:
            return redirect(Constants.FALLBACK_URL)
        return super().get()


class Q(Page):
    instructions = True

    def vars_for_template(self):
        if self.player.link:
            return dict(d=model_to_dict(self.player.link))
        else:
            return dict(d="")

    def is_displayed(self):
        return self.round_number <= self.session.vars["num_rounds"]

    def post(self):
        logger.info(f'POST: {self.request.POST.dict()} by participant {self.player.participant.code}; session {self.session.code}; round {self.round_number}; participant label {self.player.participant.label}')
        time_vars = [
            "start_decision_time",
            "end_decision_time",
        ]
        for t in time_vars:
            v = self.request.POST.get(t)
            if v:
                setattr(self.player, t, v)
        dec_sec = self.request.POST.get("decision_seconds")
        if dec_sec:
            try:
                self.player.decision_seconds = float(dec_sec)
            except Exception as e:
                print(e)
                logger.error("Failed to set duration of decision page")
        if self.player.inner_role == PRODUCER:
            field_name = "producer_decision"
        else:
            field_name = "interpreter_decision"
        raw_decisions = self.request.POST.get(field_name)
        if raw_decisions:
            decisions = json.loads(raw_decisions)
            if self.player.inner_role == PRODUCER:
                flatten_decisions = [list(i.values()) for i in decisions]
                self.player.producer_decision = json.dumps(flatten_decisions)
                self.player.inner_sentences = json.dumps(flatten_decisions)
            if self.player.inner_role == INTERPRETER:
                flatten_decisions = [i.get("choice") for i in decisions]
                self.player.interpreter_decision = json.dumps(flatten_decisions)
        return super().post()

    def before_next_page(self):
        self.player.update_batch()
        logger.info(f'before_next_page. participant {self.player.participant.code}; session {self.session.code}; round {self.round_number}; participant label {self.player.participant.label}')

        if self.round_number == self.session.vars["num_rounds"]:
            logger.info(f'Last round; participant {self.player.participant.code}; session {self.session.code}; round {self.round_number}; participant label {self.player.participant.label}')
            self.player.mark_data_processed()
            try:
                self.player.vars_dump = json.dumps(self.player.participant.vars)
            except Exception as e:
                logger.error("Failed to dump participant vars")
                logger.error(e)


class Feedback(Page):
    form_model = "player"
    form_fields = ["feedback"]

    def is_displayed(self):
        return self.round_number == self.session.vars["num_rounds"]


class FinalForProlific(Page):
    def is_displayed(self):
        return (
            self.session.config.get("for_prolific")
            and self.round_number == self.session.vars["num_rounds"]
        )

    def get(self):
        if self.player.full_return_url:
            return redirect(self.player.full_return_url)
        FALLBACK_URL = "https://cnn.com"
        return redirect(FALLBACK_URL)


page_sequence = [
    FaultyCatcher, Q,
    Feedback, FinalForProlific
]
