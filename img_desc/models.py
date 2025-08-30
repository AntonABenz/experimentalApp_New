from otree.api import (
    models,
    widgets,
    BaseConstants,
    BaseSubsession,
    BaseGroup,
    BasePlayer,
    Currency as c,
    currency_range,
)
import random
from django.forms.models import model_to_dict
from .utils import get_balance, increase_space
from otree.models import Session, Participant
import json
from pprint import pprint
from django.db import models as djmodels
from reading_xls.get_data import get_data
from django.db.models.signals import post_save
from django.dispatch import receiver
from .utils import get_study, STUBURL, get_completion_info

from .utils import get_url_for_image
import logging

logger = logging.getLogger("benzapp.models")

PRODUCER = "P"
INTERPRETER = "I"


author = "Your name here"

doc = """
Your app description
"""


class Constants(BaseConstants):
    name_in_url = "img_desc"
    players_per_group = None
    num_rounds = 85
    STUBURL = STUBURL
    PLACEMENT_ERR = "ERROR_BATCH_PLACEMENT"
    API_ERR = "API_ERROR"
    FALLBACK_URL = STUBURL + PLACEMENT_ERR
    API_ERR_URL = STUBURL + API_ERR
    INTERPRETER = INTERPRETER
    PRODUCER = PRODUCER


class Subsession(BaseSubsession):
    active_batch = models.IntegerField()
    # The following three fields are study-related (on prolific level). The specific participants
    # copy these values to avoid calling prolific api with each new user
    study_id = models.StringField()
    completion_code = models.StringField()
    full_return_url = models.StringField()

    @property
    def get_active_batch(self):
        return self.users.filter(batch=self.active_batch)

    def expand_slots(self):
        study_id = self.study_id
        max_users = self.session.vars.get("max_users", 0)
        batch_size = self.session.vars.get("batch_size", 0)

        if study_id:
            logger.info(
                f"trying to expand slots at study {study_id} for {batch_size} more users (max users {max_users}). oTree session {self.session.code}"
            )
            # just a safeguard to not expand further than a number of otree session slots
            max_users = min(
                self.session.vars.get("max_users", 0), self.session.num_participants
            )
            # there are some safeguards built in increase_space function not to expand further than max users
            # we can't do it here because to get the current number of prolific study slots we need to call api
            # and I want to keep all api calls separate from this code
            pprint(
                increase_space(
                    study_id=study_id, num_extra=batch_size, max_users=max_users
                )
            )
        else:
            logger.warning("No study id data is available! slot expansion failed. params: study_id: {}, max_users: {}, batch_size: {}".format(study_id, max_users, batch_size))

    def check_for_batch_completion(self):
        """
        Here we check if all participants users data marked as completed
        if yes we increase currrent active batch id by 1
        we also increase number of available slots by batch size so new prolific users can join
        """

        s = self
        session = s.session
        active_batch = s.active_batch
        logger.info(
            f"oTree session {session.code}. quick check if batch {active_batch} is completed"
        )
        q = session.batches.filter(batch=s.active_batch, processed=False)
        logger.info(
            f"CURRENT ACTIVE BATCH: {active_batch}; NON PROCESSED SLOTS: {q.count()}"
        )
        if not q.exists():

            session.vars["active_batch"] = active_batch + 1
            logger.info(
                f"oTree session {session.code}. batch {active_batch} is completed. Moving to batch {active_batch + 1}"
            )
            Subsession.objects.filter(session=session).update(
                active_batch=active_batch + 1
            )
            if session.config.get("expand_slots", False):
                logger.info(f' Trying to expand slots for session {session.code}')
                self.expand_slots()

    def creating_session(self):
        self.active_batch = 1
        if self.round_number == 1:
            self.session.vars["active_batch"] = 1
            filename = self.session.config.get("filename")
            excel_data = get_data(filename)
            data = excel_data.get("data")
            self.session.vars["user_data"] = data
            df = data
            self.session.vars["num_rounds"] = df.group_enumeration.max()

            logger.info(f'TOTAL NUM ROUNDS:: {self.session.vars["num_rounds"]}')
            assert (
                df.group_enumeration.max() <= Constants.num_rounds
            ), "PLEASE SET NUMBER OF ROUNDS IN OTREE HIGHER!"
            dbatches = df.to_dict(orient="records")
            raws = [
                dict(
                    session=self.session,
                    batch=i.get("Exp"),
                    item_nr=i.get("Item.Nr"),
                    condition=i.get("Condition"),
                    image=i.get("Item"),
                    round_number=i.get("group_enumeration"),
                    role=i.get("role"),
                    id_in_group=i.get("id"),
                    partner_id=i.get("partner_id"),
                    sentences=i.get("sentences"),
                )
                for i in dbatches
            ]
            raws = [Batch(**i) for i in raws]
            Batch.objects.bulk_create(raws)
             

            # practice settings
            self.session.vars["practice_settings"] = excel_data.get("practice_settings")
            
            self.session.vars["practice_pages"] = excel_data.get("practice_pages")
            self.session.vars["user_settings"] = excel_data.get("settings")
            self.session.vars["s3path"] = excel_data.get("settings").get("s3path")
            self.session.vars["extension"] = excel_data.get("settings").get("extension")
            self.session.vars["prefix"] = excel_data.get("settings").get("prefix")
            self.session.vars["suffixes"] = excel_data.get("settings").get("suffixes")
            allowed_values = excel_data.get("settings").get("allowed_values")
            allowed_values = [
                [item for item in sublist if item != ""] for sublist in allowed_values
            ]
            self.session.vars["allowed_values"] = allowed_values
            self.session.vars['allowed_regex'] = excel_data.get("settings").get("allowed_regex")
            caseflag=excel_data.get("settings").get("caseflag") in ['True', 'true', '1', 't', 'T']
            self.session.vars['caseflag'] =caseflag
            self.session.vars['EndOfIntroText'] = excel_data.get("settings").get("EndOfIntroText", "")
            print('*'*100)
            print(self.session.vars['EndOfIntroText'])
            print('*'*100)

            assert len(self.session.vars.get("suffixes", [])) == len(
                self.session.vars.get("allowed_values", [])
            ), "Number of provided fields should coincide with number of allowed values sets."

            self.session.vars["interpreter_choices"] = excel_data.get("settings").get(
                "interpreter_choices"
            )
            self.session.vars['interpreter_input_type'] = excel_data.get("settings").get(
                "interpreter_input_type"
            )
            self.session.vars['interpreter_input_choices'] = excel_data.get("settings").get(
                "interpreter_input_choices"
            )
            self.session.vars["interpreter_title"] = excel_data.get("settings").get(
                "interpreter_title"
            )
            unique_ids = df.id.unique()
            unique_ids_wz = list(filter(lambda x: x != 0, unique_ids))
            unique_exps = df[df.Exp != 0].Exp.unique()
            batch_size = len(unique_ids_wz)
            max_users = batch_size * len(unique_exps)
            if self.session.config.get("expand_slots"):
                assert (
                    max_users <= self.session.num_participants
                ), f"The number of participants ({self.session.num_participants}) should be higher than the number of users from the spreadsheet ({max_users})!"
            self.session.vars["max_users"] = max_users
            assert batch_size > 0, "Somemthing wrong with the batch size!"
            self.session.vars["batch_size"] = batch_size
            logger.info(f"{max_users=}; {batch_size=}")


class Group(BaseGroup):
    pass


class Batch(djmodels.Model):
    def __str__(self) -> str:
        if self.owner:
            return f"batch: {self.batch}; round: {self.round_number}; belongs to: {self.owner.code}"
        return f"batch: {self.batch}; round: {self.round_number}; doesnt belongs to anyone yet"

    session = djmodels.ForeignKey(
        to=Session,
        on_delete=djmodels.CASCADE,
        related_name="batches",
    )
    owner = djmodels.ForeignKey(
        to=Participant, on_delete=djmodels.CASCADE, related_name="infos", null=True
    )
    sentences = models.LongStringField()
    rewards = models.LongStringField()
    condition = models.StringField()
    item_nr = models.StringField()
    image = models.StringField()
    round_number = models.IntegerField()
    role = models.StringField()
    batch = models.IntegerField()
    id_in_group = models.IntegerField()
    partner_id = models.IntegerField()
    busy = models.BooleanField(initial=False)
    processed = models.BooleanField(initial=False)


class Player(BasePlayer):
    inner_role = models.StringField()
    inner_sentences=models.LongStringField()
    batch = models.IntegerField()
    faulty = models.BooleanField(initial=False)
    feedback=models.LongStringField(
        label=''
    )
    def role(self):
        return self.inner_role

    # BLOCK OF PROLIFIC-RELATED DATA
    prolific_id = models.StringField()
    prol_study_id = models.StringField()
    prol_session_id = models.StringField()
    completion_code = models.StringField()
    full_return_url = models.StringField()
    # END OF BLOCK OF PROLIFIC-RELATED DATA
    vars_dump = models.LongStringField()
    producer_decision = models.LongStringField()
    interpreter_decision = models.LongStringField()
    start_decision_time = djmodels.DateTimeField(null=True)
    end_decision_time = djmodels.DateTimeField(null=True)
    decision_seconds = models.FloatField()

    # link to data from excel sheet
    link = djmodels.ForeignKey(
        to=Batch, on_delete=djmodels.CASCADE, related_name="players", null=True
    )

    def get_sentences_data(self):
        if self.link:
            if self.link.partner_id == 0:
                return json.loads(self.link.sentences)
            else:
                return json.loads(self.get_previous_batch().get("sentences"))

    def get_previous_batch(self):
        if self.inner_role == INTERPRETER:
            l = self.link
            if l.partner_id == 0:
                return dict(sentences='[]')
            obj = self.session.batches.get(
                batch=self.subsession.active_batch - 1,
                role=PRODUCER,
                partner_id=l.id_in_group,
                id_in_group=l.partner_id,
                condition=l.condition,
            )
            return model_to_dict(obj)
        else:
            return dict(sentences='[]')

    def update_batch(self):
        if self.link:
            if self.inner_role == PRODUCER:
                self.link.sentences = self.producer_decision
            if self.inner_role == INTERPRETER:
                self.link.rewards = self.interpreter_decision
            self.link.save()

    def mark_data_processed(self):
        self.participant.vars['full_study_completed'] = True
        Batch.objects.filter(owner=self.participant).update(processed=True)
        self.subsession.check_for_batch_completion()

    def get_full_sentences(self):
        prefix = self.session.vars.get("prefix", "")
        suffixes = self.session.vars.get("suffixes")
        sentences = self.get_sentences_data()
        sentences = [sublist for sublist in sentences if "" not in sublist]
        res = []
        for sentence in sentences:
            expansion_list = [
                str(item) for pair in zip(sentence, suffixes) for item in pair
            ]
            if prefix:
                expansion_list.insert(0, prefix)
            full_sentence = " ".join(expansion_list)
            res.append(full_sentence)

        return res

    def get_image_url(self):
        image = self.link.image
        return get_url_for_image(self, image)

    def start(self):
        """
        This is ran once when a player starts each new round
        """
        if self.round_number == 1:
            # linking batch to participant, marking it busy

            active_batch = self.session.batches.filter(
                batch=self.subsession.active_batch,
            )
            try:
                free_user_id = (
                    active_batch.filter(busy=False, owner__isnull=True)
                    .first()
                    .id_in_group
                )
            except AttributeError as e:
                logger.error(
                    f"no more free slots for participant {self.participant.code}!!!"
                )
                # with all built-in mechanisms, it is unlikely that we will invite someone
                # without a slot for him available, but JIC:
                # in this case we mark him as faulty, and redirect back to prolific with a faulty code
                # so we can still pay him because it's definitely our fault that this would happen.
                self.faulty = True
                return
            free_user = active_batch.filter(busy=False, id_in_group=free_user_id)
            free_user.update(busy=True, owner=self.participant)
        # in each round we get the participant-connected infoset, get the corresponding data for this round and link it
        # to current user.
        self.link = self.participant.infos.get(round_number=self.round_number)
        self.inner_role = self.link.role
        self.inner_sentences=json.dumps(self.get_sentences_data())
        # the following block serves only for dealing with prolific users:
        if self.round_number == 1:
            if self.session.config.get("for_prolific"):
                vars = self.participant.vars
                prol_study_id = vars.get("study_id")
                prol_session_id = vars.get("session_id")
                # we want to set this study id, completion code and return code on subsession level only once
                # for the first person who has this study id.
                # The rest of the group will copy it from there if they have the same study id
                # that way we'll save some calls to prolific api
                ERR_COMPLETION_INFO = dict(
                    completion_code=Constants.API_ERR,
                    full_return_url=Constants.API_ERR_URL,
                )
                if prol_study_id:
                    # if it is not set on subsession level, let's set it:
                    if not self.subsession.study_id:
                        completion_info = get_completion_info(prol_study_id)
                        if completion_info:
                            Subsession.objects.filter(session=self.session).update(
                                study_id=prol_study_id, **completion_info
                            )
                    # if the participation has the same study id as the subsession-level study id,
                    # we just copy it from there
                    if prol_study_id == self.subsession.study_id:
                        completion_info = dict(
                            completion_code=self.subsession.completion_code,
                            full_return_url=self.subsession.full_return_url,
                        )
                    # if a participant has a different study id (unlikely sceneario but what if)
                    # we set his individual completion code
                    else:
                        completion_info = get_completion_info(prol_study_id)
                        # if we fail to get completion info (mostly for API err reasons, we set it still
                        # to the API errr code to deal with the submsission later)
                        if not completion_info:
                            completion_info = ERR_COMPLETION_INFO
                else:
                    completion_info = ERR_COMPLETION_INFO
                for_update = dict(
                    prolific_id=vars.get("prolific_id"),
                    prol_study_id=prol_study_id,
                    prol_session_id=prol_session_id,
                    **completion_info,
                )
                try:
                    if not prol_study_id:
                        raise Exception("Study_id from prolific is not available")
                    if not vars.get("prolific_id"):
                        raise Exception("prolific_id from prolific is not available")
                    if not vars.get("session_id"):
                        raise Exception("session_id from prolific is not available")

                except Exception as E:
                    logger.error("Trouble getting prolific data")
                    logger.error(str(E))

                finally:
                    # whatever we have here we update with all this info the
                    # entire set of players set belonging to the current participant
                    Player.objects.filter(participant=self.participant).update(
                        **for_update
                    )
                    if prol_session_id:
                        self.participant.label = prol_session_id
