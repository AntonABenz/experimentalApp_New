from otree.api import *
from otree.models import Session, Participant

import json
import logging
from pprint import pprint

from django.db import models as djmodels
from django.forms.models import model_to_dict

from reading_xls.get_data import get_data  # same helper you used before

logger = logging.getLogger("benzapp.models")

PRODUCER = "P"
INTERPRETER = "I"

# Prolific completion URL stub (was in old utils)
STUBURL = "https://app.prolific.co/submissions/complete?cc="


# =====================================================================
# CONSTANTS
# =====================================================================

class Constants(BaseConstants):
    name_in_url = "img_desc"
    players_per_group = None
    # Upper bound; actual number of rounds comes from Excel (group_enumeration.max())
    num_rounds = 85

    STUBURL = STUBURL
    PLACEMENT_ERR = "ERROR_BATCH_PLACEMENT"
    API_ERR = "API_ERROR"
    FALLBACK_URL = STUBURL + PLACEMENT_ERR
    API_ERR_URL = STUBURL + API_ERR
    INTERPRETER = INTERPRETER
    PRODUCER = PRODUCER


# =====================================================================
# SUBSESSION / GROUP
# =====================================================================

class Subsession(BaseSubsession):
    # study/batch info
    active_batch = models.IntegerField()

    # Prolific-study level data (copied to players)
    study_id = models.StringField()
    completion_code = models.StringField()
    full_return_url = models.StringField()

    @property
    def get_active_batch(self):
        return self.users.filter(batch=self.active_batch)

    def expand_slots(self):
        """
        Use Prolific API to increase slots for current study.
        Implemented lazily to avoid circular imports.
        """
        from .utils import increase_space  # lazy import

        study_id = self.study_id
        max_users = self.session.vars.get("max_users", 0)
        batch_size = self.session.vars.get("batch_size", 0)

        if study_id:
            logger.info(
                f"Trying to expand slots at study {study_id} for {batch_size} more users "
                f"(max users {max_users}). oTree session {self.session.code}"
            )
            # safeguard: not more than oTree session size
            max_users = min(max_users, self.session.num_participants)

            pprint(
                increase_space(
                    study_id=study_id, num_extra=batch_size, max_users=max_users
                )
            )
        else:
            logger.warning(
                "No study id data is available! Slot expansion failed. "
                f"params: study_id: {study_id}, max_users: {max_users}, "
                f"batch_size: {batch_size}"
            )

    def check_for_batch_completion(self):
        """
        If all Batch rows of the current batch are processed,
        move to the next batch and (optionally) expand Prolific slots.
        """
        s = self
        session = s.session
        active_batch = s.active_batch
        logger.info(
            f"oTree session {session.code}. Quick check if batch {active_batch} is completed"
        )
        q = session.batches.filter(batch=s.active_batch, processed=False)
        logger.info(
            f"CURRENT ACTIVE BATCH: {active_batch}; NON PROCESSED SLOTS: {q.count()}"
        )
        if not q.exists():
            session.vars["active_batch"] = active_batch + 1
            logger.info(
                f"oTree session {session.code}. Batch {active_batch} is completed. "
                f"Moving to batch {active_batch + 1}"
            )
            Subsession.objects.filter(session=session).update(
                active_batch=active_batch + 1
            )
            if session.config.get("expand_slots", False):
                logger.info(f"Trying to expand slots for session {session.code}")
                self.expand_slots()


class Group(BaseGroup):
    pass


# =====================================================================
# BATCH MODEL (extra Django model)
# =====================================================================

class Batch(djmodels.Model):
    """
    Extra model storing Excel trial data and assignment to participants.
    """

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
        to=Participant,
        on_delete=djmodels.CASCADE,
        related_name="infos",
        null=True,
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


# =====================================================================
# PLAYER
# =====================================================================

class Player(BasePlayer):
    inner_role = models.StringField()
    inner_sentences = models.LongStringField()
    batch = models.IntegerField()
    faulty = models.BooleanField(initial=False)

    feedback = models.LongStringField(label="")

    # PROLIFIC-related data
    prolific_id = models.StringField()
    prol_study_id = models.StringField()
    prol_session_id = models.StringField()
    completion_code = models.StringField()
    full_return_url = models.StringField()

    vars_dump = models.LongStringField()
    producer_decision = models.LongStringField()
    interpreter_decision = models.LongStringField()
    start_decision_time = djmodels.DateTimeField(null=True)
    end_decision_time = djmodels.DateTimeField(null=True)
    decision_seconds = models.FloatField()

    # link back to Batch row
    link = djmodels.ForeignKey(
        to=Batch,
        on_delete=djmodels.CASCADE,
        related_name="players",
        null=True,
    )

    # ----- convenience role property -----

    def role(self):
        return self.inner_role

    # ----- Excel / Batch helpers -----

    def get_sentences_data(self):
        if self.link:
            if self.link.partner_id == 0:
                return json.loads(self.link.sentences)
            else:
                return json.loads(self.get_previous_batch().get("sentences"))
        return []

    def get_previous_batch(self):
        if self.inner_role == INTERPRETER:
            l = self.link
            if l.partner_id == 0:
                return dict(sentences="[]")
            obj = self.session.batches.get(
                batch=self.subsession.active_batch - 1,
                role=PRODUCER,
                partner_id=l.id_in_group,
                id_in_group=l.partner_id,
                condition=l.condition,
            )
            return model_to_dict(obj)
        else:
            return dict(sentences="[]")

    def update_batch(self):
        if not self.link:
            return
        if self.inner_role == PRODUCER:
            self.link.sentences = self.producer_decision
        if self.inner_role == INTERPRETER:
            self.link.rewards = self.interpreter_decision
        self.link.save()

    def mark_data_processed(self):
        self.participant.vars["full_study_completed"] = True
        Batch.objects.filter(owner=self.participant).update(processed=True)
        self.subsession.check_for_batch_completion()

    def get_full_sentences(self):
        prefix = self.session.vars.get("prefix", "")
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []
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
        # lazy import to avoid circular deps
        from .utils import get_url_for_image

        image = self.link.image if self.link else ""
        return get_url_for_image(self, image)

    # ----- main round initialisation -----

    def start(self):
        """
        Called once per round to:
        - link the participant to a free Batch row on round 1
        - set self.link and self.inner_role each round
        - initialise PROLIFIC-related fields
        """

        # round 1: pick a free slot in active batch and mark as busy
        if self.round_number == 1:
            active_batch = self.session.batches.filter(
                batch=self.subsession.active_batch,
            )
            try:
                free_user_id = (
                    active_batch.filter(busy=False, owner__isnull=True)
                    .first()
                    .id_in_group
                )
            except AttributeError:
                logger.error(
                    f"No more free slots for participant {self.participant.code}!!!"
                )
                # mark as faulty → redirect to fallback later
                self.faulty = True
                return

            free_user = active_batch.filter(busy=False, id_in_group=free_user_id)
            free_user.update(busy=True, owner=self.participant)

        # link to Batch row for this round
        try:
            self.link = self.participant.infos.get(round_number=self.round_number)
        except Batch.DoesNotExist:
            logger.error(
                f"Player {self.participant.code} has no Batch for round {self.round_number}"
            )
            self.faulty = True
            return

        self.inner_role = self.link.role
        self.inner_sentences = json.dumps(self.get_sentences_data())

        # Prolific-specific initialisation (only once on round 1)
        if self.round_number == 1 and self.session.config.get("for_prolific"):
            from .utils import get_completion_info  # lazy import

            vars = self.participant.vars
            prol_study_id = vars.get("study_id")
            prol_session_id = vars.get("session_id")

            ERR_COMPLETION_INFO = dict(
                completion_code=Constants.API_ERR,
                full_return_url=Constants.API_ERR_URL,
            )

            if prol_study_id:
                # If not already set on subsession, set it now
                if not self.subsession.study_id:
                    completion_info = get_completion_info(prol_study_id)
                    if completion_info:
                        Subsession.objects.filter(session=self.session).update(
                            study_id=prol_study_id, **completion_info
                        )
                # If participant's study id matches subsession study id → copy
                if prol_study_id == self.subsession.study_id:
                    completion_info = dict(
                        completion_code=self.subsession.completion_code,
                        full_return_url=self.subsession.full_return_url,
                    )
                # Different study id → fetch individually
                else:
                    completion_info = get_completion_info(prol_study_id)
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
                Player.objects.filter(participant=self.participant).update(**for_update)
                if prol_session_id:
                    self.participant.label = prol_session_id


# =====================================================================
# CREATING_SESSION (Excel → Batches + settings)
# =====================================================================

def creating_session(subsession: Subsession):
    """
    Called once when the oTree session is created.
    Loads Excel via reading_xls.get_data and populates Batch table and session.vars.
    """
    subsession.active_batch = 1
    session = subsession.session

    if subsession.round_number != 1:
        return

    session.vars["active_batch"] = 1

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError("Session config must include 'filename' pointing to the Excel file.")

    excel_data = get_data(filename)
    data_df = excel_data.get("data")
    session.vars["user_data"] = data_df

    df = data_df
    session.vars["num_rounds"] = int(df.group_enumeration.max())

    logger.info(f'TOTAL NUM ROUNDS:: {session.vars["num_rounds"]}')
    assert (
        df.group_enumeration.max() <= Constants.num_rounds
    ), "PLEASE SET NUMBER OF ROUNDS IN OTREE HIGHER!"

    dbatches = df.to_dict(orient="records")
    raws = [
        dict(
            session=session,
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
    raws = [Batch(**r) for r in raws]
    Batch.objects.bulk_create(raws)

    # practice & general settings (shared with start app)
    session.vars["practice_settings"] = excel_data.get("practice_settings")
    session.vars["practice_pages"] = excel_data.get("practice_pages")
    session.vars["user_settings"] = excel_data.get("settings")

    settings = excel_data.get("settings") or {}

    session.vars["s3path"] = settings.get("s3path")
    session.vars["extension"] = settings.get("extension")
    session.vars["prefix"] = settings.get("prefix")
    session.vars["suffixes"] = settings.get("suffixes") or []

    allowed_values = settings.get("allowed_values") or []
    allowed_values = [
        [item for item in sublist if item != ""]
        for sublist in allowed_values
    ]
    session.vars["allowed_values"] = allowed_values
    session.vars["allowed_regex"] = settings.get("allowed_regex") or []

    caseflag = settings.get("caseflag") in ["True", "true", "1", "t", "T"]
    session.vars["caseflag"] = caseflag

    session.vars["EndOfIntroText"] = settings.get("EndOfIntroText", "")

    logger.info("*" * 100)
    logger.info(session.vars["EndOfIntroText"])
    logger.info("*" * 100)

    assert len(session.vars.get("suffixes", [])) == len(
        session.vars.get("allowed_values", [])
    ), "Number of provided fields should coincide with number of allowed values sets."

    session.vars["interpreter_choices"] = settings.get("interpreter_choices")
    session.vars["interpreter_input_type"] = settings.get(
        "interpreter_input_type"
    )
    session.vars["interpreter_input_choices"] = settings.get(
        "interpreter_input_choices"
    )
    session.vars["interpreter_title"] = settings.get("interpreter_title")

    unique_ids = df.id.unique()
    unique_ids_wz = list(filter(lambda x: x != 0, unique_ids))
    unique_exps = df[df.Exp != 0].Exp.unique()
    batch_size = len(unique_ids_wz)
    max_users = batch_size * len(unique_exps)

    if session.config.get("expand_slots"):
        assert (
            max_users <= session.num_participants
        ), (
            f"The number of participants ({session.num_participants}) should be higher "
            f"than the number of users from the spreadsheet ({max_users})!"
        )

    session.vars["max_users"] = max_users
    assert batch_size > 0, "Something wrong with the batch size!"
    session.vars["batch_size"] = batch_size

    logger.info(f"{max_users=}; {batch_size=}")


# =====================================================================
# PAGES
# =====================================================================

from ._builtin import Page, WaitPage  # oTree’s generated base classes
from django.shortcuts import redirect


class FaultyCatcher(Page):
    """
    If player has no slot / batch, redirect to fallback URL.
    """

    def is_displayed(self):
        return self.player.faulty

    def get(self):
        if self.player.faulty:
            return redirect(Constants.FALLBACK_URL)
        return super().get()


class Q(Page):
    """
    Main decision page for Producer / Interpreter.
    """

    instructions = True

    def is_displayed(self):
        # Ensure Player.start() has been called before we render anything
        if not self.player.faulty:
            self.player.start()
        # Only display while still within actual number of rounds from Excel
        return self.round_number <= self.session.vars["num_rounds"]

    def vars_for_template(self):
        if self.player.link:
            return dict(d=model_to_dict(self.player.link))
        else:
            return dict(d="")

    def post(self):
        logger.info(
            f'POST: {self.request.POST.dict()} by participant {self.player.participant.code}; '
            f'session {self.session.code}; round {self.round_number}; '
            f'participant label {self.player.participant.label}'
        )
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
                logger.error("Failed to set duration of decision page")
                logger.error(e)

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
        logger.info(
            f'before_next_page. participant {self.player.participant.code}; '
            f'session {self.session.code}; round {self.round_number}; '
            f'participant label {self.player.participant.label}'
        )

        if self.round_number == self.session.vars["num_rounds"]:
            logger.info(
                f'Last round; participant {self.player.participant.code}; '
                f'session {self.session.code}; round {self.round_number}; '
                f'participant label {self.player.participant.label}'
            )
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
        # After last real round only
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
    FaultyCatcher,
    Q,
    Feedback,
    FinalForProlific,
]

