# img_desc/__init__.py

from otree.api import *
import json
import logging
from pprint import pprint

from django.db import models as djmodels
from django.forms.models import model_to_dict

from .utils import get_url_for_image, get_completion_info, increase_space
from reading_xls.get_data import get_data  # reads your Excel / sheet

logger = logging.getLogger("benzapp.img_desc")

PRODUCER = "P"
INTERPRETER = "I"

STUBURL = "https://app.prolific.co/submissions/complete?cc="


# =====================================================================
# CONSTANTS
# =====================================================================

class Constants(BaseConstants):
    name_in_url = "img_desc"
    players_per_group = None
    # upper bound; real num rounds computed from Excel (group_enumeration.max)
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
    # index of currently active batch (1,2,...)
    active_batch = models.IntegerField()

    # Prolific-study level data (shared for session)
    study_id = models.StringField()
    completion_code = models.StringField()
    full_return_url = models.StringField()

    @property
    def get_active_batch(self):
        return Batch.filter(
            session_code=self.session.code,
            batch=self.active_batch,
        )


    def expand_slots(self):
        """
        Optional: expand Prolific slots using Prolific API.
        Only used if session.config['expand_slots'] is True.
        """
        study_id = self.study_id
        max_users = self.session.vars.get("max_users", 0)
        batch_size = self.session.vars.get("batch_size", 0)

        if not study_id:
            logger.warning(
                "No study id data is available! slot expansion failed. "
                f"params: study_id: {study_id}, max_users: {max_users}, "
                f"batch_size: {batch_size}"
            )
            return

        logger.info(
            f"Trying to expand slots at study {study_id} for {batch_size} more users "
            f"(max users {max_users}). oTree session {self.session.code}"
        )

        # safeguard: don't go beyond oTree participants
        max_users = min(max_users, self.session.num_participants)

        pprint(
            increase_space(
                study_id=study_id,
                num_extra=batch_size,
                max_users=max_users,
            )
        )

    def check_for_batch_completion(self):
        """
        Check if all Batch rows in the current active batch are marked processed.
        If yes:
          - move to next batch
          - optionally expand Prolific slots
        """
        session = self.session
        active_batch = self.active_batch

        logger.info(
            f"oTree session {session.code}. "
            f"Quick check if batch {active_batch} is completed"
        )

        q = Batch.filter(
            session_code=session.code,
            batch=active_batch,
            processed=False,
        )
        
        logger.info(
            f"CURRENT ACTIVE BATCH: {active_batch}; NON PROCESSED SLOTS: {len(q)}"
        )
        
        if q:
            return

        # batch completed → advance
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
# BATCH MODEL (ExtraModel, oTree-native)
# =====================================================================

# =====================================================================
# BATCH MODEL (ExtraModel, oTree-native)
# =====================================================================

class Batch(ExtraModel):
    """
    Stores trial info and matching for one 'slot' (id_in_group) across rounds.

    We link to oTree objects via codes:
      - session_code: oTree session.code
      - owner_code:   oTree participant.code (or "" if unassigned)
    """

    def __str__(self) -> str:
        if self.owner_code:
            return (
                f"session: {self.session_code}; batch: {self.batch}; "
                f"round: {self.round_number}; belongs to: {self.owner_code}"
            )
        return (
            f"session: {self.session_code}; batch: {self.batch}; "
            f"round: {self.round_number}; unassigned"
        )

    # link to oTree session via CODE, not FK
    session_code = models.StringField()

    # link to participant via CODE, not FK
    owner_code = models.StringField(blank=True)

    # design fields
    sentences = models.LongStringField()          # JSON-encoded list-of-lists
    rewards = models.LongStringField(blank=True)  # JSON-encoded list-of-int / None
    condition = models.StringField()
    item_nr = models.StringField()
    image = models.StringField()
    round_number = models.IntegerField()
    role = models.StringField()   # "P" or "I"
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

    # PROLIFIC fields (copied from participant vars or API)
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

    # store ID of Batch row (ExtraModel) instead of a Django FK
    link_id = models.IntegerField(initial=0)

    # ---- convenience ----
    def role(self):
        return self.inner_role

    @property
    def link(self):
        """Convenience accessor for the Batch row for this round."""
        if not self.link_id:
            return None
        try:
            return Batch.get(id=self.link_id)
        except LookupError:
            return None


    # ---- helpers based on Batch rows ----

    def get_sentences_data(self):
        link = self.link
        if not link:
            return []
        try:
            if link.partner_id == 0:
                return json.loads(link.sentences)
            else:
                return json.loads(self.get_previous_batch().get("sentences"))
        except Exception as e:
            logger.error("Error parsing sentences JSON")
            logger.error(e)
            return []

    def get_previous_batch(self):
        """
        For Interpreter, look into previous batch's producer info.
        """
        if self.inner_role != INTERPRETER:
            return dict(sentences="[]")

        l = self.link
        if not l or l.partner_id == 0:
            return dict(sentences="[]")

        try:
            obj = Batch.get(
                session_code=self.session.code,
                batch=self.subsession.active_batch - 1,
                role=PRODUCER,
                partner_id=l.id_in_group,
                id_in_group=l.partner_id,
                condition=l.condition,
            )
        except LookupError:
            logger.error(
                "Previous batch row not found for session=%s, partner_id=%s, id_in_group=%s",
                self.session.code,
                l.partner_id,
                l.id_in_group,
            )
            return dict(sentences="[]")

        return model_to_dict(obj)


    def update_batch(self):
        link = self.link
        if not link:
            return
        if self.inner_role == PRODUCER:
            link.sentences = self.producer_decision
        if self.inner_role == INTERPRETER:
            link.rewards = self.interpreter_decision
        link.save()

    def mark_data_processed(self):
        """
        Mark all Batch rows for this participant as processed and
        trigger batch-completion check.
        """
        self.participant.vars["full_study_completed"] = True

        for b in Batch.filter(
            session_code=self.session.code,
            owner_code=self.participant.code,):
                
            b.processed = True
            b.save()


        self.subsession.check_for_batch_completion()

    def get_full_sentences(self):
        prefix = self.session.vars.get("prefix", "")
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []
        # remove empty inner lists
        sentences = [sub for sub in sentences if "" not in sub]

        res = []
        for sentence in sentences:
            expansion = [
                str(item) for pair in zip(sentence, suffixes) for item in pair
            ]
            if prefix:
                expansion.insert(0, prefix)
            res.append(" ".join(expansion))
        return res

    def get_image_url(self):
        image = self.link.image if self.link else ""
        return get_url_for_image(self, image)

    # ---- main per-round initialisation ----

    def start(self):
        """
        Run once per round:
          - round 1: assign a free Batch "slot" (id_in_group) in active batch
          - every round: set self.link_id, inner_role, sentences
          - if Prolific: set completion info etc. on round 1
        """
        session = self.session
        subsession = self.subsession

        # --- ROUND 1: assign free slot ---
        if self.round_number == 1:
            # all rows for this session & active batch, currently free
            candidates = Batch.filter(
                session_code=session.code,
                batch=subsession.active_batch,
                busy=False,
                owner_code="",
            )
            if not candidates:
                logger.error(
                    f"No more free slots for participant {self.participant.code} "
                    f"in session {session.code}!"
                )
                self.faulty = True
                return
        
            # pick lowest id_in_group (to mimic order_by("id_in_group").first())
            free = sorted(candidates, key=lambda b: b.id_in_group)[0]
            free.busy = True
            free.owner_code = self.participant.code
            free.save()
        
        # --- EVERY ROUND: link to our Batch row ---
        try:
            row = Batch.get(
                session_code=session.code,
                owner_code=self.participant.code,
                round_number=self.round_number,
            )
        except LookupError:
            logger.error(
                f"Player {self.participant.code} has no Batch row for "
                f"round {self.round_number} in session {session.code}"
            )
            self.faulty = True
            return
        
        self.link_id = row.id
        self.inner_role = row.role
        self.inner_sentences = json.dumps(self.get_sentences_data())
        

        # --- PROLIFIC: only once on round 1, if enabled ---
        if self.round_number == 1 and session.config.get("for_prolific"):
            vars_ = self.participant.vars
            # support both prolific_id and prolific_pid, just in case
            prolific_id = vars_.get("prolific_id") or vars_.get("prolific_pid")
            
            prol_study_id = vars_.get("study_id")
            prol_session_id = vars_.get("prolific_session_id") 

            ERR_COMPLETION_INFO = dict(
                completion_code=Constants.API_ERR,
                full_return_url=Constants.API_ERR_URL,
            )

            if prol_study_id:
                # first participant with this study id sets subsession info
                if not subsession.study_id:
                    completion_info = get_completion_info(prol_study_id)
                    if completion_info:
                        Subsession.objects.filter(session=session).update(
                            study_id=prol_study_id,
                            completion_code=completion_info["completion_code"],
                            full_return_url=completion_info["full_return_url"],
                        )

                # same study id → copy subsession-level info
                if prol_study_id == subsession.study_id:
                    completion_info = dict(
                        completion_code=subsession.completion_code,
                        full_return_url=subsession.full_return_url,
                    )
                else:
                    # different study id → query individually
                    completion_info = get_completion_info(prol_study_id)
                    if not completion_info:
                        completion_info = ERR_COMPLETION_INFO
            else:
                completion_info = ERR_COMPLETION_INFO

            for_update = dict(
                prolific_id=prolific_id,
                prol_study_id=prol_study_id,
                prol_session_id=prol_session_id,
                **completion_info,
            )

            try:
                if not prol_study_id:
                    raise Exception("study_id from Prolific is not available")
                if not prolific_id:
                    raise Exception("prolific_id / prolific_pid from Prolific is not available")
                if not prol_session_id:
                    raise Exception("prolific_session_id from Prolific is not available")
            except Exception as e:
                logger.error("Trouble getting Prolific data")
                logger.error(str(e))
            finally:
                Player.objects.filter(participant=self.participant).update(**for_update)
                if prol_session_id:
                    self.participant.label = prol_session_id


# =====================================================================
# SESSION CREATION: Excel → Batch rows + session.vars
# =====================================================================

def creating_session(subsession: Subsession):
    """
    Called once per session (per round=1).
    Uses reading_xls.get_data(filename) to:
      - build Batch table for this session
      - populate session.vars with settings / practice info
    """
    session = subsession.session

    subsession.active_batch = 1

    if subsession.round_number != 1:
        return

    session.vars["active_batch"] = 1

    filename = session.config.get("filename")
    if not filename:
        raise RuntimeError(
            "Session config must include 'filename' pointing to the Excel / sheet."
        )

    excel_data = get_data(filename)   # <-- your existing reading_xls helper
    df = excel_data.get("data")

    session.vars["user_data"] = df
    session.vars["num_rounds"] = int(df.group_enumeration.max())

    logger.info(f'TOTAL NUM ROUNDS:: {session.vars["num_rounds"]}')

    assert (
        df.group_enumeration.max() <= Constants.num_rounds
    ), "PLEASE SET NUMBER OF ROUNDS IN OTREE HIGHER!"

    # --- create Batch rows in DB (ExtraModel) ---
    records = df.to_dict(orient="records")
    for r in records:
        Batch.create(
            session_code=session.code,
            owner_code="",  # unassigned at start
            batch=r.get("Exp"),
            item_nr=r.get("Item.Nr"),
            condition=r.get("Condition"),
            image=r.get("Item"),
            round_number=r.get("group_enumeration"),
            role=r.get("role"),
            id_in_group=r.get("id"),
            partner_id=r.get("partner_id"),
            sentences=r.get("sentences"),
        )

    # --- practice + general settings (also used by start app) ---
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
        [item for item in sub if item != ""]
        for sub in allowed_values
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
    ), (
        "Number of provided fields should coincide with number of "
        "allowed values sets."
    )

    session.vars["interpreter_choices"] = settings.get("interpreter_choices")
    session.vars["interpreter_input_type"] = settings.get(
        "interpreter_input_type"
    )
    session.vars["interpreter_input_choices"] = settings.get(
        "interpreter_input_choices"
    )
    session.vars["interpreter_title"] = settings.get("interpreter_title")

    # --- capacity checks & batch sizes ---
    unique_ids = df.id.unique()
    unique_ids_wz = [x for x in unique_ids if x != 0]
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
from django.shortcuts import redirect


class FaultyCatcher(Page):
    """
    If player couldn't get a slot (no free Batch row), redirect to fallback URL.
    """

    @staticmethod
    def is_displayed(player):
        return player.faulty

    def get(self):
        # 'get' is a standard view method, so 'self' is the Page here.
        # self.player works correctly in this context.
        if self.player.faulty:
            return redirect(Constants.FALLBACK_URL)
        return super().get()

class Q(Page):
    """
    Main decision page for Producer / Interpreter.
    """

    instructions = True

    @staticmethod
    def is_displayed(player):
        # ensure Player.start() runs before we show anything
        if not player.faulty:
            player.start()
        
        # Access session via the player object
        return player.round_number <= player.session.vars["num_rounds"]

    def vars_for_template(self):
        if self.player.link:
            return dict(d=model_to_dict(self.player.link))
        return dict(d="")

    def post(self):
        logger.info(
            f'POST: {self.request.POST.dict()} by participant {self.player.participant.code}; '
            f'session {self.session.code}; round {self.round_number}; '
            f'participant label {self.player.participant.label}'
        )

        # timings
        for t in ["start_decision_time", "end_decision_time"]:
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

        # save decisions
        field_name = (
            "producer_decision"
            if self.player.inner_role == PRODUCER
            else "interpreter_decision"
        )

        raw_decisions = self.request.POST.get(field_name)
        if raw_decisions:
            decisions = json.loads(raw_decisions)
            if self.player.inner_role == PRODUCER:
                flatten = [list(i.values()) for i in decisions]
                self.player.producer_decision = json.dumps(flatten)
                self.player.inner_sentences = json.dumps(flatten)
            else:
                flatten = [i.get("choice") for i in decisions]
                self.player.interpreter_decision = json.dumps(flatten)

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
        # fallback if Prolific URL missing
        return redirect("https://cnn.com")


page_sequence = [
    FaultyCatcher,
    Q,
    Feedback,
    FinalForProlific,
]
