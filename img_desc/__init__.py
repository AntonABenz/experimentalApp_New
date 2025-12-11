# img_desc/__init__.py

from otree.api import *
import json
import logging
import os
import re
from sqlalchemy import create_engine, text
from django.db import models as djmodels

logger = logging.getLogger("benzapp.img_desc")

PRODUCER = "P"
INTERPRETER = "I"

STUBURL = "https://app.prolific.co/submissions/complete?cc="

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

class Batch(ExtraModel):
    session_code = models.StringField()
    owner_code = models.StringField(blank=True)
    sentences = models.LongStringField()
    rewards = models.LongStringField(blank=True)
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

# --- SQL Helper ---
_custom_engine = None
def get_engine():
    global _custom_engine
    if _custom_engine: return _custom_engine
    db_url = os.environ.get('DATABASE_URL')
    if not db_url: db_url = "sqlite:///db.sqlite3"
    else:
        if db_url.startswith("postgres://"):
            db_url = db_url.replace("postgres://", "postgresql://", 1)
    _custom_engine = create_engine(db_url)
    return _custom_engine

def get_all_batches_sql(session_code):
    engine = get_engine()
    sql = text("SELECT * FROM img_desc_batch WHERE session_code = :session_code")
    with engine.connect() as conn:
        result = conn.execute(sql, {'session_code': session_code})
        return [dict(zip(result.keys(), row)) for row in result]

def sql_update_batch(batch_id, **kwargs):
    if not kwargs: return
    engine = get_engine()
    set_clauses = []
    params = {'id': batch_id}
    for k, v in kwargs.items():
        set_clauses.append(f"{k} = :{k}")
        params[k] = v
    sql_str = f"UPDATE img_desc_batch SET {', '.join(set_clauses)} WHERE id = :id"
    sql = text(sql_str)
    with engine.connect() as conn:
        conn.execute(sql, params)
        if hasattr(conn, 'commit'): conn.commit()

# --- Helpers ---
def normalize_key(key):
    if not key: return ""
    return re.sub(r'[\s_]+', '_', str(key).lower().strip())

# --- Subsession ---
class Subsession(BaseSubsession):
    active_batch = models.IntegerField()
    study_id = models.StringField()
    completion_code = models.StringField()
    full_return_url = models.StringField()

    @property
    def get_active_batch(self):
        all_data = get_all_batches_sql(self.session.code)
        return [b for b in all_data if b['batch'] == self.active_batch]

    def check_for_batch_completion(self):
        session = self.session
        active_batch = self.active_batch
        all_data = get_all_batches_sql(session.code)
        remaining = [b for b in all_data if b['batch'] == active_batch and not b['processed']]
        if remaining: return
        session.vars["active_batch"] = active_batch + 1
        Subsession.objects.filter(session=session).update(active_batch=active_batch + 1)

class Group(BaseGroup):
    pass

# --- Player ---
class Player(BasePlayer):
    inner_role = models.StringField()
    inner_sentences = models.LongStringField()
    batch = models.IntegerField()
    faulty = models.BooleanField(initial=False)
    feedback = models.LongStringField(label="")
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
    link_id = models.IntegerField(initial=0)

    def role(self): return self.inner_role

    # FIX: Changed to method to prevent oTree interception
    def get_linked_batch(self):
        if not self.link_id: return None
        all_data = get_all_batches_sql(self.session.code)
        for b in all_data:
            if b['id'] == self.link_id: return b
        return None

    def get_sentences_data(self):
        l = self.get_linked_batch()
        if not l: return []
        try:
            if l['partner_id'] == 0: return json.loads(l['sentences'])
            else: return json.loads(self.get_previous_batch().get("sentences"))
        except Exception: return []

    def get_previous_batch(self):
        if self.inner_role != INTERPRETER: return dict(sentences="[]")
        l = self.get_linked_batch()
        if not l or l['partner_id'] == 0: return dict(sentences="[]")
        target_batch_idx = self.subsession.active_batch - 1
        all_data = get_all_batches_sql(self.session.code)
        for obj in all_data:
            if (obj['batch'] == target_batch_idx and obj['role'] == PRODUCER and
                obj['partner_id'] == l['id_in_group'] and obj['id_in_group'] == l['partner_id'] and
                obj['condition'] == l['condition']):
                return obj
        return dict(sentences="[]")

    def update_batch(self):
        if not self.link_id: return
        updates = {}
        if self.inner_role == PRODUCER: updates['sentences'] = self.producer_decision
        if self.inner_role == INTERPRETER: updates['rewards'] = self.interpreter_decision
        if updates: sql_update_batch(self.link_id, **updates)

    def mark_data_processed(self):
        self.participant.vars["full_study_completed"] = True
        all_data = get_all_batches_sql(self.session.code)
        my_code = self.participant.code
        for b in all_data:
            if b['owner_code'] == my_code: sql_update_batch(b['id'], processed=True)
        self.subsession.check_for_batch_completion()

    def get_full_sentences(self):
        prefix = self.session.vars.get("prefix", "")
        suffixes = self.session.vars.get("suffixes") or []
        sentences = self.get_sentences_data() or []
        # Filter valid sentences
        sentences = [sub for sub in sentences if isinstance(sub, list) and "" not in sub]
        res = []
        for sentence in sentences:
            expansion = [str(item) for pair in zip(sentence, suffixes) for item in pair]
            if prefix: expansion.insert(0, prefix)
            res.append(" ".join(expansion))
        return res

    def get_image_url(self):
        l = self.get_linked_batch()
        if not l: return ""
        image_name = l['image']
        if not image_name: return ""
        ext = self.session.vars.get("extension", "png")
        if not image_name.lower().endswith(f".{ext}"): image_name = f"{image_name}.{ext}"
        base = self.session.vars.get("s3path_base", "").rstrip("/")
        if "amazonaws.com" in base: base = base.replace("/practice", "")
        return f"{base}/{image_name}"

    def start(self):
        session = self.session
        subsession = self.subsession
        if self.round_number == 1:
            all_data = get_all_batches_sql(session.code)
            candidates = [b for b in all_data if b['batch'] == subsession.active_batch and not b['busy'] and b['owner_code'] == ""]
            if not candidates:
                self.faulty = True
                return
            candidates.sort(key=lambda b: b['id_in_group'])
            free = candidates[0]
            sql_update_batch(free['id'], busy=True, owner_code=self.participant.code)
        
        all_data = get_all_batches_sql(session.code)
        my_row = None
        for b in all_data:
            if b['owner_code'] == self.participant.code and b['round_number'] == self.round_number:
                my_row = b
                break
        if not my_row:
            self.faulty = True
            return
        self.link_id = my_row['id']
        self.inner_role = my_row['role']
        self.inner_sentences = json.dumps(self.get_sentences_data())
        
        if self.round_number == 1 and session.config.get("for_prolific"):
            p = self.participant
            vars_ = p.vars
            prolific_id = vars_.get("prolific_id") or vars_.get("prolific_pid")
            if vars_.get("prolific_session_id"): p.label = vars_.get("prolific_session_id")

# --- Creating Session ---
def creating_session(subsession: Subsession):
    session = subsession.session
    if subsession.round_number != 1: return
    subsession.active_batch = 1
    session.vars["active_batch"] = 1
    
    filename = session.config.get("filename")
    if not filename: raise RuntimeError("Missing filename")
    
    from reading_xls.get_data import get_data
    excel_data = get_data(filename)
    df = excel_data.get("data")
    session.vars["user_data"] = df
    session.vars["num_rounds"] = int(df.group_enumeration.max())
    
    records = df.to_dict(orient="records")
    for r in records:
        Batch.create(
            session_code=session.code, owner_code="", batch=r.get("Exp"),
            item_nr=r.get("Item.Nr"), condition=r.get("Condition"), image=r.get("Item"),
            round_number=r.get("group_enumeration"), role=r.get("role"),
            id_in_group=r.get("id"), partner_id=r.get("partner_id"), sentences=r.get("sentences"),
        )
    
    settings = excel_data.get("settings") or {}
    # Robust Dictionary Creation
    clean_settings = {}
    for k, v in settings.items():
        clean_settings[normalize_key(k)] = v
    session.vars["user_settings"] = clean_settings
    
    for k in ["s3path_base", "extension", "prefix", "interpreter_choices", "interpreter_title"]:
        session.vars[k] = clean_settings.get(normalize_key(k))

    session.vars["suffixes"] = clean_settings.get("suffixes") or []
    
    # Robust Allowed Values
    allowed_values = []
    i = 1
    while True:
        key = f"allowed_values_{i}"
        val = clean_settings.get(key)
        
        if val: allowed_values.append([x.strip() for x in str(val).split(";") if x.strip()])
        else:
            if i > 5: break
            allowed_values.append([])
        i += 1
    session.vars["allowed_values"] = allowed_values

# --- Pages ---
class FaultyCatcher(Page):
    @staticmethod
    def is_displayed(player): return player.faulty
    def get(self):
        if self.player.faulty: return redirect(Constants.FALLBACK_URL)
        return super().get()

class Q(Page):
    instructions = True
    form_model = "player"

    @staticmethod
    def get_form_fields(player):
        """
        Define which fields are saved on each round.
        We let oTree read POST data for us instead of touching self.request.POST.
        """
        fields = []

        # If you want to keep the timing info:
        fields += ["start_decision_time", "end_decision_time"]

        if player.inner_role == PRODUCER:
            # JSON string from the hidden input
            fields.append("producer_decision")
        elif player.inner_role == INTERPRETER:
            # The selected option (we'll rename the input in the template)
            fields.append("interpreter_decision")

        return fields

    @staticmethod
    def is_displayed(player):
        if not player.faulty:
            player.start()
        return player.round_number <= player.session.vars["num_rounds"]

    @staticmethod
    def vars_for_template(player):
        return dict(
            d=player.get_linked_batch(),
            allowed_values=player.session.vars.get("allowed_values", []),
            suffixes=player.session.vars.get("suffixes", []),
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        # producer_decision / interpreter_decision are already filled by oTree
        # no need to parse request manually
        player.update_batch()
        if player.round_number == player.session.vars["num_rounds"]:
            player.mark_data_processed()

class Feedback(Page):
    form_model = "player"
    form_fields = ["feedback"]
    def is_displayed(self): return self.round_number == self.session.vars["num_rounds"]

class FinalForProlific(Page):
    @staticmethod
    def is_displayed(player):
        return player.session.config.get("for_prolific") and player.round_number == player.session.vars["num_rounds"]
    def get(self):
        if self.player.full_return_url: return redirect(self.player.full_return_url)
        return redirect("https://cnn.com")

page_sequence = [FaultyCatcher, Q, Feedback, FinalForProlific]
