"""Keep one database connection open for the app's own SQL instead of a new one each time.

The cohort and Prolific code reads and writes a few small tables with plain SQL
(outside oTree's own database layer). It used to open a new connection for every
lookup: about 11 per question page. On Heroku each new connection costs tens of
milliseconds, and oTree handles one request at a time, so every other participant
waited for this as well.

``connect(url)`` is used exactly like ``psycopg2.connect(url)``: the callers keep
calling ``commit()`` and ``close()``. The difference is that ``close()`` keeps the
connection open for the next caller; anything not committed is rolled back, as a
real close would do. The connection is checked with a tiny query before it is
handed out, and replaced if the database dropped it. If it is already in use
(another thread), the caller gets a normal one-off connection, so nothing ever
waits for it.
"""
from __future__ import annotations

import logging
import threading

import psycopg2
import psycopg2.extensions

logger = logging.getLogger("benzapp.pg_pool")

_lock = threading.Lock()
_conn = None
_url = ""
_in_use = False


def _drop_shared():
    global _conn
    if _conn is not None:
        try:
            _conn.close()
        except Exception:
            pass
    _conn = None


def _alive(conn) -> bool:
    if conn is None or conn.closed:
        return False
    try:
        if conn.get_transaction_status() != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
            conn.rollback()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.rollback()
        return True
    except Exception:
        return False


class _Connection:
    """What the callers get back: the psycopg2 calls they use, and a close() that keeps it open."""

    def __init__(self, conn, shared: bool):
        self._conn = conn
        self._shared = shared
        self._closed = False

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    @property
    def closed(self):
        return 1 if self._closed else self._conn.closed

    def close(self):
        global _in_use
        if self._closed:
            return
        self._closed = True
        if not self._shared:
            try:
                self._conn.close()
            except Exception:
                pass
            return
        with _lock:
            try:
                if self._conn.closed:
                    _drop_shared()
                elif self._conn.get_transaction_status() != psycopg2.extensions.TRANSACTION_STATUS_IDLE:
                    self._conn.rollback()
            except Exception:
                _drop_shared()
            _in_use = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def connect(database_url: str) -> _Connection:
    global _conn, _url, _in_use
    with _lock:
        if not _in_use:
            if _conn is not None and (_url != database_url or not _alive(_conn)):
                _drop_shared()
            if _conn is None:
                _conn = psycopg2.connect(database_url)
                _url = database_url
            _in_use = True
            return _Connection(_conn, shared=True)
    return _Connection(psycopg2.connect(database_url), shared=False)
