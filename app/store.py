"""SQLite persistence for evaluations and cached dossier decisions."""
import json, os, sqlite3, threading

DB_PATH = os.getenv("DB_PATH", "/srv/data/mailroom.db")
os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
_lock = threading.Lock()
_db = sqlite3.connect(DB_PATH, check_same_thread=False, isolation_level=None)
_db.execute("PRAGMA journal_mode=WAL")
_db.execute("PRAGMA synchronous=FULL")
_db.executescript("""
CREATE TABLE IF NOT EXISTS evaluations (
  id TEXT PRIMARY KEY, propose_hash TEXT NOT NULL, state TEXT NOT NULL,
  verifier TEXT NOT NULL, input_digest TEXT NOT NULL, proposals TEXT NOT NULL,
  propose_response TEXT NOT NULL, commit_hash TEXT, commit_response TEXT, receipts TEXT);
CREATE TABLE IF NOT EXISTS decisions (key TEXT PRIMARY KEY, body TEXT NOT NULL);
""")


def get_eval(eid):
    with _lock:
        r = _db.execute("SELECT propose_hash, state, verifier, input_digest, proposals, propose_response, "
                        "commit_hash, commit_response FROM evaluations WHERE id=?", (eid,)).fetchone()
    if not r:
        return None
    return {"propose_hash": r[0], "state": r[1], "verifier": json.loads(r[2]), "input_digest": r[3],
            "proposals": json.loads(r[4]), "propose_response": json.loads(r[5]),
            "commit_hash": r[6], "commit_response": json.loads(r[7]) if r[7] else None}


def put_eval(eid, propose_hash, verifier, input_digest, proposals, response):
    with _lock:
        _db.execute("INSERT INTO evaluations(id, propose_hash, state, verifier, input_digest, proposals, propose_response) "
                    "VALUES(?,?,?,?,?,?,?)", (eid, propose_hash, "awaiting_receipts", json.dumps(verifier),
                                               input_digest, json.dumps(proposals), json.dumps(response)))


def complete_eval(eid, commit_hash, receipts, response):
    with _lock:
        cur = _db.execute("UPDATE evaluations SET state='completed', commit_hash=?, receipts=?, commit_response=? "
                          "WHERE id=? AND state='awaiting_receipts'",
                          (commit_hash, json.dumps(receipts), json.dumps(response), eid))
        return cur.rowcount == 1


def get_decision(key):
    with _lock:
        r = _db.execute("SELECT body FROM decisions WHERE key=?", (key,)).fetchone()
    return json.loads(r[0]) if r else None


def put_decision(key, body):
    with _lock:
        _db.execute("INSERT OR REPLACE INTO decisions(key, body) VALUES(?,?)", (key, json.dumps(body)))
