"""ga5-mailroom-action-gate/v2 endpoint: propose + commit with Ed25519 receipt verification."""
import asyncio, base64, json, logging, re
from collections import defaultdict

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from fastapi import FastAPI, Request
from fastapi.responses import Response

from . import store
from .decide import canonical, decide, sha

logging.basicConfig(level=logging.INFO)
PROFILE = "ga5-mailroom-action-gate/v2"
ACTIONS = ["create_draft", "update_internal_record", "send_approved_notice",
           "request_confirmation", "quarantine_item", "no_action"]
CALL_RE = re.compile(r"^[A-Za-z0-9._:-]{12,128}$")
MAX_BODY = 8 * 1024 * 1024
app = FastAPI(redirect_slashes=False, docs_url=None, redoc_url=None, openapi_url=None)
_locks = defaultdict(asyncio.Lock)


def js(body, status=200):
    return Response(canonical(body), status_code=status, media_type="application/json")


def fail(status, code, msg):
    return js({"profile": PROFILE, "status": "error", "error": {"code": code, "message": msg}}, status)


def b64d(s, url=False):
    s = s.strip()
    s += "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s) if url or "-" in s or "_" in s else base64.b64decode(s, validate=True)


def proposal_digest(p):
    view = {"dossierId": p["dossierId"], "callId": p["callId"], "action": p["action"],
            "target": p.get("target"), "payload": p["payload"], "evidence": sorted(p["evidence"])}
    return sha(canonical(view))


# ---------------- validation (no AI / tool work before this passes) ----------------

def is_str(x, allow_empty=False):
    return isinstance(x, str) and (allow_empty or x.strip() != "")


def validate_propose(b):
    if b.get("profile") != PROFILE or not is_str(b.get("evaluationId")):
        return "profile/evaluationId"
    rv = b.get("receiptVerifier")
    if not isinstance(rv, dict) or rv.get("algorithm") != "Ed25519":
        return "receiptVerifier"
    jwk = rv.get("publicKeyJwk")
    if not isinstance(jwk, dict) or jwk.get("kty") != "OKP" or jwk.get("crv") != "Ed25519" or not is_str(jwk.get("x")):
        return "publicKeyJwk"
    try:
        Ed25519PublicKey.from_public_bytes(b64d(jwk["x"], url=True))
    except Exception:
        return "publicKeyJwk.x"
    if "corpus" in b and not isinstance(b["corpus"], dict):
        return "corpus"
    allowed = b.get("allowedActions", ACTIONS)
    if not isinstance(allowed, list) or not allowed or any(a not in ACTIONS for a in allowed) or len(set(allowed)) != len(allowed):
        return "allowedActions"
    ds = b.get("dossiers")
    if not isinstance(ds, list) or not ds or len(ds) > 500:
        return "dossiers"
    seen = set()
    for d in ds:
        if not isinstance(d, dict) or not is_str(d.get("dossierId")) or d["dossierId"] in seen:
            return "dossierId"
        seen.add(d["dossierId"])
        if d.get("partition") not in ("stable_core", "fresh_audit"):
            return "partition"
        if not is_str(d.get("mailbox")) or not isinstance(d.get("objective"), str) or not isinstance(d.get("receivedAt"), str):
            return "dossier fields"
        srcs = d.get("sources")
        if not isinstance(srcs, list) or not srcs:
            return "sources"
        line_ids = set()
        for s in srcs:
            if not isinstance(s, dict) or not is_str(s.get("sourceId")) or not isinstance(s.get("lines"), list):
                return "source"
            for k in ("kind", "provenance", "title"):
                if k in s and not isinstance(s[k], str):
                    return "source fields"
            for ln in s["lines"]:
                if not isinstance(ln, dict) or not is_str(ln.get("lineId")) or not isinstance(ln.get("text"), str) \
                        or ln["lineId"] in line_ids:
                    return "lines"
                line_ids.add(ln["lineId"])
        if not line_ids:
            return "lines"
    return None


def validate_commit(b):
    if b.get("profile") != PROFILE or not is_str(b.get("evaluationId")) or not is_str(b.get("inputDigest")):
        return "profile/evaluationId/inputDigest"
    rs = b.get("receipts")
    if not isinstance(rs, list) or not rs:
        return "receipts"
    for r in rs:
        if not isinstance(r, dict):
            return "receipt"
        for k in ("dossierId", "callId", "action", "proposalDigest", "receiptId", "receiptSignature"):
            if not is_str(r.get(k)):
                return f"receipt.{k}"
        if not isinstance(r.get("accepted"), bool):
            return "receipt.accepted"
    return None


# ---------------- endpoint ----------------

@app.api_route("/{path:path}", methods=["GET", "HEAD", "POST"])
async def endpoint(path: str, request: Request):
    if request.method != "POST":
        return js({"profile": PROFILE, "status": "ok"})
    raw = await request.body()
    if len(raw) > MAX_BODY:
        return fail(413, "BODY_TOO_LARGE", "Request body too large")
    try:
        body = json.loads(raw)
        assert isinstance(body, dict)
    except Exception:
        return fail(400, "INVALID_JSON", "Body must be a JSON object")
    op = body.get("operation")
    if op == "propose":
        return await propose(body)
    if op == "commit":
        return await commit(body)
    return fail(400, "INVALID_OPERATION", "operation must be propose or commit")


async def propose(b):
    bad = validate_propose(b)
    if bad:
        return fail(422, "INVALID_SCHEMA", f"Invalid propose request: {bad}")
    eid = b["evaluationId"]
    req_hash = sha(canonical(b))
    async with _locks[eid]:
        ev = store.get_eval(eid)
        if ev:
            if ev["propose_hash"] != req_hash:
                return fail(409, "EVALUATION_CONFLICT", "evaluationId was already used with different content")
            return js(ev["propose_response"])
        allowed = b.get("allowedActions", ACTIONS)
        try:
            proposals = await asyncio.wait_for(decide(b["dossiers"], allowed), timeout=50)
        except Exception:
            logging.exception("decision failed")
            return fail(503, "UNAVAILABLE", "Decision step unavailable; retry")
        calls = [p["callId"] for p in proposals]
        assert len(set(calls)) == len(calls) and all(CALL_RE.match(c) for c in calls)
        digest = sha(canonical(b["dossiers"]))
        resp = {"profile": PROFILE, "evaluationId": eid, "status": "awaiting_receipts",
                "inputDigest": digest, "proposals": proposals}
        store.put_eval(eid, req_hash, b["receiptVerifier"], digest, proposals, resp)
        return js(resp)


async def commit(b):
    bad = validate_commit(b)
    if bad:
        return fail(422, "INVALID_SCHEMA", f"Invalid commit request: {bad}")
    eid = b["evaluationId"]
    req_hash = sha(canonical(b))
    async with _locks[eid]:
        ev = store.get_eval(eid)
        if not ev:
            return fail(404, "UNKNOWN_EVALUATION", "Unknown evaluation")
        if ev["state"] == "completed":
            if ev["commit_hash"] == req_hash:
                return js(ev["commit_response"])
            return fail(409, "EVALUATION_CONFLICT", "Evaluation already committed with different content")
        if b["inputDigest"] != ev["input_digest"]:
            return fail(409, "DIGEST_MISMATCH", "inputDigest does not match the proposal")
        props = {p["dossierId"]: p for p in ev["proposals"]}
        rs = b["receipts"]
        if len(rs) != len(props) or {r["dossierId"] for r in rs} != set(props) \
                or len({r["receiptId"] for r in rs}) != len(rs) or len({r["receiptSignature"] for r in rs}) != len(rs):
            return fail(422, "RECEIPT_SET_MISMATCH", "Exactly one unique receipt per proposal is required")
        key = Ed25519PublicKey.from_public_bytes(b64d(ev["verifier"]["publicKeyJwk"]["x"], url=True))
        outcomes = []
        for r in rs:
            p = props[r["dossierId"]]
            if r["callId"] != p["callId"] or r["action"] != p["action"] or r["proposalDigest"] != proposal_digest(p):
                return fail(409, "PROPOSAL_MISMATCH", "Receipt does not match the persisted proposal")
            inner = {k: v for k, v in r.items() if k != "receiptSignature"}
            msg = canonical({"profile": PROFILE, "evaluationId": eid, "inputDigest": b["inputDigest"], "receipt": inner})
            try:
                key.verify(b64d(r["receiptSignature"]), msg.encode("utf-8"))
            except (InvalidSignature, ValueError, Exception):
                return fail(422, "INVALID_RECEIPT_SIGNATURE", "A receipt signature failed verification; nothing executed")
            outcomes.append({"dossierId": p["dossierId"], "callId": p["callId"], "action": p["action"],
                             "proposalDigest": r["proposalDigest"], "receiptId": r["receiptId"],
                             "status": "executed" if r["accepted"] is True else "rejected"})
        resp = {"profile": PROFILE, "evaluationId": eid, "status": "completed",
                "inputDigest": ev["input_digest"], "outcomes": outcomes}
        if not store.complete_eval(eid, req_hash, rs, resp):
            return fail(409, "EVALUATION_CONFLICT", "Evaluation already committed")
        return js(resp)
