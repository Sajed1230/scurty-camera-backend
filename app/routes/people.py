import secrets
from functools import wraps

import numpy as np

from flask import Blueprint, request, jsonify
from flask_jwt_extended import verify_jwt_in_request, get_jwt, get_jwt_identity
from sqlalchemy import or_

from ..extensions import db
from ..models import Person, FaceEmbedding, AuditLog
from ..face_ai import get_embedding

people_bp = Blueprint("people", __name__)

STATUSES = ["Wanted", "Criminal", "Hacker", "Cleared"]

# similarity above this with the SAME person = the same photo again
SAME_PHOTO = 0.95
# similarity above this with ANOTHER person = the same face under a second name
SAME_FACE = 0.5


def admin_only(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        verify_jwt_in_request()
        if get_jwt().get("role") != "admin":
            return jsonify({"error": "Admins only"}), 403
        return fn(*args, **kwargs)
    return wrapper


def person_to_dict(p):
    return {
        "id": p.id,
        "name": p.name,
        "national_id": p.national_id,
        "status": p.status,
        "case_history": p.case_history or "",
        "photos": len(p.embeddings),
        "created_at": p.created_at.strftime("%d %b %Y") if hasattr(p.created_at, "strftime") else str(p.created_at or ""),
    }


def check_data(data):
    name = (data.get("name") or "").strip()
    national_id = (data.get("national_id") or "").strip()

    if not name:
        return "Name is required"
    if national_id and (not national_id.isdigit() or len(national_id) != 16):
        return "National ID must be 16 digits, or leave it empty"
    if data.get("status") not in STATUSES:
        return "Status is not valid"
    return None


def make_national_id():
    # fake 16-digit ID for testing, starts with 9 so it is easy to spot
    while True:
        new_id = "9" + "".join(str(secrets.randbelow(10)) for _ in range(15))
        if not Person.query.filter_by(national_id=new_id).first():
            return new_id


def log(action, detail):
    db.session.add(AuditLog(user_id=int(get_jwt_identity()), action=action, detail=detail))


@people_bp.get("")
@admin_only
def list_people():
    query = Person.query

    q = (request.args.get("q") or "").strip()
    if q:
        like = f"%{q}%"
        query = query.filter(or_(Person.name.like(like), Person.national_id.like(like)))

    status = request.args.get("status")
    if status in STATUSES:
        query = query.filter_by(status=status)

    people = query.order_by(Person.created_at.desc()).all()
    return jsonify({"people": [person_to_dict(p) for p in people]})


@people_bp.post("")
@admin_only
def add_person():
    data = request.get_json(silent=True) or {}

    error = check_data(data)
    if error:
        return jsonify({"error": error}), 400

    national_id = (data.get("national_id") or "").strip()
    if not national_id:
        national_id = make_national_id()
    elif Person.query.filter_by(national_id=national_id).first():
        return jsonify({"error": "This National ID is already registered"}), 409

    person = Person(
        name=data["name"].strip(),
        national_id=national_id,
        status=data["status"],
        case_history=data.get("case_history", ""),
    )
    db.session.add(person)
    log("added person", person.name)
    db.session.commit()

    return jsonify(person_to_dict(person)), 201


@people_bp.put("/<int:person_id>")
@admin_only
def update_person(person_id):
    person = db.session.get(Person, person_id)
    if not person:
        return jsonify({"error": "Person not found"}), 404

    data = request.get_json(silent=True) or {}

    error = check_data(data)
    if error:
        return jsonify({"error": error}), 400

    national_id = (data.get("national_id") or "").strip() or person.national_id
    other = Person.query.filter_by(national_id=national_id).first()
    if other and other.id != person.id:
        return jsonify({"error": "This National ID is already registered"}), 409

    person.name = data["name"].strip()
    person.national_id = national_id
    person.status = data["status"]
    person.case_history = data.get("case_history", "")
    log("edited person", person.name)
    db.session.commit()

    return jsonify(person_to_dict(person))


@people_bp.delete("/<int:person_id>")
@admin_only
def delete_person(person_id):
    person = db.session.get(Person, person_id)
    if not person:
        return jsonify({"error": "Person not found"}), 404

    log("deleted person", person.name)
    db.session.delete(person)
    db.session.commit()

    return jsonify({"deleted": True})


@people_bp.post("/<int:person_id>/photos")
@admin_only
def add_photos(person_id):
    person = db.session.get(Person, person_id)
    if not person:
        return jsonify({"error": "Person not found"}), 404

    files = request.files.getlist("photos")[:5]
    if not files:
        return jsonify({"error": "No photos were sent"}), 400

    results = []
    saved = 0
    # every saved signature in the system, and who owns it
    rows = FaceEmbedding.query.all()
    known = [np.frombuffer(r.vector, dtype=np.float32) for r in rows]
    owners = [r.person_id for r in rows]

    for f in files:
        vector, error = get_embedding(f.read())
        if error:
            results.append({"file": f.filename, "ok": False, "error": error})
            continue

        if known:
            scores = np.vstack(known) @ vector
            best = int(np.argmax(scores))
            best_score = float(scores[best])
            owner_id = owners[best]

            if owner_id == person.id and best_score > SAME_PHOTO:
                results.append({"file": f.filename, "ok": False, "error": "same photo already added"})
                continue

            if owner_id != person.id and best_score >= SAME_FACE:
                other = db.session.get(Person, owner_id)
                results.append({
                    "file": f.filename,
                    "ok": False,
                    "error": f"this face is already registered as {other.name}",
                })
                continue

        known.append(vector)
        owners.append(person.id)

        db.session.add(FaceEmbedding(person_id=person.id, vector=vector.tobytes()))
        saved += 1
        results.append({"file": f.filename, "ok": True})

    if saved:
        log("added photos", f"{person.name}: {saved}")
    db.session.commit()

    total = FaceEmbedding.query.filter_by(person_id=person.id).count()
    return jsonify({"saved": saved, "results": results, "photos": total})