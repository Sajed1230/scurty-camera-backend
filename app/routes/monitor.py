import os
import tempfile
import time

import cv2
import numpy as np
from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity

from ..extensions import db
from ..models import Person, FaceEmbedding, AuditLog
from ..face_ai import get_faces

monitor_bp = Blueprint("monitor", __name__)

SAMPLES_PER_SECOND = 3
MAX_SECONDS = 60


def load_registry():
    """All saved signatures, who owns each one, and the people details."""
    rows = FaceEmbedding.query.all()
    if not rows:
        return np.empty((0, 512), dtype=np.float32), [], {}

    vectors = np.vstack([np.frombuffer(r.vector, dtype=np.float32) for r in rows])
    owners = [r.person_id for r in rows]
    people = {
        p.id: {
            "id": p.id,
            "name": p.name,
            "national_id": p.national_id,
            "status": p.status,
            "case_history": p.case_history or "",
        }
        for p in Person.query.all()
    }
    return vectors, owners, people


def best_match(vectors, owners, people, vector):
    """The most similar saved person and the score. No threshold here."""
    if len(owners) == 0:
        return None, 0.0
    scores = vectors @ vector
    best = int(np.argmax(scores))
    return people.get(owners[best]), float(scores[best])


# the registry is kept in memory for a few seconds, so live frames are fast
_cache = {"time": 0.0, "data": None}


def get_registry():
    if _cache["data"] is None or time.time() - _cache["time"] > 5:
        _cache["data"] = load_registry()
        _cache["time"] = time.time()
    return _cache["data"]


def find_track(tracks, used, box, t):
    """Is this face the same one we saw a moment ago?"""
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    size = box[2] - box[0]
    best, best_dist = None, None

    for tr in tracks:
        if tr["id"] in used or t - tr["t"] > 1.0:
            continue
        dist = ((cx - tr["cx"]) ** 2 + (cy - tr["cy"]) ** 2) ** 0.5
        if dist < max(size, tr["size"]) and (best is None or dist < best_dist):
            best, best_dist = tr, dist
    return best


def norm(value, size):
    return round(min(max(value / size, 0), 1), 4)


@monitor_bp.post("/analyze")
@jwt_required()
def analyze():
    video = request.files.get("video")
    if not video:
        return jsonify({"error": "No video was sent"}), 400

    suffix = os.path.splitext(video.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        video.save(tmp)
        path = tmp.name

    cap = cv2.VideoCapture(path)
    try:
        if not cap.isOpened():
            return jsonify({"error": "Cannot read this video"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        step = max(1, round(fps / SAMPLES_PER_SECOND))
        vectors, owners, people = load_registry()

        tracks, samples, next_id, index = [], [], 1, 0

        while True:
            t = index / fps
            if t > MAX_SECONDS:
                break

            if index % step == 0:
                ok, frame = cap.read()
            else:
                ok, frame = cap.grab(), None
            if not ok:
                break

            if frame is not None:
                # smaller frames are faster to analyze
                if frame.shape[1] > 960:
                    scale = 960 / frame.shape[1]
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                h, w = frame.shape[:2]

                faces, used = [], set()
                for f in get_faces(frame):
                    box = f["box"]

                    track = find_track(tracks, used, box, t)
                    if track is None:
                        track = {"id": next_id}
                        next_id += 1
                        tracks.append(track)
                    track.update(
                        cx=(box[0] + box[2]) / 2,
                        cy=(box[1] + box[3]) / 2,
                        size=box[2] - box[0],
                        t=t,
                    )
                    used.add(track["id"])

                    person, score = best_match(vectors, owners, people, f["vector"])

                    faces.append({
                        "track": track["id"],
                        "box": [norm(box[0], w), norm(box[1], h), norm(box[2], w), norm(box[3], h)],
                        "score": round(score, 3),
                        "person": person,
                    })

                samples.append({"t": round(t, 2), "faces": faces})

            index += 1

        db.session.add(AuditLog(
            user_id=int(get_jwt_identity()),
            action="analyzed video",
            detail=video.filename or "",
        ))
        db.session.commit()

        return jsonify({"samples": samples})
    finally:
        cap.release()
        os.remove(path)


@monitor_bp.post("/decision")
@jwt_required()
def decision():
    data = request.get_json(silent=True) or {}
    kind = data.get("decision")
    person_id = data.get("person_id")

    person = db.session.get(Person, person_id) if person_id else None
    if not person or kind not in ("confirmed", "false_match"):
        return jsonify({"error": "Invalid decision"}), 400

    db.session.add(AuditLog(
        user_id=int(get_jwt_identity()),
        action="match " + kind.replace("_", " "),
        detail=f"{person.name} (score {data.get('score')})",
    ))
    db.session.commit()

    return jsonify({"saved": True})


@monitor_bp.post("/frame")
@jwt_required()
def frame():
    """One picture from the live camera. Nothing is saved."""
    upload = request.files.get("frame")
    if not upload:
        return jsonify({"error": "No frame was sent"}), 400

    img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Not a valid image"}), 400

    h, w = img.shape[:2]
    vectors, owners, people = get_registry()

    faces = []
    for f in get_faces(img):
        box = f["box"]
        person, score = best_match(vectors, owners, people, f["vector"])
        faces.append({
            "box": [norm(box[0], w), norm(box[1], h), norm(box[2], w), norm(box[3], h)],
            "score": round(score, 3),
            "person": person,
        })

    return jsonify({"faces": faces})