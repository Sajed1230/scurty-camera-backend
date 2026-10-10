from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from werkzeug.security import check_password_hash

from ..extensions import db
from ..models import User, AuditLog

auth_bp = Blueprint("auth", __name__)


@auth_bp.post("/login")
def login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    user = User.query.filter_by(username=username).first()

    if not user or not check_password_hash(user.password_hash, password):
        db.session.add(AuditLog(action="failed login", detail=f"username: {username}"))
        db.session.commit()
        # same message for both cases, so nobody can guess which usernames exist
        return jsonify({"error": "Wrong username or password"}), 401

    token = create_access_token(identity=str(user.id), additional_claims={"role": user.role})

    db.session.add(AuditLog(user_id=user.id, action="login"))
    db.session.commit()

    return jsonify({
        "token": token,
        "user": {"id": user.id, "username": user.username, "role": user.role},
    })


@auth_bp.get("/me")
@jwt_required()
def me():
    user = db.session.get(User, int(get_jwt_identity()))
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"id": user.id, "username": user.username, "role": user.role})