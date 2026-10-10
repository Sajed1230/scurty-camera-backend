from datetime import datetime
from .extensions import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="operator")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class Person(db.Model):
    __tablename__ = "people"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    national_id = db.Column(db.String(16), unique=True, nullable=False)
    status = db.Column(db.String(20), nullable=False)
    case_history = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    embeddings = db.relationship(
        "FaceEmbedding", backref="person", cascade="all, delete-orphan", passive_deletes=True
    )


class FaceEmbedding(db.Model):
    __tablename__ = "face_embeddings"

    id = db.Column(db.Integer, primary_key=True)
    person_id = db.Column(db.Integer, db.ForeignKey("people.id", ondelete="CASCADE"), nullable=False)
    vector = db.Column(db.LargeBinary, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class AuditLog(db.Model):
    __tablename__ = "audit_log"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"))
    action = db.Column(db.String(100), nullable=False)
    detail = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)