from getpass import getpass
from werkzeug.security import generate_password_hash

from app import create_app
from app.extensions import db
from app.models import User

app = create_app()

with app.app_context():
    username = input("Admin username: ").strip()
    password = getpass("Admin password (min 8 characters): ")

    if len(password) < 8:
        print("Password is too short.")
        raise SystemExit

    if User.query.filter_by(username=username).first():
        print("That username already exists.")
        raise SystemExit

    admin = User(
        username=username,
        password_hash=generate_password_hash(password),
        role="admin",
    )
    db.session.add(admin)
    db.session.commit()
    print("Admin created.")