from flask import Flask
from flask_cors import CORS

from .config import Config
from .extensions import db, jwt


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # lets the React app (Vite) talk to this server
    CORS(app, origins="*")

    db.init_app(app)
    jwt.init_app(app)
    from . import models  # noqa: F401

    @app.get("/api/health")
    def health():
        return {"status": "ok"}
    from .routes.auth import auth_bp
    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    from .routes.people import people_bp
    app.register_blueprint(people_bp, url_prefix="/api/people")
    from .routes.monitor import monitor_bp
    app.register_blueprint(monitor_bp, url_prefix="/api/monitor")
    return app