#! /usr/bin/env python


from pathlib import Path
from ...db.config import DB_CONFIG


TOLTECA_FLASK_DB_FILENAME = 'tolteca_flask_db.sqlite'
TOLTECA_FLASK_DB_PATH = Path(__file__).with_name(
        TOLTECA_FLASK_DB_FILENAME).resolve().as_posix()

SQLALCHEMY_TRACK_MODIFICATIONS = False

SQLALCHEMY_DATABASE_URI = f"sqlite:////{TOLTECA_FLASK_DB_PATH}"

SQLALCHEMY_BINDS = {k: v['uri'] for k, v in DB_CONFIG.items()}
