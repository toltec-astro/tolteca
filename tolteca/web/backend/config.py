#! /usr/bin/env python


from pathlib import Path
from ...db.config import DB_CONFIG


tolteca_flask_db_filename = 'tolteca_flask_db.sqlite'
tolteca_flask_db_path = Path(__file__).with_name(
        tolteca_flask_db_filename).resolve().as_posix()

SQLALCHEMY_TRACK_MODIFICATIONS = False
SQLALCHEMY_DATABASE_URI = f"sqlite:////{tolteca_flask_db_path}"

SQLALCHEMY_BINDS = {'flask_db': SQLALCHEMY_DATABASE_URI}
SQLALCHEMY_BINDS.update({
    k: v['uri'] for k, v in DB_CONFIG.items()
    })
