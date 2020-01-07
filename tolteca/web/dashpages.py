#! /usr/bin/env python
import os

# from . import create_server  # noqa: F401
TITLE = "TolTECA"
SECRET_KEY = os.environ.get('TOLTECA_SECRET_KEY', "")

from .backend.config import *  # noqa: F401, F403

# from .backend.db.models import Base as db_Base  # noqa: F401


# def setup_db(server, db):
#     from .backend import config as db_config  # noqa: F401
#     from .backend.db.models import load_models
#     from .backend.db import setup_flask_db

#     server.config.from_object(db_config)
#     db.init_app(server)

#     with server.app_context():
#         load_models(db)

#     @server.before_first_request
#     def setup():
#         setup_flask_db(db.session)


from . import frontend  # noqa: F401
