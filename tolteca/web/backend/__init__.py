#! /usr/bin/env python

from . import config
from . import cache_config
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
from flask_migrate import Migrate
import pandas as pd
import flask
from .db.models import Base
from .db import setup_flask_db
from .db.models import load_models
# from .misc.jwt import setup_jwt
from tolteca.utils.cli.click_log import init as init_log

# extensions
db = SQLAlchemy(model_class=Base, metadata=Base.metadata)
migrate = Migrate()
cache = Cache()


def init_app(server):

    server.config.from_object(config)
    server.config.from_object(cache_config)

    init_log(level='DEBUG' if server.debug else 'INFO')
    db.init_app(server)
    migrate.init_app(server, db)
    cache.init_app(server, config=server.config)

    with server.app_context():
        load_models(db)

    @server.teardown_appcontext
    def shutdown_db_session(exception=None):
        db.session.remove()

    @server.before_first_request
    def setup():
        setup_flask_db(db.session)
        # setup_jwt(server, db.session)
    return server


def create_db_session(bind, server=None):
    if server is None:
        server = flask.current_app
    return db.create_scoped_session(
        options={'bind': db.get_engine(server, bind)})


def dataframe_from_db(bind, query, **kwargs):
    """Return dataframe from database."""
    session = create_db_session(bind)

    return pd.read_sql_query(
            query,
            con=session.bind,
            parse_dates=['Date'],
            )
