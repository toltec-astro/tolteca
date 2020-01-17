#! /usr/bin/env python

"""DashA site for TolTECA."""

from ..db.config import DB_CONFIG
from pathlib import Path

tolteca_flask_db_filename = 'tolteca_flask_db.sqlite'
tolteca_flask_db_path = Path(__file__).with_name(
        tolteca_flask_db_filename).resolve().as_posix()

db_config = {
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:////{tolteca_flask_db_path}",
        "SQLALCHEMY_BINDS": {
            k: v['uri'] for k, v in DB_CONFIG.items()
            }
        }
db_config['SQLALCHEMY_BINDS'].update({
    'flask_db': db_config['SQLALCHEMY_DATABASE_URI'],
    })

cache_config = {
        "CACHE_TYPE": 'redis',
        "CACHE_DEFAULT_TIMEOUT": 60 * 5,
        "CACHE_KEY_PREFIX": 'tolteca_',
        "CACHE_REDIS_URL": 'redis://localhost:6379/0',
        }

app_title = 'TolTECA'

# site runtime
extensions = [
    {
        'module': 'dasha.web.extensions.db',
        'config': db_config,
        },
    {
        'module': 'dasha.web.extensions.cache',
        'config': cache_config,
        },
    {
        'module': 'dasha.web.extensions.dasha',
        'config': {
            "TITLE": app_title,
            "template": "slapdash",
            "pages": [],
            },
        },
    ]

# from .defaults import create_server  # noqa: F401
