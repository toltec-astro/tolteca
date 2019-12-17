#! /usr/bin/env python

from dash import Dash
from .utils import get_dash_args_from_flask_config
from tolteca.utils.log import timeit
import flask


@timeit
def init_app(server):
    app = Dash(
        name=__package__,
        server=server,
        suppress_callback_exceptions=True,
        **get_dash_args_from_flask_config(server.config),
    )
    # Update the Flask config a default "TITLE" and then with any new Dash
    # configuration parameters that might have been updated so that we can
    # access Dash config easily from anywhere in the project with Flask's
    # 'current_app'
    server.config.setdefault("TITLE", "Dash")
    server.config.update({key.upper(): val for key, val in app.config.items()})

    app.title = server.config["TITLE"]

    if "SERVE_LOCALLY" in server.config:
        app.scripts.config.serve_locally = server.config["SERVE_LOCALLY"]
        app.css.config.serve_locally = server.config["SERVE_LOCALLY"]

    with server.app_context():
        server.dash_app = app
        from . import index  # noqa: F401
        # app.layout = main_layout_sidebar()
        app.layout = index.layout

    return server


def get_current_dash_app():
    return flask.current_app.dash_app
