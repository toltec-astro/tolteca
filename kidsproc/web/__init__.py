from flask import Flask
from dash import Dash

from .utils import get_dash_args_from_flask_config

ENV_SETTINGS = 'W_KIDSPROC_SETTINGS'


def create_flask(config=f"{__package__}.settings"):
    server = Flask(__package__)
    server.config.from_object(config)
    server.config.from_envvar(ENV_SETTINGS, silent=True)
    return server


def create_dash(server):
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

    return app
