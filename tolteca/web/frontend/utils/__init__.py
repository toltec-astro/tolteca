import inspect
from urllib.parse import parse_qs

import dash
from flask import current_app as server
from werkzeug.datastructures import MultiDict

from collections import OrderedDict


def get_query_params(search):
    return MultiDict(parse_qs(search.lstrip("?")))


def odict_from_list(l, key):
    return OrderedDict([(key(v), v) for v in l])


def get_dash_args_from_flask_config(config):
    """Get a dict of Dash params that were specified """
    # all arg names less 'self'
    dash_args = set(inspect.getfullargspec(dash.Dash.__init__).args[1:])
    return {key.lower(): val for key, val in config.items()
            if key.lower() in dash_args}


def get_url(path):
    """Expands an internal URL to include prefix the app is mounted at"""
    return f"{server.config['ROUTES_PATHNAME_PREFIX']}{path}"
