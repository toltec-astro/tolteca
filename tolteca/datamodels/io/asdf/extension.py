#! /usr/bin/env python

from pathlib import Path
from asdf import util
from asdf.extension import BuiltinExtension

# from .tags import *  # noqa: F403, F401
from .types import _tolteca_types, ToltecaType


SCHEMA_PATH = Path(__file__).with_name('schemas').resolve()
ORGANIZATION_URL_BASE = ToltecaType.organization
STANDARD = ToltecaType.standard
SCHEMA_URL_BASE = f'{ORGANIZATION_URL_BASE}/schemas/{STANDARD}'


class ToltecaExtension(BuiltinExtension):

    @property
    def types(self):
        return _tolteca_types

    @property
    def tag_mapping(self):
        return [(
            f'tag:{ORGANIZATION_URL_BASE}:{STANDARD}',
            f'http://{SCHEMA_URL_BASE}{{tag_suffix}}')]

    @property
    def url_mapping(self):
        result = [(
            f'http://{SCHEMA_URL_BASE}',
            util.filepath_to_url(
                SCHEMA_PATH.joinpath(ORGANIZATION_URL_BASE).as_posix()) +
            f"/{STANDARD}{{url_suffix}}.yaml")]
        return result
