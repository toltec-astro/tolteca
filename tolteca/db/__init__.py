#!/usr/bin/env python
# encoding: utf-8


import inspect
import importlib
from astropy import log
from ..utils import get_pkg_data_path
from .DatabaseConnection import connect_database

DB_CONFIG = {

        # 'debug_a': {

        #     'uri': 'mysql+mysqldb://debug:debug@localhost/debug'
        # },
        'debug_b': {
            'uri': f'sqlite+pysqlite:///'
                   f'{get_pkg_data_path().joinpath("debug.sqlite")}',
            'schema': 'toltecdatadb',
        }
    }


class DatabaseRuntime(object):
    '''Class to hold database related states.'''

    def __init__(self, name, uri, schema):
        self.name = name
        self.uri = uri
        self.schema = schema
        self.connection = None
        self.session = None
        self.models = {}
        self.ok = False
        self.error = []
        self.initialize()

    def initialize(self):
        try:
            self.connection = connect_database(self.uri)
        except Exception as e:
            log.error(
                    f"unable to connect to {self.name} {self.uri}: {e}")
        else:
            self._init_models()
            self._init_session()
            self._test_db_connection()

    def _init_models(self):
        log.debug("initialize database models")

        try:
            m = importlib.import_module(f".tables.{self.schema}", __name__)
            m.create_tables(self.connection)
        except Exception as e:
            log.warning(f"unable to load {self.schema} tables: {e}")
        else:
            try:
                m = importlib.import_module(f".models.{self.schema}", __name__)
            except Exception as e:
                log.warning(f"unable to load {self.schema} models: {e}")
            else:
                self.models[self.schema] = m
        log.debug(f"database models: {self.models}")

    def _init_session(self):
        self.session = self.connection.Session()

    def _test_db_connection(self):
        if self.connection and self.models:
            try:
                # self.session.query(
                #     self.models["toltecdatadb"].version).first()
                connection = self.session.connection()
            except Exception as e:
                self.ok = False
                # self.error.append(f'error connecting to database: {e}')
                log.error(f'error querying database: {e}')
            else:
                self.ok = True
        else:
            self.ok = False

    def modelclasses(self, name="default", lower=None):
        if name not in self.models:
            return dict()
        module = self.models[name]

        classes = dict()
        for model in inspect.getmembers(module, inspect.isclass):
            keyname = model[0].lower() if lower else model[0]
            if hasattr(model[1], '__tablename__'):
                classes[keyname] = model[1]
        return classes


def get_databases(config=DB_CONFIG):
    result = dict()
    for k, v in config.items():
        result[k] = DatabaseRuntime(k, **v)
    return result
