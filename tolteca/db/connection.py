#!/usr/bin/env python
# encoding: utf-8

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, scoped_session
# from sqlalchemy.ext.declarative import declarative_base
from tollan.utils.log import get_logger, logit
from tollan.utils.registry import Registry
from tollan.utils import rupdate
from copy import deepcopy
from tollan.utils.fmt import pformat_dict


_sa_connections = Registry.create()


class DatabaseConnection(object):

    logger = get_logger()
    _config = {
            'engine': {
                'echo': True
                },

            'session': {
                'autocommit': True,
                'expire_on_commit': True
                }
            }

    def __init__(self, uri, **kwargs):
        config = deepcopy(self.__class__._config)
        rupdate(config, kwargs)
        with logit(self.logger.debug, f"connect to database {uri}"):
            self._uri = uri
            self.engine = create_engine(
                    self._uri, **config.get('engine', dict()))
            self.metadata = MetaData(bind=self.engine)
            self.Session = scoped_session(sessionmaker(
                    bind=self.engine,
                    **config.get('session', dict())))

    def reflect_tables(self):
        self.metadata.reflect(self.engine)
        self.logger.debug(f"tables {pformat_dict(self.metadata.tables)}")
        return self.metadata.tables

    def clear_tables(self):
        self.metadata.drop_all()
        self.metadata.clear()

    def __repr__(self):
        return f"{self.__class__.__name__}({self._uri.rsplit('//', 1)[-1]})"

    def __new__(cls, uri, *args, **kwargs):
        if uri not in _sa_connections:
            _sa_connections[uri] = super().__new__(cls)
        return _sa_connections[uri]
