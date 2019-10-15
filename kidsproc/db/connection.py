#!/usr/bin/env python
# encoding: utf-8

from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.event import listen
# from sqlalchemy.pool import Pool
from hashlib import md5
from dogpile.cache.region import make_region
from .utils import caching_query
from astropy import log


class DatabaseConnection(object):

    def __init__(self, label, uri, **kwargs):
        self._label = label
        self._uri = uri
        engine_kwargs = dict(
                echo=True,
                )
        self.engine = create_engine(self.uri, **engine_kwargs)
        self.metadata = MetaData(bind=self.engine)
        self.Base = declarative_base(bind=self.engine)
        self.Session = scoped_session(
                    sessionmaker(
                        bind=self.engine, autocommit=True,
                        query_cls=caching_query.query_callable(regions),
                        expire_on_commit=expire_on_commit))
            # ------------------------------------------------

    @property
    def label(self):
        return self._label

    @property
    def uri(self):
        return self._uri

    def __new__(cls, database_connection_string=None, expire_on_commit=True):
        """This overrides the object's usual creation mechanism."""

        if cls not in cls._singletons:
            assert database_connection_string is not None,\
                    "A database connection string must be specified!"
            cls._singletons[cls] = object.__new__(cls)

            # ------------------------------------------------
            # This is the custom initialization
            # ------------------------------------------------
            me = cls._singletons[cls]  # just for convenience (think "self")

            me.database_connection_string = database_connection_string

            # change 'echo' to print each SQL query
            # (for debugging/optimizing/the curious)
            me.engine = create_engine(
                    me.database_connection_string,
                    echo=True)
            # pool_size=10, pool_recycle=1800)

            me.metadata = MetaData()
            me.metadata.bind = me.engine
            me.Base = declarative_base(bind=me.engine)
            me.Session = scoped_session(
                    sessionmaker(
                        bind=me.engine, autocommit=True,
                        query_cls=caching_query.query_callable(regions),
                        expire_on_commit=expire_on_commit))
            # ------------------------------------------------

        return cls._singletons[cls]


def connect_database(uri):
    log.debug(f"connect to database at {uri}")
    db = DatabaseConnection(database_connection_string=uri)
    log.debug(f"database engine {db.engine}")
    return db
