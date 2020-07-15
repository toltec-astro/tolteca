#!/usr/bin/env python

from dasha.web.extensions.db import db, dataframe_from_db
from tollan.utils.db import SqlaDB
from tolteca.db.toltec import dataprod
from tollan.utils.log import get_logger
from collections import UserDict


dbrt = ObjectProxy(None)
"""A global db runtime instance.

This is made available if this module is imported
after the dasha db extension is loaded.
"""


class DatabaseRuntime(UserDict):
    """A helper class that provide access to
    TolTEC db connections.

    Parameters
    ----------
    binds_required : list, optional
        The list of binds required. If any is missing, `RuntimeError`
        is raised. Default is to ignore.
    """

    def __init__(self, binds_required=None):
        result = dict()
        if binds_required is None:
            binds_required = []
        for bind in self._binds:
            result[bind] = self._get_sqladb(
                    bind, raise_on_error=bind in binds_required)
        # setup db
        for b, d in result.items():
            self._setup_sqladb(
                    d, func=getattr(self, f'setup_{b}'),
                    raise_on_error=b in binds_required
                    )
        super().__init__(result)

    logger = get_logger()
    # this shall be defined in the dasha db config.
    _binds = ('toltec', 'tolteca')

    @staticmethod
    def _setup_toltec(d):
        d.reflect_tables()

    @staticmethod
    def _setup_tolteca(d):
        data_prod.init_db(d)

    @classmethod
    def _setup_sqladb(cls, sqladb, func, raise_on_error=True):
        try:
            func(sqladb)
        except Exception as e:
            cls.logger.error(
                    f"unable to init db bind={bind}: {e}",
                    exc_info=True)
            if raise_on_error:
                raise
        return sqladb

   def _get_sqladb(cls, bind, raise_on_error=True):
        if not name in cls._binds:
            raise ValueError(f'bind name shall be {cls._binds}')
        try:
            result = SqlaDB.from_flask_sqla(db, bind=bind)
        except Exception as e:
            cls.logger.error(
                    f"unable to connect to db bind={bind}: {e}",
                    exc_info=True)
            if raise_on_error:
                raise
        return result


if db is not None:
    # make available a global instance if dasha db is initialized.
    dbrt.__wrapped__ = DatabaseRuntime()
