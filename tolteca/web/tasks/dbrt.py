#!/usr/bin/env python

from dasha.web.extensions.db import db, DatabaseRuntime  # , dataframe_from_db
from tolteca.datamodels.db.toltec import data_prod
from wrapt import ObjectProxy


dbrt = ObjectProxy(None)
"""A global db runtime instance.

This is made available if this module is imported
after the dasha db extension is loaded.
"""


def setup_tolteca_db(d):
    data_prod.init_db(d, create_tables=False)
    data_prod.init_orm(d)


if db is not None:
    # make available a global instance if dasha db is initialized.
    dbrt.__wrapped__ = DatabaseRuntime(
        binds=['toltec', 'tolteca', 'toltec_userlog_tool'],
        setup_funcs={
            'toltec': 'reflect_tables',
            'toltec_userlog_tool': 'reflect_tables',
            'tolteca': setup_tolteca_db
            }
        )
