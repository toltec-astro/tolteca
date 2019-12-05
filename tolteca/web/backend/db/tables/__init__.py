#! /usr/bin/env python

from sqlalchemy import Table, Column, String, Boolean
from tolteca.utils.log import get_logger


def load_tables(db, create=False, **kwargs):

    logger = get_logger()

    tables = []

    from tolteca.db.utils.conventions import (
            fk, pfk, pk, label, created_at)  # noqa: F401

    def qualified(name):
        return name

    def tbl(name, *args):
        return Table(qualified(name), db.metadata, *args)

    tables.extend([
        tbl(
            "user", pk(index=True),
            Column('name_first', String),
            Column('name_last', String),
            created_at(),
            Column('email', String, unique=True),
            Column('password', String),
            Column('is_active', Boolean, default=True),
            Column('is_superuser', Boolean, default=False),
        ),
        tbl(
            "role", pk(index=True),
            label(index=True),
            created_at(),
        ),
        tbl(
            "user_role", pfk('user'),
            fk('role'),
        ),
        ])
    if create:
        try:
            db.metadata.create_all(db.engine, **kwargs)
        except Exception as e:
            logger.error(f"unable to create tables: {e}")
        else:
            db.reflect_tables()
