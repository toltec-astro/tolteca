#! /usr/bin/env python

from sqlalchemy import Table, Column, Integer, String, ForeignKey, MetaData
from astropy import log


def qualified(name):
    return name


def create_tables(db):
    tables = []

    def fk(other):
        return Column(
            f'{other}_pk', Integer,
            ForeignKey(
               f"{other}.pk", onupdate="cascade", ondelete="cascade"),
            nullable=False)

    tables.extend([
        Table(
            qualified("kidsdata"),
            db.metadata,
            Column('pk', Integer, primary_key=True),
            fk('dataspec'),
            fk('procinfo'),
            Column('obsid', Integer),
            Column('source', String),
        ),
        Table(
            qualified("dataspec"),
            db.metadata,
            Column('pk', Integer, primary_key=True),
            fk('instru'),
            Column('version', Integer)
        ),
        Table(
            qualified("instru"),
            db.metadata,
            Column('pk', Integer, primary_key=True),
            Column('label', String)
        ),
        Table(
            qualified("procinfo"),
            db.metadata,
            Column('pk', Integer, primary_key=True),
            Column('label', String)
        )
        ])
    try:
        db.metadata.create_all(db.engine)
    except Exception as e:
        log.error(f"unable to create tables: {e}")
    else:
        db.metadata.reflect(db.engine)
        log.debug(f"tables {db.metadata.tables}")
