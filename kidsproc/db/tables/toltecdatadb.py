#! /usr/bin/env python

from ...utils.fmt import pformat_dict
from sqlalchemy import Table, Column, Integer, String, ForeignKey
from astropy import log


def create_tables(db):
    tables = []

    def qualified(name):
        return name

    def fk(other):
        return Column(
            f'{other}_pk', Integer,
            ForeignKey(
               f"{other}.pk", onupdate="cascade", ondelete="cascade"),
            nullable=False)

    def pfk(other):
        return Column(
            'pk', Integer,
            ForeignKey(
               f"{other}.pk", onupdate="cascade", ondelete="cascade"),
            primary_key=True)

    def pk():
        return Column('pk', Integer, primary_key=True)

    def label():
        return Column('label', String)

    def tbl(name, *args):
        return Table(qualified(name), db.metadata, *args)

    tables.extend([
        tbl(
            "master_info", pk(),
            Column('id_obs', Integer),
            Column('id_subobs', Integer),
            Column('id_scan', Integer),
            fk('master_name'),
            fk('file')
        ),
        tbl(
            "master_name", pk(), label(),
        ),
        tbl(
            "file", pk(),
            fk('interface'),
            Column('source', String),
        ),
        tbl(
            "interface", pk(), label(),
            fk('instrument'),
        ),
        tbl(
            "instrument", pk(), label(),
        ),
        tbl(
            "kidsdataproc", pk(),
            fk('kidsdata'),
            fk('proc_info'),
        ),
        tbl(
            "kidsdata", pk(),
            fk('kidsdatakind'),
            fk('file'),
        ),
        tbl(
            "kidsdatakind", pk(), label(),
            fk('dataspec'),
        ),
        tbl(
            "dataspec", pk(),
            fk("dataspec_name"),
            fk("dataspec_version"),
        ),
        tbl(
            "dataspec_name", pk(), label(),
        ),
        tbl(
            "dataspec_version", pk(), label(),
        ),
        tbl(
            "proc_info", pk(),
            fk('proc_name'),
            fk('pipeline_info'),
        ),
        tbl(
            "proc_name", pk(), label()
        ),
        tbl(
            "pipeline_info", pk(),
            fk("pipeline_name"),
            fk("pipeline_version"),
        ),
        tbl(
            "pipeline_name", pk(), label(),
        ),
        tbl(
            "pipeline_version", pk(), label(),
        ),
        ])
    try:
        db.metadata.create_all(db.engine)
    except Exception as e:
        log.error(f"unable to create tables: {e}")
    else:
        db.metadata.reflect(db.engine)
        log.debug(f"tables {pformat_dict(db.metadata.tables)}")
