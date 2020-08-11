#! /usr/bin/env python

from ...utils.fmt import pformat_dict
from sqlalchemy import Table, Column, Integer, String
from ...utils.log import get_logger


def create_tables(db):
    logger = get_logger()

    tables = []

    from ..utils.conventions import fk, pk, label  # noqa: F401

    def qualified(name):
        return name

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
        logger.error(f"unable to create tables: {e}")
    else:
        db.metadata.reflect(db.engine)
        logger.debug(f"tables {pformat_dict(db.metadata.tables)}")
