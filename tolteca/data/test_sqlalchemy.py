#! /usr/bin/env python

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String

class DB(object):
    def __init__(self):
        self.engine = create_engine(
                "sqlite+pysqlite:////Users/ma/Codes/toltec/kids/tolteca/tolteca/data/debug.sqlite", echo = True)
        self.meta = MetaData()

db = DB()

procinfo = Table(
        "procinfo",
        db.meta,
        Column('pk', Integer, primary_key=True),
        Column('label', String)
        )
db.meta.create_all(db.engine)
