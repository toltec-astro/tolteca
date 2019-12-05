#! /usr/bin/env python

from tolteca.utils.log import init_logging, get_logger
from tolteca.web.tolteca_flask import config
from tolteca.web.tolteca_flask.db import load_tables
from tolteca.db.connection import DatabaseConnection
import sys


if __name__ == "__main__":
    init_logging()
    logger = get_logger()
    url = config.SQLALCHEMY_DATABASE_URI

    db = DatabaseConnection(
            url, engine=dict(echo=False))
    if db.reflect_tables():
        from prompt_toolkit.shortcuts import confirm
        if not confirm(f'Tables exist in "{url}", override?'):
            print("Exit without any change.")
            sys.exit(0)
    # proceed to re construct all tables
    db.clear_tables()
    load_tables(db, create=True)
    # print the schema

    dumped = []

    def metadata_dump(sql, *args, **kwargs):
        dumped.append(str(sql.compile(dialect=db.engine.dialect)).strip())

    db_mock = DatabaseConnection(
            url, engine=dict(
                echo=False, strategy="mock", executor=metadata_dump))
    load_tables(db_mock)
    # db_mock.reflect_tables()
    db_mock.metadata.create_all()

    logger.info("schema:\n{}".format("\n".join(dumped)))
