#! /usr/bin/env python

from tollan.utils.log import get_logger
from . import main_parser
from pathlib import Path
from tollan.utils.db import SqlaDB


@main_parser.register_action_parser(
        'migrate',
        help="Create/migrate the data product database."
        )
def cmd_migrate(parser):

    parser.add_argument(
            '--schema_graph', metavar='FILE', default=None,
            help='If set, a schema diagram is saved to given path.'
            )

    @parser.parser_action
    def action(option):
        logger = get_logger()

        dpdb_url = option.config['db']['tolteca']['url']
        logger.debug(f"migrate database: {dpdb_url}")

        db = SqlaDB.from_uri(dpdb_url, engine_options={'echo': True})

        from ..db.toltec import dataprod

        dataprod.init_db(db, create_tables=True)

        if option.schema_graph is not None:
            from sqlalchemy_schemadisplay import create_schema_graph

            graph = create_schema_graph(
                    metadata=db.metadata,
                    show_datatypes=False,
                    show_indexes=False,
                    rankdir='LR',
                    concentrate=False
                    )
            graph.write(
                    option.schema_graph,
                    fmt=Path(option.schema_graph).suffix)
