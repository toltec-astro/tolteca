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
    parser.add_argument(
            '--recreate_all', action='store_true',
            help='If set, all tables are recreated.'
            )

    @parser.parser_action
    def action(option):
        logger = get_logger()

        dpdb_uri = option.config['db']['tolteca']['uri']
        logger.debug(f"migrate database: {dpdb_uri}")

        db = SqlaDB.from_uri(dpdb_uri, engine_options={'echo': True})

        from ..datamodels.db.toltec import data_prod

        data_prod.init_db(
                db, create_tables=True, recreate=option.recreate_all)

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
