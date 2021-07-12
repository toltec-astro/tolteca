#! /usr/bin/env python

from tollan.utils.log import get_logger
from . import main_parser, config
from .check import register_cli_checker
from pathlib import Path
from tollan.utils.db import SqlaDB
from schema import Schema


@register_cli_checker('db')
def check_db(result):
    logger = get_logger()
    schema = Schema({
            'db': {
                str: {
                    'uri': str
                    }
                },
            },
            ignore_extra_keys=True
            )

    add_db_instruction = False
    try:
        db_cfg = schema.validate(config.__wrapped__)
        n_dbs = len(db_cfg)
        if n_dbs > 0:
            result.add_line('ok', f' Loaded {len(db_cfg)} database entries')
        else:
            result.add_line('info', 'No database entry found in config')
            add_db_instruction = True
    except Exception as e:
        logger.debug(f'{e}', exc_info=True)
        result.add_line(
            'error',
            f'Invalid database config ({e}), please check the config files.')
        add_db_instruction = True
    if add_db_instruction:
        result.add_line(
            'note',
            'Valid database is required to manage data products. '
            'To setup, add in the config file with key "db" and '
            'value like: {"<name>": {"uri": "<sqla_engine_url>"}} '
            'where the engine URL follows the convention of '
            'SQLAlchemy: https://docs.sqlalchemy.org/en/14/core/engines.html')
    return result


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
    def action(option, **kwargs):
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
