#! /usr/bin/env python

from pathlib import Path

from tollan.utils.log import get_logger
from tollan.utils import ensure_abspath
from tollan.utils.fmt import pformat_yaml

from . import main_parser, config_loader
from .check import (
    register_cli_checker,
    _check_load_config, _check_load_rc, _MISSING)


@register_cli_checker('db')
def check_db(result, option):

    from ..db import DBConfig

    def _check_db_config(result, config, source):
        add_db_instruction = True
        add_dpdb_instruction = True
        if DBConfig.config_key in config:
            add_db_instruction = False
            db_config_dict = config[DBConfig.config_key]
            if 'binds' not in db_config_dict:
                db_binds_list = list()
            else:
                db_binds_list = db_config_dict['binds']
            n_dbs = len(db_binds_list)
            result.add_item(
                result.S.info,
                f'Found {n_dbs} database entries in {source}',
                details=pformat_yaml(db_binds_list)
                )
            try:
                db_config = DBConfig.from_config_dict(config)
                if db_config.dpdb is not None:
                    add_dpdb_instruction = False
                    # report the db_config info
                    result.add_item(
                        result.S.ok,
                        f'Found dpdb entry in {source}',
                        details=pformat_yaml(db_config.dpdb.to_dict())
                        )
            except Exception as e:
                result.add_item(
                    result.S.error,
                    'Unable to create DBConfig object',
                    details=f'Reason: {e}'
                    )
        else:
            result.add_item(
                result.S.error,
                f'No db config found in {source}')
        return add_db_instruction, add_dpdb_instruction

    add_db_instruction = True
    add_dpdb_instruction = True
    config = _check_load_config(result)
    if config is not _MISSING:
        add_db_instruction, add_dpdb_instruction = \
            _check_db_config(result, config, source='config files')
    else:
        result.add_item(
            result.S.info,
            'Skipped checking db config in config files (not loaded).')
    rc = _check_load_rc(result)
    if rc is not _MISSING and rc is not None:
        add_db_instruction, add_dpdb_instruction = \
            _check_db_config(result, rc.config, source=f'{rc}')
    else:
        result.add_item(
            result.S.info,
            'Skipped checking db config in runtime context (not loaded).')
    if add_db_instruction:
        result.add_item(
            result.S.note,
            'Valid database is required to manage data products. '
            'To setup, add in the config file with key "db" and '
            'a list of "bind" dicts, each of which contains a '
            'name and the SQLAlchemy engine URI, '
            'where the engine URL follows the convention of '
            'SQLAlchemy: https://docs.sqlalchemy.org/en/14/core/engines.html'
            )
    return result


@main_parser.register_action_parser(
        'migrate',
        help="Create/migrate the data product database."
        )
def cmd_migrate(parser):

    logger = get_logger()

    parser.add_argument(
            '--schema_graph', metavar='FILE', default=None,
            help='If set, a schema diagram is saved to given path.'
            )
    parser.add_argument(
            '--recreate_all', action='store_true',
            help='If set, all tables are recreated.'
            )

    @parser.parser_action
    def action(option, unknown_args=None):

        logger.debug(f"option: {option}")
        logger.debug(f"unknown_args: {unknown_args}")

        workdir = config_loader.runtime_context_dir

        if workdir is None:
            # in this special case we just use the current directory
            workdir = ensure_abspath('.')

        from ..db import DatabaseRuntime
        from ..datamodels.db.toltec import data_prod

        try:
            dbrt = DatabaseRuntime(workdir)
        except Exception as e:
            raise RuntimeError(
                f'Cannot create database runtime: {e}. Check the config files '
                f'and runtime context dir for config key "db". A valid '
                f'db bind entry named "dpdb" is required to create the '
                f'data product database.')
        logger.debug(f'dbrt config: {dbrt.config.to_dict()}')

        # carry on with operations specified in option
        dpdb = dbrt.dpdb
        logger.debug(f"migrate data product database: {dpdb}")

        data_prod.init_db(
            dpdb,
            create_tables=True,
            recreate=option.recreate_all)

        if option.schema_graph is not None:
            from sqlalchemy_schemadisplay import create_schema_graph

            graph = create_schema_graph(
                    metadata=dpdb.metadata,
                    show_datatypes=False,
                    show_indexes=False,
                    rankdir='LR',
                    concentrate=False
                    )
            graph.write(
                    option.schema_graph,
                    fmt=Path(option.schema_graph).suffix)
