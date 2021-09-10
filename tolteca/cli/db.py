#! /usr/bin/env python

from tollan.utils.log import get_logger
from . import main_parser, config, base_runtime_context, _make_runtime_context
from .check import register_cli_checker
from pathlib import Path
from tollan.utils.db import SqlaDB
from tollan.utils.fmt import pformat_yaml
from schema import Schema, Regex, Optional, Literal
from ..utils import RuntimeContext, RuntimeContextError


class DatabaseRuntimeError(RuntimeContextError):
    """Raise when errors occur in `DatabaseRuntime`."""
    pass


class DatabaseRuntime(RuntimeContext):
    """A class that manages the runtime context of the databases.

    This class facilitates the interaction with the data file database and
    the data product database.
    """

    _dpdb_name = 'tolteca'
    _config_key = 'db'
    _db_entry_schema = {
        Literal('uri', description='The SQLAlchemy engine URI'):
            Regex(r'.+:\/\/.+'),
        }

    @property
    def db_config(self):
        return self.config[self._config_key]

    @property
    def dpdb_uri(self):
        return self.db_config[self._dpdb_name]['uri']

    @classmethod
    def extend_config_schema(cls):
        # this defines the subschema relevant to the db.
        return {
                Literal(
                    cls._config_key,
                    description='The database runtime config.'): {
                        Literal(
                            cls._dpdb_name,
                            description='The data product database config.'):
                        cls._db_entry_schema,
                    Optional(str, description='Any other databases.'):
                        cls._db_entry_schema,
                },
            }


@register_cli_checker('db')
def check_db(result):
    logger = get_logger()

    add_db_instruction = False
    add_dpdb_instruction = False

    def _check_db_cfg(db_cfg, source):
        nonlocal add_db_instruction
        nonlocal add_dpdb_instruction
        if db_cfg is None:
            add_db_instruction = True
            return
        db_cfg_schema = Schema(
            {Optional(str): DatabaseRuntime._db_entry_schema})
        try:
            db_cfg = db_cfg_schema.validate(db_cfg)
            n_dbs = len(db_cfg)
            if n_dbs > 0:
                result.add_item(
                    result.S.info,
                    f'Found {n_dbs} database entries in {source}',
                    details=pformat_yaml(db_cfg)
                    )
            else:
                result.add_item(
                    result.S.error,
                    f'No database entry found in {source}')
                add_db_instruction = True
            if DatabaseRuntime._dpdb_name in db_cfg:
                result.add_item(
                    result.S.ok,
                    f'Found dpdb (tolteca) entry in {source}')
            else:
                result.add_item(
                    result.S.error,
                    f'No dpdb (tolteca) entry in {source}')
                add_dpdb_instruction = True
        except Exception as e:
            logger.debug(f'{e}', exc_info=True)
            result.add_item(
                result.S.error,
                f'Invalid database config ({e}), please check the {source}')
            add_db_instruction = True
    if base_runtime_context:
        db_cfg = base_runtime_context.config.get(
            DatabaseRuntime._config_key, None)
        if db_cfg is not None:
            flag = result.S.info
            verb = 'Found'
        else:
            flag = result.S.error
            verb = 'No'
        result.add_item(
            flag,
            f'{verb} db config in {base_runtime_context}',
            details=pformat_yaml(base_runtime_context.config_files)
            )
        _check_db_cfg(db_cfg, f'{base_runtime_context}')
    else:
        result.add_item(
            result.S.info,
            'Base runtime context is not loaded for checking db config'
            )

    db_cfg = config.get(DatabaseRuntime._config_key, None)
    config_files = config['runtime']['config_info']['sources']
    if db_cfg is not None:
        flag = result.S.info
        verb = 'Found'
    else:
        flag = result.S.error
        verb = 'No'
    result.add_item(
        flag,
        f'{verb} db config in config files',
        details=pformat_yaml(config_files)
        )
    _check_db_cfg(db_cfg, 'config files')

    if add_db_instruction:
        result.add_item(
            result.S.note,
            'Valid database is required to manage data products. '
            'To setup, add in the config file with key "db" and '
            'value like: {"<name>": {"uri": "<sqla_engine_url>"}} '
            'where the engine URL follows the convention of '
            'SQLAlchemy: https://docs.sqlalchemy.org/en/14/core/engines.html')
    if add_db_instruction or add_dpdb_instruction:
        result.add_item(
            result.S.note,
            'A database entry named "tolteca" is required to manage data '
            'products. After the entry is added, use "tolteca migrate" to '
            'initialize the data product database.')

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

    logger = get_logger()

    @parser.parser_action
    def action(option, **kwargs):
        try:
            dbrt = _make_runtime_context(DatabaseRuntime)
        except Exception as e:
            raise DatabaseRuntimeError(
                f'Cannot create database runtime: {e}. Check the config files '
                f'and runtime context dir for config key "db". A valid '
                f'db entry named "tolteca" is required to create the '
                f'data product database.')
        logger.debug(f'loaded db config:{pformat_yaml(dbrt.db_config)}')
        dpdb_uri = dbrt.dpdb_uri
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
