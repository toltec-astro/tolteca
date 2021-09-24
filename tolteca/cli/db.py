#! /usr/bin/env python

from tollan.utils.log import get_logger
from tollan.utils import ensure_abspath
from tollan.utils.fmt import pformat_yaml
from . import main_parser, config_loader
from .check import (
    register_cli_checker,
    _check_load_config, _check_load_rc, _MISSING)
from pathlib import Path
from tollan.utils.db import SqlaDB
from tollan.utils.dataclass import add_schema
from tollan.utils.schema import NestedSchemaHelperMixin
# from tollan.utils.fmt import pformat_yaml
from schema import Schema, Regex, Optional, Literal, Use
from ..utils import RuntimeContext, RuntimeContextError
from dataclasses import dataclass, field
from ..utils.config_mapper import ConfigMapperMixin
from tollan.utils.namespace import Namespace


@add_schema
@dataclass
class DBConfigEntry(NestedSchemaHelperMixin):
    """The config of a database."""
    uri: str = field(
        metadata={
            'description': 'The SQLAlchemy engine URI of the database.',
            'schema': Regex(r'.+:\/\/.+')
            })

    class Meta:
        schema = {
            'ignore_extra_keys': True,
            'description': 'The config dict of a database.'
            }


class DBConfig(Namespace, ConfigMapperMixin, NestedSchemaHelperMixin):
    """The class mapped to the database config."""

    _dpdb_name = 'tolteca'
    _config_key = 'db'  # consumed by ConfigMapperMixin
    _namespace_from_dict_schema = Schema({
        Literal(
            _dpdb_name, description='The data product database config.'):
        DBConfigEntry,
        Optional(str, description='Any other databases.'):
        DBConfigEntry,
        })
    _namespace_to_dict_schema = Schema(Use(
        lambda d: {k: v for k, v in d.items() if isinstance(v, DBConfigEntry)}
        ))
    schema = _namespace_from_dict_schema  # consumed by ConfigMapperMixin

    @property
    def dpdb(self):
        """The config entry for dpdb."""
        return getattr(self, self._dpdb_name)


class DatabaseRuntimeError(RuntimeContextError):
    """Raise when errors occur in `DatabaseRuntime`."""
    pass


class DatabaseRuntime(RuntimeContext):
    """A class that manages the runtime context of the databases.

    This class facilitates the interaction with the data file database and
    the data product database.
    """

    @property
    def db_config(self):
        return DBConfig.from_config(self.config)

    @property
    def dpdb_uri(self):
        return self.db_config.dpdb.uri


@register_cli_checker('db')
def check_db(result):

    def _check_db_config(result, config, source):
        add_db_instruction = True
        add_dpdb_instruction = True
        if DBConfig._config_key in config:
            add_db_instruction = False
            db_config_dict = config[DBConfig._config_key]
            n_dbs = len(db_config_dict)
            result.add_item(
                result.S.info,
                f'Found {n_dbs} database entries in {source}',
                details=pformat_yaml(db_config_dict)
                )
            try:
                import pdb
                pdb.set_trace()
                db_config = DBConfig.from_dict(config)
                add_dpdb_instruction = False
                # report the db_config info
                result.add_item(
                    result.S.ok,
                    f'Found dpdb (tolteca) entry in {source}',
                    details=pformat_yaml(db_config.dpdb)
                    )
            except Exception as e:
                raise
                result.add_item(
                    result.S.error,
                    'Unable to create DBConfig object',
                    details=f'Reason: {e}'
                    )

                pass
        return add_db_instruction, add_dpdb_instruction

    add_db_instruction = True
    add_dpdb_instruction = True
    config = _check_load_config(result)
    if config is not _MISSING:
        add_db_instruction, add_dpdb_instruction = \
            _check_db_config(result, config, source='config files')
    rc = _check_load_rc(result)
    if rc is not _MISSING and rc is not None:
        add_db_instruction, add_dpdb_instruction = \
            _check_db_config(result, rc.config, source=f'{rc}')
    if add_db_instruction:
        result.add_item(
            result.S.note,
            'Valid database is required to manage data products. '
            'To setup, add in the config file with key "db" and '
            'value like: {"<name>": {"uri": "<sqla_engine_url>"}} '
            'where the engine URL follows the convention of '
            'SQLAlchemy: https://docs.sqlalchemy.org/en/14/core/engines.html'
            )
    return result
#     add_db_instruction = False
#     add_dpdb_instruction = False

#     def _check_db_cfg(db_cfg, source):
#         nonlocal add_db_instruction
#         nonlocal add_dpdb_instruction
#         if db_cfg is None:
#             add_db_instruction = True
#             return
#         db_cfg_schema = Schema(
#             {Optional(str): DatabaseRuntime._db_entry_schema})
#         try:
#             db_cfg = db_cfg_schema.validate(db_cfg)
#             n_dbs = len(db_cfg)
#             if n_dbs > 0:
#                 result.add_item(
#                     result.S.info,
#                     f'Found {n_dbs} database entries in {source}',
#                     details=pformat_yaml(db_cfg)
#                     )
#             else:
#                 result.add_item(
#                     result.S.error,
#                     f'No database entry found in {source}')
#                 add_db_instruction = True
#             if DatabaseRuntime._dpdb_name in db_cfg:
#                 result.add_item(
#                     result.S.ok,
#                     f'Found dpdb (tolteca) entry in {source}')
#             else:
#                 result.add_item(
#                     result.S.error,
#                     f'No dpdb (tolteca) entry in {source}')
#                 add_dpdb_instruction = True
#         except Exception as e:
#             logger.debug(f'{e}', exc_info=True)
#             result.add_item(
#                 result.S.error,
#                 f'Invalid database config ({e}), please check the {source}')
#             add_db_instruction = True
#     if base_runtime_context:
#         db_cfg = base_runtime_context.config.get(
#             DatabaseRuntime._config_key, None)
#         if db_cfg is not None:
#             flag = result.S.info
#             verb = 'Found'
#         else:
#             flag = result.S.error
#             verb = 'No'
#         result.add_item(
#             flag,
#             f'{verb} db config in {base_runtime_context}',
#             details=pformat_yaml(base_runtime_context.config_files)
#             )
#         _check_db_cfg(db_cfg, f'{base_runtime_context}')
#     else:
#         result.add_item(
#             result.S.info,
#             'Base runtime context is not loaded for checking db config'
#             )

#     db_cfg = config.get(DatabaseRuntime._config_key, None)
#     config_files = config['runtime']['config_info']['sources']
#     if db_cfg is not None:
#         flag = result.S.info
#         verb = 'Found'
#     else:
#         flag = result.S.error
#         verb = 'No'
#     result.add_item(
#         flag,
#         f'{verb} db config in config files',
#         details=pformat_yaml(config_files)
#         )
#     _check_db_cfg(db_cfg, 'config files')

#     if add_db_instruction:
#         result.add_item(
#             result.S.note,
#             'Valid database is required to manage data products. '
#             'To setup, add in the config file with key "db" and '
#             'value like: {"<name>": {"uri": "<sqla_engine_url>"}} '
#             'where the engine URL follows the convention of '
#             'SQLAlchemy: https://docs.sqlalchemy.org/en/14/cngines.html')
#     if add_db_instruction or add_dpdb_instruction:
#         result.add_item(
#             result.S.note,
#             'A database entry named "tolteca" is required to manage data '
#             'products. After the entry is added, use "tolteca migrate" to '
#             'initialize the data product database.')

#     return result


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

        try:
            rc = DatabaseRuntime(workdir)
        except Exception as e:
            raise DatabaseRuntimeError(
                f'Cannot create database runtime: {e}. Check the config files '
                f'and runtime context dir for config key "db". A valid '
                f'db entry named "tolteca" is required to create the '
                f'data product database.')
        logger.debug(f'db rc:{rc}')

        # carray on with operations specified in option
        dpdb_uri = rc.dpdb_uri
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
