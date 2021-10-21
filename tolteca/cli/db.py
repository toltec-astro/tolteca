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
from tollan.utils.dataclass_schema import (add_schema, DataclassNamespace)
# from tollan.utils.fmt import pformat_yaml
from schema import Schema, Regex, Optional, Literal, Use
from ..utils import RuntimeContext, RuntimeContextError
from dataclasses import dataclass, field
from ..utils.config_schema import add_config_schema


@add_schema
@dataclass
class DBConfigEntry(object):
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


@add_config_schema
class DBConfig(DataclassNamespace):
    """The class mapped to the database config."""

    _dpdb_name = 'tolteca'
    _config_key = 'db'  # consumed by ConfigMapperMixin
    _namespace_from_dict_schema = Schema({
        Literal(
            _dpdb_name, description='The data product database config.'):
        DBConfigEntry.schema,
        Optional(str, description='Any other databases.'):
        DBConfigEntry.schema,
        })
    _namespace_to_dict_schema = Schema(Use(
        lambda d: {k: v for k, v in d.items() if isinstance(v, DBConfigEntry)}
        ))

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
def check_db(result, option):

    def _check_db_config(result, config, source):
        add_db_instruction = True
        add_dpdb_instruction = True
        if DBConfig.config_key in config:
            add_db_instruction = False
            db_config_dict = config[DBConfig.config_key]
            n_dbs = len(db_config_dict)
            result.add_item(
                result.S.info,
                f'Found {n_dbs} database entries in {source}',
                details=pformat_yaml(db_config_dict)
                )
            try:
                db_config = DBConfig.from_config(config)
                add_dpdb_instruction = False
                # report the db_config info
                result.add_item(
                    result.S.ok,
                    f'Found dpdb (tolteca) entry in {source}',
                    details=pformat_yaml(db_config.dpdb.to_dict())
                    )
            except Exception as e:
                result.add_item(
                    result.S.error,
                    'Unable to create DBConfig object',
                    details=f'Reason: {e}'
                    )

                pass
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
            'value like: {"<name>": {"uri": "<sqla_engine_url>"}} '
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
