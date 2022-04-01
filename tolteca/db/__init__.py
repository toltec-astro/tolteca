#! /usr/bin/env python

from dataclasses import dataclass, field
from schema import Regex
from typing import ClassVar, Sequence

from tollan.utils.log import get_logger
from tollan.utils import odict_from_list
from tollan.utils.db import SqlaDB
from tollan.utils.dataclass_schema import (add_schema, )
# from tollan.utils.fmt import pformat_yaml

from ..utils import RuntimeBase, RuntimeBaseError
from ..utils.config_schema import add_config_schema
from ..utils.doc_helper import collect_config_item_types


__all__ = ['DBConfig', 'DatabaseRuntime', 'DatabaseRuntimeError']


@add_schema
@dataclass
class SqlaBindConfig(object):
    """The config of a database connection."""
    name: str = field(
        metadata={
            'description': 'The name to identify the connection.',
            })

    uri: str = field(
        metadata={
            'description': 'The SQLAlchemy engine URI to connect.',
            'schema': Regex(r'.+:\/\/.*')
            })

    class Meta:
        schema = {
            'ignore_extra_keys': True,
            'description': 'The config dict of a database connection.'
            }

    def __call__(self, **kwargs):
        """Return the `SqlaDB` instance from the URI."""
        return SqlaDB.from_uri(self.uri, **kwargs)


@add_config_schema
@add_schema
@dataclass
class DBConfig(object):
    """The config for `tolteca.db`."""

    _dpdb_bind_name: ClassVar = 'dpdb'

    binds: Sequence[SqlaBindConfig] = field(
        default_factory=lambda: [SqlaBindConfig.from_dict({
            'name': 'dpdb',
            'uri': 'sqlite://',
            })],
        metadata={
            'description': 'The list of database binds (connections).',
            'schema': [SqlaBindConfig.schema, ],
            }
        )

    class Meta:
        schema = {
            'ignore_extra_keys': True,
            'description': 'The config dict for databases.'
            }
        config_key = 'db'

    def __post_init__(self):
        # make a dict for accessing binds by name
        self._binds_by_name = odict_from_list(self.binds, key=lambda b: b.name)

    @property
    def dpdb(self):
        """The bind config for data product database."""
        binds_by_name = self._binds_by_name
        dpdb_bind_name = self._dpdb_bind_name
        dpdb = binds_by_name.get(dpdb_bind_name, None)
        if dpdb is None:
            raise DatabaseRuntimeError(
                "No DPDB bind defined in db.binds config. "
                "To use DPDB, specify an entry in the db.binds with "
                "name 'dpdb'."
                )
        return dpdb


class DatabaseRuntimeError(RuntimeBaseError):
    """Raise when errors occur in `DatabaseRuntime`."""
    pass


class DatabaseRuntime(RuntimeBase):
    """A class that manages databases.

    This class facilitates the interaction with the data product database.
    """

    config_cls = DBConfig

    logger = get_logger()

    @property
    def dpdb_uri(self):
        return self.config.dpdb.uri

    @property
    def dpdb(self):
        return self.config.dpdb()


db_config_item_types = collect_config_item_types(list(locals().values()))
