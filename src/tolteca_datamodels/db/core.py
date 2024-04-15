from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

from pydantic import Field, PrivateAttr, field_validator
from pydantic.networks import Url
from tollan.config.types import ImmutableBaseModel
from tollan.db import SqlaDB
from tollan.utils.general import ObjectProxy
from tollan.utils.log import logger

from tolteca_config.core import ConfigHandler, ConfigModel, SubConfigKeyTransformer

__all__ = ["DB", "db_context"]


if TYPE_CHECKING:
    current_dpdb: SqlaDB
else:
    current_dpdb = ObjectProxy()
    """The proxy to current data product database."""


class SqlaBindConfig(ImmutableBaseModel):
    """The config of a database connection."""

    name: str = Field(description="The name to identify the connection.")
    url: Url = Field(description="The SQLAlchemy Url to connect.")
    engine_options: dict = Field(
        description="The options passed to engine.",
        default_factory=dict,
    )

    def connect(self, **kwargs):
        """Return SqlaDB instance."""
        kwargs["engine_options"] = self.engine_options | kwargs.get(
            "engine_options",
            {},
        )
        logger.debug(f"connect database {self}")
        return SqlaDB.from_url(self.url.unicode_string(), **kwargs)


DPDB_BIND_NAME = "dpdb"
"""The data product database bind name."""

_default_dpdb_config = {"name": DPDB_BIND_NAME, "url": "sqlite://"}


class DBConfig(ConfigModel):
    """The config for database settings."""

    binds: list[SqlaBindConfig] = Field(
        description="The list of database conections.",
        default_factory=list,
    )

    _binds_by_name: dict[str:SqlaBindConfig] = PrivateAttr()

    @field_validator("binds", mode="after")
    @classmethod
    def _validate(cls, binds: list[SqlaBindConfig]):
        binds_by_name = {b.name: b for b in binds}
        if len(binds_by_name) != len(binds):
            raise ValueError("duplicated bind names.")
        if DPDB_BIND_NAME not in binds_by_name:
            # install a default dpdp bind
            dpdb = SqlaBindConfig.model_validate(_default_dpdb_config)
            binds.insert(0, dpdb)
            logger.info(f"use fallback dpdb {dpdb.url}")
        return binds

    def __init__(self, **data):
        super().__init__(**data)
        self._binds_by_name = {b.name: b for b in self.binds}

    @property
    def dpdb(self):
        """The bind config for the data product database."""
        return self[DPDB_BIND_NAME]

    def __getitem__(self, name: str) -> SqlaBindConfig:
        if name not in self._binds_by_name:
            raise ValueError(f"db config of name={name} does not exist.")
        return self._binds_by_name[name]


class DB(SubConfigKeyTransformer[Literal["db"]], ConfigHandler[DBConfig]):
    """The class to work with KIDs data."""

    def connect(self, name: str, **kwargs):
        """Return connection of specified database."""
        return self.config[name].connect(**kwargs)

    def connect_dpdb(self, **kwargs):
        """Return the data product database connection."""
        return self.config.dpdb.connect(**kwargs)


@contextmanager
def db_context(
    db_config: None | str | DB | DBConfig = None,
    name: str = DPDB_BIND_NAME,
    **kwargs,
):
    """Open context with db connection setup."""
    if db_config is None:
        dbc = DBConfig.model_validate({})
    elif isinstance(db_config, DBConfig):
        dbc = db_config
    elif isinstance(db_config, DB):
        dbc = db_config.config
    elif isinstance(db_config, str):
        dbc = DB(db_config).config
    else:
        raise TypeError(f"invalid db_config type: {type(db_config)}")
    db_obj = dbc[name].connect(**kwargs)
    if name == DPDB_BIND_NAME:
        # update current_dpdb proxy
        current_dpdb.proxy_init(db_obj)
        logger.debug(f"enter current_dpdb context: {current_dpdb.engine.url}")
    try:
        yield db_obj
    finally:
        if name == DPDB_BIND_NAME:
            logger.debug(f"exit current_dpdb context: {current_dpdb.engine.url}")
            current_dpdb.proxy_reset()
