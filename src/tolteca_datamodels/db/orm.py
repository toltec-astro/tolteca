from tollan.db.orm import BetterDeclarativeBase, ClientInfoMixin, SqlaORM


class Base(BetterDeclarativeBase):
    """The declarative base."""

    __abstract__ = True


class ClientInfo(Base, ClientInfoMixin):
    """The client info."""


orm = SqlaORM(Base=Base, ClientInfo=ClientInfo)
