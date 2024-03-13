import os
from pathlib import Path

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, declared_attr, mapped_column, relationship
from tollan.db import SqlaDB
from tollan.db import mapped_types as mtypes
from tollan.utils.general import ensure_abspath

from .core import current_dpdb
from .orm import Base, orm

__all__ = ["FileDB"]


class RootDirectory(Base, mtypes.TimestampMixin, orm.ClientInfoRefMixin):
    """The root directories of files."""

    __tablename__ = "root_directories"
    pk: Mapped[mtypes.Pk] = mapped_column(init=False)
    path: Mapped[str] = mapped_column(unique=True, comment="The path.")
    files: Mapped[list["File"]] = relationship(
        repr=False,
        default_factory=list,
        back_populates="root_directory",
    )
    priority: Mapped[int] = mapped_column(
        default=0,
        comment=(
            "The priority. Root directory with larger value "
            "is first considered when resolving conflictions."
        ),
    )


class File(Base, mtypes.TimestampMixin, orm.ClientInfoRefMixin):
    """The files."""

    __tablename__ = "files"

    @declared_attr.directive
    @classmethod
    def __table_args__(cls):
        args = {} | Base.__table_args__
        return sa.UniqueConstraint("name", "path", "root_directory_pk"), args

    pk: Mapped[mtypes.Pk] = mapped_column(init=False)
    name: Mapped[str] = mapped_column(comment="The filename.")
    path: Mapped[str] = mapped_column(comment="The relative path to root directory.")
    readable: Mapped[bool] = mapped_column(comment="True if file is reable.")
    root_directory_pk: Mapped[int] = mtypes.fk(
        RootDirectory,
        comment="The root directory id.",
    )
    root_directory: Mapped["RootDirectory"] = relationship(back_populates="files")
    io_cls: Mapped[str] = mapped_column(
        comment="The python class to operate this file.",
    )


class FileDB(orm.WorkflowBase):
    """The database task to manage files."""

    def __init__(self, db: SqlaDB = current_dpdb):
        super().__init__(db=db, client_name="file_db")

    def _collect_files(self, root_directory: RootDirectory):
        rootpath = Path(root_directory.path)
        data = []
        for parent_, _, files in os.walk(rootpath):
            parent = Path(parent_)
            for file in files:
                relpath = (parent / file).relative_to(rootpath)
                readable = True
                data.append(
                    {
                        "name": file,
                        "path": relpath.as_posix(),
                        "readable": readable,
                        "root_directory_pk": root_directory.pk,
                        "client_info_pk": self.client_info.pk,
                    },
                )
        with self.db.session_context():
            File.batch_upsert(
                data,
                index_elements=[
                    "name",
                    "path",
                    "root_directory_pk",
                ],
            )

    def collect_files(self, rootpath):
        """Collect files from `rootpath`."""
        rootpath = ensure_abspath(rootpath)
        with self.db.session_context():
            rootdir, _ = RootDirectory.get_or_create(
                path=rootpath.as_posix(),
                client_info=self.client_info,
            )
        self._collect_files(rootdir)
