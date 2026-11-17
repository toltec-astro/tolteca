"""Portable membership directory of an incrementally published version group.

Replacing the root index commits a prefix of numbered membership records. A
per-UID lookup must resolve within that prefix, so abandoned writes are invisible.
The immutable realization owns its artifact inventory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self
from urllib.parse import quote

from pydantic import Field, TypeAdapter, field_validator, model_validator
from pydantic_core import to_json
from tollan.config import FrozenBaseModel

from ..core.data_prod import DataProdAssoc, DataProdRef, RunName
from ..core.data_prod._io import _read_yaml
from .index import DataProdIndex
from .metadata import VersionedProductMeta


class MemberRecord(FrozenBaseModel):
    position: int = Field(ge=1)
    name: RunName
    edge: DataProdAssoc

    @field_validator("edge")
    @classmethod
    def _validate_membership_edge(cls, edge: DataProdAssoc) -> DataProdAssoc:
        if edge.data_prod_assoc_type != "dpa_version_group_data_prod":
            msg = "Version member requires a version-group membership association"
            raise ValueError(msg)
        return edge


class GroupHeader(DataProdIndex):
    document: Literal["version_group_header"] = "version_group_header"
    version_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _validate_version_group(self) -> Self:
        if self.product.meta.data_prod_type != "dp_version_group":
            msg = "Expected a version group index"
            raise ValueError(msg)
        return self

    @classmethod
    def from_index(cls, index: DataProdIndex) -> GroupHeader:
        """Return a live header for a version-group index, with no member records."""
        return cls.model_validate(
            index.model_dump(round_trip=True, exclude={"document"})
        )

    @classmethod
    def load(cls, path: Path) -> GroupHeader:
        """Read a live header, or an exported version group before its first update."""
        document = parse_index_document(_read_yaml(path))
        if isinstance(document, GroupHeader):
            return document
        return cls.from_index(document)

    def lookup(self, directory, uid):
        """Return the committed membership record of the member with product UID."""
        path = directory / ".members" / "by_uid" / f"{quote(uid, safe='')}.yaml"
        if not path.is_file():
            raise KeyError(uid)
        member = MemberRecord.model_validate_json(to_json(_read_yaml(path)))
        if member.position > self.version_count or member != self.member_at(
            directory, member.position
        ):
            raise KeyError(uid)
        if member.edge.uid != uid:
            msg = "Committed version membership differs from the requested member"
            raise ValueError(msg)
        return member

    def member_at(self, directory, position):
        member = MemberRecord.model_validate_json(
            to_json(_read_yaml(directory / ".members" / f"{position}.yaml"))
        )
        if member.position != position:
            msg = "Invalid version membership position"
            raise ValueError(msg)
        return member

    def expand(self, directory, run_uids):
        data = self.model_dump(round_trip=True, exclude={"version_count", "document"})
        if run_uids is None:
            members = (
                self.member_at(directory, p) for p in range(1, self.version_count + 1)
            )
        else:
            members = []
            # The store composes a realization's UID from its group and run.
            for uid in dict.fromkeys(run_uids):
                try:
                    members.append(self.lookup(directory, f"{self.product.uid}/{uid}"))
                except KeyError:
                    continue
        for m in members:
            p = DataProdIndex.read(directory / m.name / "index.yaml")
            if (
                p.run is None
                or p.store_run is None
                or p.store_run.name != m.name
                or (p.product.uid, p.product.namespace_uid)
                != (m.edge.uid, m.edge.namespace_uid)
                or not isinstance(p.product.meta, VersionedProductMeta)
                or p.product.meta.version_group != DataProdRef(uid=self.product.uid)
            ):
                msg = "Version membership differs from its realized product"
                raise ValueError(msg)
            for loc in p.locations.values():
                if loc.is_relative:
                    loc.root = (Path(m.name) / loc.root).as_posix()
            data["products"].append(p)
            data["product"]["assocs"].append(m.edge)
            data["actors"].update(p.actors)
        group = DataProdIndex.model_validate(data)
        group._bind_locations(directory.resolve())
        return group


_INDEX_DOCUMENT = TypeAdapter(
    Annotated[GroupHeader | DataProdIndex, Field(discriminator="document")]
)


def parse_index_document(document):
    """Validate a loaded product index or live version-group header by its kind."""
    return _INDEX_DOCUMENT.validate_json(to_json(document))
