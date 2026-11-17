"""Exact artifact selections used by observation preparation and calibration."""

import hashlib

from pydantic import ConfigDict, Field
from tollan.utils.fileloc import fileloc

from ..core.types import FrozenDataModel
from ..data_prod import DataProdIndex, DataProdItem, DataProdRef, DataProdSource


class ArtifactBinding(FrozenDataModel):
    """One artifact in an exact product realization.

    Parameters
    ----------
    product : DataProdRef
        Namespaced product realization; group tags are not resolved here.
    producer, name : str
        Exact artifact metadata selectors.
    interface, scope : str or None, optional
        Additional selectors when the producer emits several scoped artifacts.
        None leaves that selector unspecified; resolution must still be unique.
    source : DataProdSource or None, optional
        Exact selected source, distinguishing file formats and storage copies.
        None requires the metadata selectors alone to be unique.
    """

    model_config = ConfigDict(extra="forbid")
    product: DataProdRef
    producer: str = Field(min_length=1)
    name: str = Field(min_length=1)
    interface: str | None = None
    scope: str | None = None
    source: DataProdSource | None = None

    def resolve(self, index: DataProdIndex) -> DataProdItem:
        """Return the uniquely selected artifact from its pinned index.

        Parameters
        ----------
        index : DataProdIndex
            Exact product index, with its producing execution record.

        Returns
        -------
        DataProdItem
            Matching artifact; use the index to resolve its source location.

        Raises
        ------
        ValueError
            Identity, execution attribution or unique membership differs.
        """
        if index.product.reference != self.product or index.run is None:
            msg = "Artifact binding requires its exact product realization"
            raise ValueError(msg)
        selectors = self.model_dump(exclude={"product", "source"}, exclude_none=True)
        matches = [
            item
            for item in index.product.data_items
            if item.meta.get("type") == "artifact"
            and all(item.meta.get(key) == value for key, value in selectors.items())
            and (self.source is None or item.source == self.source)
        ]
        if len(matches) != 1:
            msg = "Artifact binding must select exactly one artifact"
            raise ValueError(msg)
        [item] = matches
        if item.meta.get("run_uid") != str(index.run.uid):
            msg = "Artifact belongs to a different producing run"
            raise ValueError(msg)
        return item

    def resolve_path(self, index):
        """Resolve and verify a local artifact before passing it to a native engine.

        Parameters
        ----------
        index : DataProdIndex
            Exact product realization. Its named locations must be resolvable.

        Returns
        -------
        pathlib.Path
            Existing local artifact, checked against its SHA-256 receipt when
            present. No file is copied or modified.

        Raises
        ------
        ValueError
            The binding, local location or recorded checksum does not match.
        """
        item = self.resolve(index)
        location = fileloc(index.get_source_uri(item.source))
        if location.is_remote() or not location.path.is_file():
            msg = "Native artifact consumption requires an existing local file"
            raise ValueError(msg)
        if item.source.checksum is not None:
            with location.path.open("rb") as handle:
                checksum = "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()
            if checksum != item.source.checksum:
                msg = f"Pinned artifact checksum differs: {location.path}"
                raise ValueError(msg)
        return location.path
