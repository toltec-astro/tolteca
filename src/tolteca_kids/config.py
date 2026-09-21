"""Input selection and filesystem settings for KIDs runs."""

from tollan.config import FrozenBaseModel
from tollan.config.types import AbsAnyPath, AbsDirectoryPath

__all__ = ["KidsInputConfig"]


class KidsInputConfig(FrozenBaseModel):
    """Settings used to discover and select inputs before per-input validation.

    Explicit input files do not require ``data_lmt_path``. A product root need
    not exist yet; validating it does not create directories or enable output.
    ``select`` is a pandas query over the metadata catalog. Other settings,
    such as pipeline stages, are left for the per-input KIDs handler to validate.
    """

    data_lmt_path: AbsDirectoryPath | None = None
    dataprod_toltec_path: AbsAnyPath | None = None
    select: str | None = None
