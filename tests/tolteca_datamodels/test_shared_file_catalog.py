"""Shared file projections have no observation-schema dependency."""

from typing import ClassVar

from tollan.accessor.pandas.model_table import ModelColumn

from tolteca_datamodels.core.types import DataModelInfo
from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.toltec.file import get_info_from_source


class _ObservationsDataModel(DataModelInfo):
    name = "observations"


def test_subclass_owns_its_projection_factory():
    class ObservationFiles(FileAccessor[_ObservationsDataModel]):
        _observation_columns: ClassVar = FileAccessor._source_columns | {
            "obsnum": ModelColumn("obsnum", default=None, dtype="Int64"),
        }

        @classmethod
        def from_observations(cls, observations):
            return cls.from_models(observations, columns=cls._observation_columns)

    info = get_info_from_source("tel_toltec_2025-10-31_145609_00_0002.nc")
    frame = ObservationFiles.from_observations([info])
    assert list(frame) == [
        "source",
        "source_loc",
        "source_loc_is_local",
        "source_loc_host",
        "source_loc_scheme",
        "obsnum",
    ]
    assert len(ObservationFiles(frame).query("obsnum == 145609")) == 1
    assert list(ObservationFiles.from_observations([])) == list(frame)
    assert "data_kind" not in frame
