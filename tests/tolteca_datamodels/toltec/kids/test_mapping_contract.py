"""Native readers obey the supplied node and resolved field locations."""

from dataclasses import replace

import pytest
import xarray as xr
from tollan.accessor import FieldLocation

from tolteca_datamodels.toltec.kids.core import (
    ToltecKidsAccessor,
    ToltecKidsIOMapper,
)
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_native_reader_does_not_search_processing_children(real_tune_file):
    with xr.open_dataset(real_tune_file) as ds:
        tree = xr.DataTree.from_dict({"raw": ds})
        assert ToltecKidsAccessor(tree).data_kind is ToltecDataKind.unknown
        assert ToltecKidsAccessor(tree["raw"]).meta == ds.toltec_kids.meta
        # A child must not override an acquisition at the supplied node.
        tree.dataset = ds.assign({"Header.Toltec.ObsNum": 1})
        meta = ToltecKidsAccessor(tree).meta
        assert isinstance(meta, ToltecRawObsMetadata)
        assert meta.obsnum == 1


@pytest.mark.parametrize("location", [FieldLocation.var, FieldLocation.attr])
def test_lo_access_respects_resolved_location(location):
    ds = xr.Dataset(
        {"Data.Toltec.LoFreq": ("time", [1.0, 2.0])},
        attrs={"chosen": 8.0},
        coords={"chosen": ("time", [3.0, 4.0])},
    )
    mapper = ToltecKidsIOMapper.from_data_source(ds)
    mapper = ToltecKidsIOMapper(
        {
            field: replace(mapped, name="chosen", location=location)
            if field == mapper.schema.f_los
            else mapped
            for field, mapped in mapper.mapped_fields.items()
        }
    )
    reader = ToltecKidsAccessor(ds, mapper=mapper)
    if location == FieldLocation.var:
        frequency = reader.f_lo
        assert isinstance(frequency, xr.DataArray)
        assert frequency.variable is ds["chosen"].variable
    else:
        assert reader.f_lo == 8.0
