"""The NetCDF model representation transports an ECSV report without loss."""

import numpy as np
import pytest
import xarray as xr
from astropy.table import Table

from tolteca_datamodels.toltec.kids.model_table import (
    pack_model_tables,
    read_model_table,
)


def raw():
    return xr.Dataset({"Header.Toltec.ToneFreq": (("block", "chan"), np.zeros((2, 2)))})


def test_complete_header_and_extra_columns_survive_per_block_roundtrip(tmp_path):
    first = Table(
        {"chan": [4, 8], "flag": [1 << 22, 1 << 13], "future": [1.25, np.nan]}
    )
    first["future"].unit = "Hz"
    first["future"].description = "A future scalar model column"
    first.meta = {
        "arbitrary": {"groups": [{"ids": ["a", "b", "c"], "a_nl": [0.1, 0.2, 0.3]}]},
        "Header.Toltec.AccumLen": 524288,
        "flag_definitions": {"fit_joint_required": 1 << 22},
    }
    second = Table(
        {"chan": [4, 8], "flag": [0, 0], "other": [2.0, 3.0]},
        meta={"label": "other block"},
    )
    base = raw()
    packed = pack_model_tables(base, {0: first, 1: second})
    path = tmp_path / "stub.nc"
    packed.to_netcdf(path)
    with xr.open_dataset(path) as reopened:
        for block, expected in enumerate((first, second)):
            actual = read_model_table(reopened, block=block)
            assert actual.meta == expected.meta
            assert actual.colnames == expected.colnames
            for name in actual.colnames:
                np.testing.assert_array_equal(actual[name], expected[name])
                assert actual[name].dtype == expected[name].dtype
                assert actual[name].unit == expected[name].unit
                assert actual[name].description == expected[name].description
        replacement = pack_model_tables(base.merge(reopened), {1: first})
        assert read_model_table(replacement, block=0).meta == first.meta
        assert read_model_table(replacement, block=1).meta == first.meta
    assert "Header.Toltec.ModelParamsAccumLen" not in packed


@pytest.mark.parametrize("values", [[2**53 + 1, 0], ["text", "value"]])
def test_nonrepresentable_column_is_rejected(values):
    table = Table({"chan": [0, 1], "value": values})
    with pytest.raises(ValueError, match="representable|numeric"):
        pack_model_tables(raw(), {0: table})
