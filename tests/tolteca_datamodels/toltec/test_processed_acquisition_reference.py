"""Recover the acquisition link stored by both native KIDs writers."""

import numpy as np
import pytest
import xarray as xr

from tolteca_datamodels.file import FileAccessor
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.mark.parametrize(
    "legacy", [True, False], ids=["legacy-netcdf", "kids-reduction-1"]
)
def test_processed_acquisition_reference_survives_renaming(tmp_path, legacy):
    parent = "/unmounted/toltec0_145609_000_0001_2025_10_31_09_26_59_tune.nc"
    if legacy:
        # The writer's fixed-width C strings may contain arbitrary bytes after NUL.
        data = xr.Dataset(
            {
                "Header.Kids.kind": np.bytes_(b"processed_sweep\x00\xff"),
                "Header.Kids.source": np.bytes_(parent.encode() + b"\x00\xff"),
            }
        )
    else:
        data = xr.Dataset(
            attrs={
                "contract": "kids.reduction/1",
                "kind": "ReducedSweep",
                "source": parent,
            }
        )
    path = tmp_path / "renamed.nc"
    data.to_netcdf(path)
    with open_toltec_kids(path) as data:
        assert data.toltec_kids.meta.filepath_orig == parent
        assert data.toltec_kids.meta.filepath_info_orig.obsnum == 145609
    table = FileAccessor.from_sources(
        [path], mode="guess_or_read", on_error="raise"
    ).table
    assert table.filepath_orig.tolist() == [parent]
    assert table.data_kind_toltec.tolist() == [ToltecDataKind.reduced_sweep.value]
    assert table.data_models.tolist() == [frozenset({"toltec"})]
