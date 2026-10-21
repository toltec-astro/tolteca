"""Exported Zarr schema validation is lazy and preserves resource ownership."""

import dask.array as da
import pytest
import xarray as xr
from dask import delayed

from tolteca_kids._zarr_input import (
    ToltecKidsZarrMapper,
    open_zarr_dataset,
)


@pytest.fixture
def exported():
    @delayed
    def unread():
        pytest.fail("Zarr validation must not read detector arrays")

    payload = da.from_delayed(unread(), shape=(2, 3), dtype=float)
    return xr.Dataset(
        {
            "I": (("chan", "sample"), payload),
            "Q": (("chan", "sample"), payload),
            "tone_freq": ("chan", [1.0, 2.0]),
        },
        coords={"lo_freq": ("sample", [10.0, 11.0, 12.0])},
        attrs={"lo_center_freq_hz": 11.0},
    )


def test_open_validates_without_loading_or_closing(exported, monkeypatch):
    closed = []
    exported.set_close(lambda: closed.append(True))
    monkeypatch.setattr(xr, "open_zarr", lambda path: exported)
    ds = open_zarr_dataset("export.zarr")
    assert ds is exported
    assert not closed
    assert not ds.I.variable._in_memory
    ds.close()
    assert closed == [True]


@pytest.mark.parametrize(
    "name", ["I", "Q", "tone_freq", "lo_freq", "lo_center_freq_hz"]
)
def test_missing_required_field_closes_rejected_store(exported, monkeypatch, name):
    if name in exported.attrs:
        del exported.attrs[name]
    else:
        exported = exported.drop_vars(name)
    closed = []
    exported.set_close(lambda: closed.append(True))
    monkeypatch.setattr(xr, "open_zarr", lambda path: exported)
    with pytest.raises(ValueError, match=name):
        open_zarr_dataset("export.zarr")
    assert closed == [True]


@pytest.mark.parametrize("name", ["I", "Q", "tone_freq", "lo_freq"])
def test_variable_and_coordinate_roles_are_preserved(exported, name):
    wrong = (
        exported.reset_coords(name) if name == "lo_freq" else exported.set_coords(name)
    )
    with pytest.raises(ValueError, match=name):
        ToltecKidsZarrMapper.from_data_source(wrong).validate(wrong)
