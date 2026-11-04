"""KIDs header and structural identification boundaries."""

import subprocess
import sys

import pytest
import xarray as xr
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tolteca_datamodels.toltec.kids.core import ToltecKidsAccessor
from tolteca_datamodels.toltec.types import ToltecDataKind


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", None])
def test_raw_signal_structure_remains_identifiable(data_lmt_path, suffix, monkeypatch):
    # Remove the explicit type from actual payloads to exercise fallback only.
    pattern = "*09_27_57.nc" if suffix is None else f"*1456*_{suffix}.nc"
    source = next((data_lmt_path / "toltec/tcs/toltec0").glob(pattern))

    def unexpected(*args, **kwargs):
        pytest.fail("Structural identification must not read signal/header values")

    with xr.open_dataset(source) as data:
        field = data.toltec_kids.mapper.schema.obs_type_code
        name = data.toltec_kids.mapper.get_name(field)
        without_type = data.drop_vars(name)
        monkeypatch.setattr(NetCDF4ArrayWrapper, "_getitem", unexpected)
        expected = (
            ToltecDataKind.raw_timestream
            if suffix is None
            else ToltecDataKind.raw_sweep
        )
        assert without_type.toltec_kids.data_kind == expected


@pytest.mark.parametrize("as_tree", [False, True])
def test_accessor_classification_is_lazy_and_cached(
    data_lmt_path, monkeypatch, as_tree
):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    getitem = NetCDF4ArrayWrapper._getitem
    reads = []

    def classification_only(array, key):
        reads.append(array.variable_name)
        assert array.variable_name == "Header.Toltec.ObsType"
        return getitem(array, key)

    with xr.open_dataset(path) as data:
        source = xr.DataTree(data) if as_tree else data
        with monkeypatch.context() as patch:
            patch.setattr(NetCDF4ArrayWrapper, "_getitem", classification_only)
            reader = ToltecKidsAccessor(source) if as_tree else source.toltec_kids
            assert reader is not None
            if not as_tree:
                assert reader is source.toltec_kids
            assert reader.data_kind == ToltecDataKind.tune
            # Repeated access must not read the classification header again.
            assert reader.data_kind == ToltecDataKind.tune
        assert reads == ["Header.Toltec.ObsType"]


def test_unknown_accessor_remains_usable():
    data = xr.Dataset()
    reader = data.toltec_kids
    assert reader.data_kind is ToltecDataKind.unknown
    assert data.toltec_kids is reader


@pytest.mark.parametrize(
    ("pattern", "obs_type", "n_blocks", "has_timestream"),
    [("*145609*_tune.nc", "tune", 2, False), ("*_09_27_57.nc", "nominal", 1, True)],
)
def test_open_defers_block_preparation_until_requested(
    data_lmt_path, pattern, obs_type, n_blocks, has_timestream
):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob(pattern))
    script = """
import sys
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_datamodels.toltec.metadata import ToltecRawObsBlocksMetadata
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

read = NetCDF4ArrayWrapper.__getitem__
def guarded(array, key):
    assert array.variable_name not in {"Data.Toltec.Is", "Data.Toltec.Qs"}
    return read(array, key)
NetCDF4ArrayWrapper.__getitem__ = guarded

with open_toltec_kids(sys.argv[1]) as data:
    reader = data.toltec_kids
    assert reader.meta.obs_type == sys.argv[2]
    assert "tolteca_kids" not in sys.modules
    index = reader.block_index
    assert "tolteca_kids.block_identify" in sys.modules
    assert index is reader.block_index
    assert type(index.meta) is ToltecRawObsBlocksMetadata
    assert index.meta.n_blocks == int(sys.argv[3])
    assert index.index.sizes["block"] == int(sys.argv[3])
    selected = index.meta.block_index.data_kind & ToltecDataKind.raw_timestream.value
    assert bool(selected.any()) == (sys.argv[4] == "True")
    assert index.meta.block_index is index.index
    assert len(index.meta.chan_axis_table) == index.chan.sizes["chan_entry"]
    assert len(index.meta.probe_axis_table) == index.probe.sizes["probe"]
    assert not hasattr(index, "blocks")
    assert not hasattr(index.meta, "block")
    assert index.meta.obsnum == reader.meta.obsnum
    assert not data["Data.Toltec.Is"].variable._in_memory
    assert not hasattr(reader, "blocks")
    assert not hasattr(reader, "sweep")
    assert not hasattr(reader, "to_zarr")
"""
    result = subprocess.run(  # noqa: S603 -- Fixed inspection script and reference file.
        [
            sys.executable,
            "-c",
            script,
            str(path),
            obs_type,
            str(n_blocks),
            str(has_timestream),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
