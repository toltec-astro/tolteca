"""Discovered probe bounds and independent result frequency coordinates."""

import dask
import dask.array as da
import pytest
import xarray as xr
from tollan.accessor.xarray import ensure_datatree

from tests.kids_helpers import apply_step
from tolteca_kids import BlockCreate, ReducedSweepView, SweepReduce


def test_discovery_reads_lo_once_and_reduction_uses_probe_bounds(data_lmt_path):
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    calls = []
    with xr.open_dataset(path) as original:
        raw = original.toltec_kids.select_channels([0, 1]).load()
    values = raw["Data.Toltec.LoFreq"].values.copy()

    @dask.delayed
    def read_lo():
        calls.append("LO")
        return values

    raw["Data.Toltec.LoFreq"] = (
        "time",
        da.from_delayed(
            read_lo(),
            shape=values.shape,
            dtype=values.dtype,
        ),
    )
    with dask.config.set(scheduler="synchronous"):
        tree = BlockCreate()(ensure_datatree(raw))
        blocks = tree.toltec_kids.blocks
        assert calls == ["LO"]
        tree = apply_step(SweepReduce(), tree)
        assert calls == ["LO"]
        assert tree.toltec_kids.blocks.n_blocks == blocks.n_blocks == 2
        # Original acquisition sample bounds also work after detaching block 1.
        detached = blocks[1].detach()
        apply_step(SweepReduce(), detached, force=True)
        assert calls == ["LO"]
        xr.testing.assert_identical(detached.toltec_kids.sweep.I, blocks[1].sweep.I)


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune"])
def test_result_channel_frequency_is_independently_selectable(data_lmt_path, suffix):
    source = next((data_lmt_path / "toltec/tcs/toltec0").glob(f"*1456*_{suffix}.nc"))
    with xr.open_dataset(source) as raw:
        tree = apply_step(SweepReduce(), raw)
        for block in tree.toltec_kids.blocks:
            view = block.sweep
            frequencies = view.f_chan
            field = view.mapper.schema.f_chan
            assert view.mapper.get_name(field) in view.data_source.coords
            chan_dim = frequencies.dims[-1]
            assert chan_dim == view.I.dims[-2]
            selected = ReducedSweepView(view.data_source.isel({chan_dim: [3, 0, 3]}))
            xr.testing.assert_identical(
                selected.frequency, view.frequency.isel({chan_dim: [3, 0, 3]})
            )
            assert selected.f_chan.u.unit_str == "Hz"
            xr.testing.assert_identical(view.f_chan, frequencies)


def test_block_averaging_reads_each_signal_once(data_lmt_path, monkeypatch):
    from collections import Counter

    from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

    calls = Counter()
    original = NetCDF4ArrayWrapper.__getitem__

    def read(array, key):
        calls[array.variable_name] += 1
        return original(array, key)

    monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", read)
    path = next((data_lmt_path / "toltec/tcs/toltec0").glob("*145609*_tune.nc"))
    with xr.open_dataset(path) as source:
        raw = source.toltec_kids.select_channels([0, 1])
        tree = BlockCreate()(ensure_datatree(raw))
        assert calls["Data.Toltec.LoFreq"] == 1
        assert calls["Data.Toltec.Is"] == calls["Data.Toltec.Qs"] == 0
        reducer = SweepReduce()
        apply_step(reducer, tree)
        assert calls["Data.Toltec.LoFreq"] == 1
        assert calls["Data.Toltec.Is"] == calls["Data.Toltec.Qs"] == 2
        apply_step(reducer, tree)
        assert calls["Data.Toltec.Is"] == calls["Data.Toltec.Qs"] == 2
        apply_step(reducer, tree, force=True)
        assert calls["Data.Toltec.LoFreq"] == 1
        assert calls["Data.Toltec.Is"] == calls["Data.Toltec.Qs"] == 4
