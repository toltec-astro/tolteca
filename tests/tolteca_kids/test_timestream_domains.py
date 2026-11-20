"""Python time/PSD views preserve reference-file identity, units and lazy I/O."""

import numpy as np
import pytest
import xarray as xr
from astropy import units as u
from scipy.signal import welch
from tollan.accessor.xarray import ensure_datatree
from xarray.backends.netCDF4_ import NetCDF4ArrayWrapper

from tests.kids_helpers import apply_step
from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_kids import BlockCreate, TimestreamReduce, TimestreamView
from tolteca_kidsproc.analysis import PSDAnalysis


def _source(data_lmt_path):
    return next((data_lmt_path / "toltec/tcs/toltec0").glob("*_09_27_57.nc"))


def test_raw_time_view_borrows_arrays(data_lmt_path, monkeypatch):
    original = NetCDF4ArrayWrapper.__getitem__

    def guarded(array, key):
        assert array.variable_name not in {"Data.Toltec.Is", "Data.Toltec.Qs"}
        return original(array, key)

    with open_toltec_kids(_source(data_lmt_path)) as source:
        monkeypatch.setattr(NetCDF4ArrayWrapper, "__getitem__", guarded)
        ds = source.toltec_kids.select_channels([10, 0, 5])
        view = BlockCreate()(ensure_datatree(ds)).toltec_kids.timestream
        assert view.meta.block == 0
        assert view.I.dims == ("chan", "time")
        assert view.chan.values.tolist() == [10, 0, 5]
        assert view.sample.values.tolist() == list(range(7802))
        assert not view.I.variable._in_memory
        assert not hasattr(view, "time_domain")
        assert not hasattr(view, "psd_domain")
        assert view.I_psd is None
        reduced = apply_step(TimestreamReduce(psd={"nperseg": 128}), ds)
        assert not reduced.toltec_kids.timestream.I_psd.variable._in_memory
        assert view.time.attrs["units"] == "s"
        np.testing.assert_allclose(np.diff(view.time), 1 / 122.0703125)
        assert float(view.time[-1]) < 64


@pytest.mark.parametrize("orientation", ["channel_first", "time_first"])
def test_welch_signals_and_units_match_scipy(data_lmt_path, orientation):
    with open_toltec_kids(_source(data_lmt_path), chunks={}) as source:
        selected = (
            source.assign_coords(time=np.arange(source.sizes["time"]))
            .isel(time=slice(100, 700))
            .toltec_kids.select_channels([10, 0, 5])
        )
        tree = BlockCreate()(ensure_datatree(selected))
        raw = tree.toltec_kids.timestream.data_source
        block = tree.toltec_kids.blocks[0].detach(include_raw=False)
        if orientation == "time_first":
            raw = raw.transpose("time", "chan")
        raw.I.attrs["units"] = "adu"
        raw.Q.attrs["units"] = "adu"
        reducer = TimestreamReduce(
            psd={"nperseg": 128, "stat_f_lims": (10.0 << u.Hz, 30.0 << u.Hz)}
        )
        block["timestream"] = xr.DataTree(raw)
        result = apply_step(reducer, block)
        view = TimestreamView(result)
        assert view.sample.values[0] == 100
        assert view.chan.values.tolist() == [10, 0, 5]
        assert not view.I_psd.variable._in_memory
        I, Q = (  # noqa: E741 -- scientific I/Q labels
            raw[name].transpose("chan", "time").values for name in ("I", "Q")
        )
        for name, signal in {
            "I": I,
            "Q": Q,
            "amplitude": np.hypot(I.astype(float), Q.astype(float)),
            "phase": np.unwrap(np.arctan2(Q, I), axis=-1),
        }.items():
            frequencies, expected = welch(signal, fs=raw.attrs["f_smp"], nperseg=128)
            actual = getattr(view, f"{name}_psd")
            np.testing.assert_array_equal(view.f_psd, frequencies)
            np.testing.assert_allclose(actual, expected, rtol=1e-12)
            expected_unit = u.rad**2 / u.Hz if name == "phase" else u.adu**2 / u.Hz
            assert u.Unit(actual.attrs["units"]) == expected_unit
        xr.testing.assert_equal(view.I, raw.I)
        np.testing.assert_allclose(view.S21, raw.I + 1j * raw.Q)
        np.testing.assert_allclose(raw.kids.multi_timestream.S21, view.S21)


@pytest.mark.parametrize("include_raw", [True, False])
@pytest.mark.parametrize("scope", ["acquisition", "block"])
def test_timestream_domains_roundtrip(data_lmt_path, tmp_path, include_raw, scope):
    target = tmp_path / "time.zarr"
    with open_toltec_kids(_source(data_lmt_path)) as source:
        ds = source.isel(time=slice(0, 512)).toltec_kids.select_channels([10, 5])
        reducer = TimestreamReduce(psd={"nperseg": 128, "compute_common": True})
        result = apply_step(reducer, ds)
        if scope == "block":
            result = result["blocks/0"]
        view = result.toltec_kids.timestream
        expected = view.I_psd.compute()
        common = view.psd.psd_common.compute()
        raw = view.I.compute()
        if not include_raw:
            detached = result.toltec_kids.detach(include_raw=False)
            assert detached.toltec_kids.timestream.I is None
            xr.testing.assert_equal(
                detached.toltec_kids.timestream.I_psd.compute(), expected
            )
        result.toltec_kids.to_zarr(target, include_raw=include_raw)
    with xr.open_datatree(target, engine="zarr", chunks={}) as tree:
        view = tree.toltec_kids.timestream
        xr.testing.assert_equal(view.I_psd, expected)
        assert view.psd.psd_analysis == reducer.config.psd
        xr.testing.assert_equal(view.psd.psd_common, common)
        assert not any("I" in node.data_vars for node in tree.subtree)
        if include_raw:
            xr.testing.assert_equal(view.I, raw)
        else:
            assert view.I is None
            assert view.time is None
        assert view.chan.values.tolist() == [10, 5]
        raw_leaves = [node for node in tree.subtree if "Data.Toltec.Is" in node.dataset]
        assert len(raw_leaves) == int(include_raw)


@pytest.mark.parametrize(
    "problem", ["negative_rate", "zero_rate", "no_bins", "overlap"]
)
def test_spectral_invalid_inputs_are_explicit(problem):
    ds = xr.Dataset(
        {"I": ("time", np.arange(32.0)), "Q": ("time", np.ones(32))},
        coords={"time": np.arange(32.0) / 100},
        attrs={"f_smp": 100.0},
    )
    ds["time"] = ds.time.u.set("s")
    settings = {"nperseg": 16}
    if problem == "negative_rate":
        ds.attrs["f_smp"] = -100
    elif problem == "zero_rate":
        ds.attrs["f_smp"] = 0
    elif problem == "no_bins":
        settings["stat_f_lims"] = (60.0 << u.Hz, 80.0 << u.Hz)
    else:
        ds = ds.isel(time=slice(0, 4))
        settings["noverlap"] = 8
    with pytest.raises(ValueError):
        PSDAnalysis(**settings)(ds.kids.timestream)


def test_spectrum_scaling_and_custom_time_axis():
    ds = xr.Dataset({"I": ("tick", np.arange(32.0)), "Q": ("tick", np.ones(32))})
    spectra = PSDAnalysis(scaling="spectrum", nperseg=16).make_psd(
        ds, f_smp=100 << u.Hz, dim="tick"
    )
    _, expected = welch(ds.I.values, fs=100, nperseg=16, scaling="spectrum")
    np.testing.assert_allclose(spectra.I, expected)
    assert spectra.I.u.unit == u.one


def test_explicit_mapper_is_respected_by_time_and_psd_views():
    from dataclasses import replace

    from tolteca_kids.timestream import ToltecTimestreamMapper

    source = xr.Dataset(
        {
            "I": ("time", [1.0, 2.0]),
            "Q": ("time", [3.0, 4.0]),
            "I_psd": ("f_psd", [5.0, 6.0]),
            "f_psd": ("f_psd", [0.0, 1.0]),
        },
        coords={"time": [0.0, 1.0]},
    )
    mapping = {
        "I": "in_phase",
        "Q": "quadrature",
        "I_psd": "power",
        "f_psd": "frequency",
    }
    original = ToltecTimestreamMapper.from_data_source(source)
    mapper = ToltecTimestreamMapper(
        {
            field: replace(resolved, name=mapping.get(resolved.name, resolved.name))
            for field, resolved in original.mapped_fields.items()
        }
    )
    view = TimestreamView(source.rename(mapping), mapper=mapper)
    assert view.mapper is mapper
    np.testing.assert_array_equal(view.I, [1.0, 2.0])
    np.testing.assert_array_equal(view.Q, [3.0, 4.0])
    np.testing.assert_array_equal(view.I_psd, [5.0, 6.0])
    np.testing.assert_array_equal(view.f_psd, [0.0, 1.0])
    raw = TimestreamView(source[["I", "Q"]], ToltecTimestreamMapper.from_defaults())
    assert raw.I_psd is None
    assert raw.f_psd is None


def test_reanalyze_selection_from_unified_view(data_lmt_path):
    with open_toltec_kids(_source(data_lmt_path), chunks={}) as source:
        native = source.toltec_kids.select_channels([10, 5])
        first = apply_step(TimestreamReduce(psd={"nperseg": 256}), native)
        selected = first.toltec_kids.timestream.data_source.isel(time=slice(100, 700))
        second = PSDAnalysis(nperseg=128)(selected.kids.timestream)
        view = TimestreamView(second)
        frequencies, expected = welch(selected.I.values, fs=122.0703125, nperseg=128)
        np.testing.assert_array_equal(view.f_psd, frequencies)
        np.testing.assert_allclose(view.I_psd, expected, rtol=1e-12)
        np.testing.assert_array_equal(view.I, selected.I)
        assert int(view.sample[0]) == 100
        assert first.toltec_kids.timestream.f_psd.size == 129


def test_flat_psd_settings_are_rejected():
    with pytest.raises(ValueError, match="Extra inputs"):
        TimestreamReduce(psd_nperseg=128)


def test_prepared_signals_and_shared_psd_include_selected_observables(data_lmt_path):
    from tollan.utils.xarray import reduce_median_mad

    from tolteca_kidsproc.analysis import PSDView

    block = _prepared_signals(data_lmt_path, n_samples=128, channels=[10, 0, 5])
    source = block["timestream"].to_dataset().rename(chan="n_chans")
    source["r"] = (source.I / 10).u.set(u.one)
    source["x"] = (source.Q / 20).u.set(u.one)
    block["timestream"] = xr.DataTree(source)
    reducer = TimestreamReduce(psd={"nperseg": 32, "compute_common": True})
    result = apply_step(reducer, block)
    time_view = result.toltec_kids.timestream
    psd = result.kids.psd
    assert TimestreamReduce.get_context(result).config == reducer.config
    assert time_view.psd.psd_analysis == reducer.config.psd
    assert psd.signal_names == ("I", "Q", "r", "x", "amplitude", "phase")
    assert result.kids.multi_timestream.n_chans == 3
    assert result.kids.timestream.time.dims == ("time",)
    assert list(result["timestream"].data_vars) == ["I", "Q", "r", "x"]
    expected_median, expected_mad = reduce_median_mad(psd.psd, dim="chan")
    xr.testing.assert_identical(psd.psd_common, expected_median)
    xr.testing.assert_identical(psd.psd_common_mad_std, expected_mad)
    xr.testing.assert_identical(time_view.I_psd, psd.psd.I.rename("I_psd"))
    restored = PSDView.from_namespace(psd.data_source.copy(deep=False))
    assert restored.psd_analysis == reducer.config.psd
    assert restored.signal_names == psd.signal_names


def test_reduction_reuse_failure_and_replacement_preserve_scope(data_lmt_path):
    tree = xr.DataTree()
    tree["blocks/0"] = _prepared_signals(data_lmt_path, n_samples=64, channels=[0, 5])
    tree["blocks/0/other"] = xr.DataTree(dataset=xr.Dataset({"value": 7}))
    reducer = TimestreamReduce(psd={"nperseg": 16})
    assert apply_step(reducer, tree, parent_namespace="blocks/0") is tree
    block = tree["blocks/0"]
    old = block.toltec_kids.timestream
    assert apply_step(reducer, block) is block
    assert block.kids.psd.data_source is old.psd.data_source
    invalid = TimestreamReduce(
        psd={"nperseg": 16, "stat_f_lims": (70 << u.Hz, 80 << u.Hz)}
    )
    with pytest.raises(ValueError, match="no frequency bins"):
        apply_step(invalid, block)
    assert block.kids.psd.data_source is old.psd.data_source
    assert not TimestreamReduce.get_context(block).completed
    replacement = TimestreamReduce(psd={"nperseg": 32}, compute_amplitude_phase=False)
    apply_step(replacement, block)
    assert old.f_psd.size == 9
    assert block.toltec_kids.timestream.f_psd.size == 17
    assert block.kids.psd.signal_names == ("I", "Q")
    assert block["other/value"].item() == 7
    apply_step(TimestreamReduce(psd=None), block)
    assert block.toltec_kids.timestream.psd is None
    assert block.toltec_kids.timestream.I_psd is None
    assert block.kids.timestream.I.shape == (2, 64)


def test_reduction_retains_solved_samples_when_their_spectra_are_disabled(
    data_lmt_path,
):
    block = _prepared_signals(data_lmt_path, n_samples=32, channels=[0])
    source = block["timestream"].to_dataset()
    source["r"] = source.I / 2
    source["x"] = source.Q / 2
    block["timestream"] = xr.DataTree(source)
    result = apply_step(
        TimestreamReduce(
            psd={"nperseg": 16}, compute_r_x=False, compute_amplitude_phase=False
        ),
        block,
    )
    xr.testing.assert_equal(result.toltec_kids.timestream.r, source.r)
    xr.testing.assert_equal(result.toltec_kids.timestream.x, source.x)
    assert result.kids.psd.signal_names == ("I", "Q")


def test_force_reduction_reads_current_raw_block_samples(data_lmt_path):
    with open_toltec_kids(_source(data_lmt_path)) as source:
        selected = source.isel(time=slice(0, 256)).toltec_kids.select_channels([0, 5])
        tree = BlockCreate()(ensure_datatree(selected))
        reducer = TimestreamReduce(psd={"nperseg": 64})
        apply_step(reducer, tree)
        before = tree.toltec_kids.timestream.I_psd.compute()
        tree["raw"]["Data.Toltec.Is"] = tree["raw"]["Data.Toltec.Is"] * 2
        apply_step(reducer, tree, force=True)
        np.testing.assert_allclose(tree.toltec_kids.timestream.I_psd, before * 4)
        np.testing.assert_array_equal(
            tree["blocks/0"].kids.timestream.I,
            selected["Data.Toltec.Is"].values.T * 2,
        )


def _prepared_signals(data_lmt_path, *, n_samples, channels):
    """A detached reference block with materialized samples in its result leaf."""
    with open_toltec_kids(_source(data_lmt_path)) as source:
        selected = source.isel(time=slice(0, n_samples)).toltec_kids.select_channels(
            channels
        )
        tree = BlockCreate()(ensure_datatree(selected))
        signals = tree.toltec_kids.timestream.data_source.load()
        block = tree.toltec_kids.blocks[0].detach(include_raw=False)
        block["timestream"] = xr.DataTree(signals)
        return block
