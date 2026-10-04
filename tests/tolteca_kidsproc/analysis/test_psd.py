"""Tests for Welch spectra, explicit sampling rates, units and tree results."""

import subprocess
import sys
import textwrap
from dataclasses import replace

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from astropy import units as u
from dask import delayed
from scipy.signal import welch
from tollan.accessor import AccessorBase, NameMapping, SubpathMapping
from tollan.accessor.xarray import XarrayMapper, prepare_analysis_input_view
from tollan.utils.xarray import reduce_median_mad

from tolteca_kidsproc.analysis import PSDAnalysis, PSDMapper, PSDView


def _timestream(*, n_samples=64, n_chans=None):
    shape = (n_samples,) if n_chans is None else (n_chans, n_samples)
    dims = ("time",) if n_chans is None else ("chan", "time")
    values = np.random.default_rng(42).normal(size=shape)
    return xr.Dataset(
        {"I": (dims, values), "Q": (dims, values * 2)},
        coords={"time": ("time", np.arange(n_samples) / 100, {"units": "s"})},
        attrs={"f_smp": 100.0},
    )


def test_multi_channel_kids_time_axis_through_psd_and_export(tmp_path):
    from tolteca_kidsproc.accessors import make_kids_dataset

    time = np.arange(128) / 64
    frequencies = np.array([4, 8, 12])
    signal = np.exp(2j * np.pi * frequencies[:, None] * time)
    source = make_kids_dataset(
        signal.real, signal.imag, time=time * u.s, f_smp=64 << u.Hz
    )
    result = PSDAnalysis(nperseg=64, compute_common=True)(source.kids.multi_timestream)
    view = result.kids.psd
    expected_frequency, expected_psd = welch(signal.real, fs=64, nperseg=64)
    np.testing.assert_allclose(view.f_psd, expected_frequency)
    np.testing.assert_allclose(view.psd.I.compute(), expected_psd)
    common = np.median(expected_psd, axis=0)
    common_mad = np.median(abs(expected_psd - common), axis=0) * 1.4826
    np.testing.assert_allclose(view.psd_common.I.compute(), common)
    np.testing.assert_allclose(view.psd_common_mad_std.I.compute(), common_mad)
    assert view.psd_common.I.dims == ("f_psd",)
    assert view.psd_median.I.dims == ("chan",)
    assert result.kids.multi_timestream.time.dims == ("time",)

    target = tmp_path / "multi-channel.zarr"
    result.to_zarr(target, zarr_format=2)
    with xr.open_datatree(target, engine="zarr") as reopened:
        input_view = reopened.kids.multi_timestream
        assert input_view.time.shape == (128,)
        assert input_view.I.shape == (3, 128)
        np.testing.assert_array_equal(input_view.time, time)
        np.testing.assert_allclose(reopened.kids.psd.psd.I, expected_psd)
        np.testing.assert_allclose(reopened.kids.psd.psd_common.I, common)
        np.testing.assert_allclose(reopened.kids.psd.psd_common_mad_std.I, common_mad)


def test_low_level_calculations_compose_high_level_outputs():
    source = _timestream(n_samples=128, n_chans=4).chunk({"chan": 2, "time": 32})
    source["I"] = source.I.u.set(u.V)
    source["Q"] = source.Q.u.set(u.V)
    analysis = PSDAnalysis(
        nperseg=32, stat_f_lims=(10 << u.Hz, 40 << u.Hz), compute_common=True
    )
    input_view = source.kids.multi_timestream
    spectra = analysis.make_psd(source, f_smp=input_view.f_smp)
    median, mad = reduce_median_mad(spectra.sel(f_psd=slice(10, 40)), dim="f_psd")
    common, common_mad = reduce_median_mad(spectra, dim="chan")
    result = analysis(input_view).kids.psd
    for actual, expected in (
        (spectra, result.psd),
        (median, result.psd_median),
        (mad, result.psd_mad_std),
        (common, result.psd_common),
        (common_mad, result.psd_common_mad_std),
    ):
        assert isinstance(actual.I.data, da.Array)
        xr.testing.assert_identical(actual.compute(), expected.compute())
        assert actual.I.u.unit == u.V**2 / u.Hz
        assert "tolteca_kidsproc.analysis.psd.analysis" not in actual.attrs


def test_spectra_and_common_calculation_do_not_require_a_statistic_band():
    source = _timestream(n_chans=2)
    analysis = PSDAnalysis(nperseg=16, stat_f_lims=(60 << u.Hz, 70 << u.Hz))
    spectra = analysis.make_psd(source, f_smp=100 << u.Hz)
    assert spectra.sizes["f_psd"] == 9
    common, _ = reduce_median_mad(spectra, dim="chan")
    assert common.sizes["f_psd"] == 9
    with pytest.raises(ValueError, match="no frequency bins"):
        analysis(source.kids.timestream)


def test_common_result_reuse_repair_and_config_change(monkeypatch):
    monkeypatch.setattr(
        PSDMapper,
        "schema",
        replace(PSDMapper.schema, psd_common_subpath=SubpathMapping.path("combined")),
    )
    source = _timestream(n_chans=2)
    analysis = PSDAnalysis(nperseg=16, compute_common=True)
    result = analysis(source.kids.multi_timestream)
    input_view = result.kids.multi_timestream
    saved = result.kids.psd
    assert "combined" in saved.data_source.children
    assert analysis(input_view, output=result) is result
    assert result.kids.psd.data_source is saved.data_source
    del saved.data_source["combined"]
    analysis(input_view, output=result)
    assert result.kids.psd.data_source is not saved.data_source
    assert result.kids.psd.psd_common.I.dims == ("f_psd",)
    saved = result.kids.psd.data_source
    analysis(input_view, output=result, force=True)
    assert result.kids.psd.data_source is not saved
    PSDAnalysis(nperseg=16)(input_view, output=result)
    assert "combined" not in result.kids.psd.data_source.children
    with pytest.raises(KeyError):
        _ = result.kids.psd.psd_common


def test_common_spectrum_uses_the_supplied_view_channel_mapping():
    from tolteca_kidsproc.accessors import make_kids_dataset
    from tolteca_kidsproc.accessors.kids import KidsMapper
    from tolteca_kidsproc.accessors.views import MultiTimestreamView

    time = np.arange(128) / 64
    signal = np.exp(2j * np.pi * np.array([4, 8, 12])[:, None] * time)
    source = make_kids_dataset(
        signal.real, signal.imag, time=time * u.s, f_smp=64 << u.Hz
    )
    mapper = KidsMapper.from_data_source(source)
    mapped_fields = dict(mapper.mapped_fields)
    field = mapper.schema.chan_dim
    mapped_fields[field] = replace(mapped_fields[field], name="detector")
    mapper = KidsMapper(mapped_fields)
    source = source.rename(chan="detector")
    view = MultiTimestreamView(source, mapper=mapper)
    result = PSDAnalysis(nperseg=64, compute_common=True)(view).kids.psd
    _, expected = welch(signal.real, fs=64, nperseg=64)
    np.testing.assert_allclose(result.psd_common.I, np.median(expected, axis=0))
    assert result.psd.I.dims == ("detector", "f_psd")
    assert result.psd_common.I.dims == ("f_psd",)
    assert view.mapper is mapper
    with pytest.raises(ValueError, match="chan"):
        PSDAnalysis(nperseg=64, compute_common=True)(source.kids.timestream)


def test_high_level_uses_all_input_mappings_and_ignores_other_series(monkeypatch):
    from tolteca_kidsproc.accessors.kids import KidsMapper
    from tolteca_kidsproc.accessors.views import MultiTimestreamView

    source = _timestream(n_samples=128, n_chans=3)
    original = source.copy(deep=False)
    source = source.rename(
        {"I": "in_phase", "Q": "quadrature", "time": "tick", "chan": "detector"}
    )
    source = source.assign_coords(clock=source.tick.u.to(u.ms))
    source.attrs = {"recorded_rate": 100.0, "f_smp": -1.0}
    source["I"] = xr.zeros_like(source.in_phase)
    source["Q"] = xr.ones_like(source.quadrature)
    source["unrelated"] = xr.ones_like(source.in_phase)
    source["time"] = xr.DataArray(np.arange(128), dims="tick").u.set(u.s)
    schema = replace(
        KidsMapper.schema,
        I=NameMapping.var("in_phase"),
        Q=NameMapping.var("quadrature"),
        time=NameMapping.var("clock"),
        f_smp=NameMapping.attr("recorded_rate"),
        chan_dim=NameMapping.dim("detector"),
    )
    monkeypatch.setattr(KidsMapper, "schema", schema)
    input_view = MultiTimestreamView(source)
    result = PSDAnalysis(nperseg=32, compute_common=True)(input_view).kids.psd
    assert result.signal_names == ("I", "Q")
    assert result.f_smp == 100
    assert result.psd.I.dims == ("detector", "f_psd")
    for name in ("I", "Q"):
        frequencies, expected = welch(original[name].values, fs=100, nperseg=32)
        np.testing.assert_allclose(result.f_psd, frequencies)
        np.testing.assert_allclose(result.psd[name], expected)
        np.testing.assert_allclose(result.psd_common[name], np.median(expected, axis=0))


@pytest.mark.parametrize("scaling", ["density", "spectrum"])
@pytest.mark.parametrize("unit", [None, u.one, u.mV])
def test_low_level_quantities_and_signal_units(scaling, unit):
    values = np.arange(64.0)
    source = xr.DataArray(
        values,
        dims="time",
        coords={"time": np.arange(64.0) ** 2},
        name="voltage",
        attrs={"f_smp": "invalid"},
    )
    if unit is not None:
        source = source.u.set(unit)
    analysis = PSDAnalysis(nperseg=16, scaling=scaling)
    spectra = analysis.make_psd(source, f_smp=0.1 << u.kHz)
    frequency, expected = welch(values, fs=100, nperseg=16, scaling=scaling)
    np.testing.assert_allclose(spectra.f_psd, frequency)
    np.testing.assert_allclose(spectra.voltage, expected)
    assert spectra.f_psd.u.unit == u.Hz
    assert spectra.attrs["f_smp"] == 100
    expected_unit = u.one if unit is None else unit**2
    if scaling == "density":
        expected_unit /= u.Hz
    assert spectra.voltage.u.unit == expected_unit
    assert source.u.unit == unit


def test_low_level_sampling_frequency_is_required_and_has_frequency_units():
    source = xr.DataArray(np.arange(64.0), dims="time", attrs={"f_smp": 100})
    analysis = PSDAnalysis(nperseg=16)
    with pytest.raises(TypeError, match="f_smp"):
        analysis.make_psd(source)
    with pytest.raises(u.UnitConversionError):
        analysis.make_psd(source, f_smp=100 << u.s)


def test_band_statistics_use_frequency_quantities_and_recover_config():
    analysis = PSDAnalysis(nperseg=16, stat_f_lims=(0.01 << u.kHz, 40 << u.Hz))
    source = _timestream()
    spectra = analysis.make_psd(source, f_smp=0.1 << u.kHz)
    band = spectra.I.sel(f_psd=slice(10, 40)).values
    view = analysis(source.kids.timestream).kids.psd
    np.testing.assert_allclose(view.psd_median.I, np.median(band))
    np.testing.assert_allclose(
        view.psd_mad_std.I, np.median(abs(band - np.median(band))) * 1.4826
    )
    assert view.psd_analysis == analysis
    assert view.psd_analysis.model_dump()["stat_f_lims"] == ("0.01 kHz", "40.0 Hz")


@pytest.mark.parametrize(
    "input_kind", ["array", "dataset", "tree", "prepared", "accessor", "sweep"]
)
def test_high_level_requires_a_timestream_view(input_kind):
    from tolteca_kidsproc.accessors.views import SweepView

    source = _timestream()
    if input_kind == "array":
        supplied = source.I
    elif input_kind == "dataset":
        supplied = source
    elif input_kind == "tree":
        supplied = xr.DataTree.from_dict({"timestream": source})
    elif input_kind == "prepared":
        supplied = prepare_analysis_input_view(source)
    elif input_kind == "sweep":
        supplied = SweepView(
            source.assign_coords(
                frequency=("time", np.arange(64) + 1e9, {"units": "Hz"})
            )
        )
    else:
        supplied = AccessorBase(source, mapper=XarrayMapper.from_data_source(source))
    with pytest.raises(TypeError, match="requires a timestream view"):
        PSDAnalysis()(supplied)


@pytest.mark.parametrize("payload_node", [False, True])
def test_explicit_output_retains_input_and_targets_selected_scope(payload_node):
    from tolteca_kidsproc.accessors.views import TimestreamView

    source = _timestream().chunk({"time": 16})
    supplied = xr.DataTree(dataset=source) if payload_node else source
    before = supplied.copy(deep=False)
    input_view = TimestreamView(supplied)
    analysis = PSDAnalysis(nperseg=16)
    expected = analysis(input_view).kids.psd.psd.compute()
    destination = xr.DataTree.from_dict({"blocks/0": None, "blocks/1": None})
    untouched = destination["blocks/1"].copy(deep=False)
    assert (
        analysis(input_view, output=destination, parent_namespace="blocks/0")
        is destination
    )
    assert set(destination["blocks/0"].children) == {"tolteca_kidsproc.analysis.psd"}
    xr.testing.assert_identical(
        destination["blocks/0"].kids.psd.psd.compute(), expected
    )
    xr.testing.assert_identical(destination["blocks/1"], untouched)
    xr.testing.assert_identical(supplied, before)


def test_timestream_facet_requires_explicit_output_to_modify_its_block():
    source = xr.Dataset(
        {"I": ("time", np.arange(64.0)), "Q": ("time", np.zeros(64))},
        coords={"time": ("time", np.arange(64) / 100, {"units": "s"})},
        attrs={"f_smp": 100.0},
    )
    tree = xr.DataTree.from_dict({"blocks/0/timestream": source})
    block = tree["blocks/0"]
    input_node = block["timestream"]
    facet = block.kids.timestream
    analysis = PSDAnalysis(nperseg=16)
    standalone = analysis(facet)
    assert set(block.children) == {"timestream"}
    assert standalone.kids.timestream.data_source.identical(source)
    assert analysis(facet, output=block) is block
    assert block["timestream"] is input_node
    xr.testing.assert_identical(block.kids.psd.psd, standalone.kids.psd.psd)
    saved = block.kids.psd.data_source
    assert analysis(facet, output=block) is block
    assert block.kids.psd.data_source is saved
    assert analysis(facet, output=block, force=True) is block
    assert block.kids.psd.data_source is not saved


def test_explicit_output_failure_preserves_existing_tree():
    source = _timestream(n_samples=1)
    output = xr.DataTree.from_dict({"other": xr.Dataset(attrs={"keep": True})})
    before = output.copy(deep=False)
    with pytest.raises(ValueError, match="at least two"):
        PSDAnalysis()(source.kids.timestream, output=output)
    xr.testing.assert_identical(output, before)
    with pytest.raises(TypeError, match="output must be a DataTree"):
        PSDAnalysis()(source.kids.timestream, output=xr.Dataset())
    with pytest.raises(TypeError, match="requires an output tree"):
        PSDAnalysis()(source.kids.timestream, parent_namespace="block")


def test_standalone_invocation_without_instrument_imports(tmp_path):
    """A fresh interpreter can analyse a signal using only the public API."""
    script = textwrap.dedent("""
        import sys
        import numpy as np
        import xarray as xr
        from tolteca_kidsproc.analysis import PSDAnalysis

        time = np.arange(1024) / 128.0
        temperature = xr.DataArray(
            np.sin(2 * np.pi * 8 * time), dims="time",
            coords={"time": time}, name="temperature", attrs={"units": "K"},
        )
        from astropy import units as u
        spectra = PSDAnalysis(nperseg=256).make_psd(temperature, f_smp=128 << u.Hz)
        peak = spectra.temperature.argmax(dim="f_psd").compute().item()
        assert spectra.f_psd[peak].item() == 8.0
        assert not any(
            name == package or name.startswith(package + ".")
            for name in sys.modules
            for package in ("tolteca_datamodels", "tolteca_kids")
        )
    """)
    result = subprocess.run(  # noqa: S603 -- fixed test script in this interpreter.
        [sys.executable, "-c", script],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("time_first", [False, True])
def test_arbitrary_signals_match_welch_and_statistics(time_first):
    values = np.random.default_rng(42).normal(size=(3, 128))
    source = xr.Dataset(
        {
            "temperature": (
                ("sensor", "tick"),
                da.from_array(values, chunks=(1, 32)),
                {"units": "K"},
            ),
            "pressure": ("tick", da.from_array(values[0], chunks=32), {"units": "Pa"}),
            "label": ("sensor", ["a", "b", "c"]),
        },
        coords={
            "tick": ("tick", np.arange(128) * 10, {"units": "ms"}),
            "sensor": [10, 0, 5],
        },
    )
    if time_first:
        source = source.transpose("tick", "sensor")
    analysis = PSDAnalysis(
        nperseg=32,
        stat_f_lims=(10.0 << u.Hz, 40.0 << u.Hz),
        batch_size=2,
    )
    result = analysis.make_psd(source, f_smp=100 << u.Hz, dim="tick")
    median, mad = reduce_median_mad(result.sel(f_psd=slice(10, 40)), dim="f_psd")
    assert isinstance(result.temperature.data, da.Array)
    assert "tick" not in result.dims
    assert "label" not in result
    assert result.sensor.values.tolist() == [10, 0, 5]
    for name, expected_input in [("temperature", values), ("pressure", values[0])]:
        frequencies, expected = welch(expected_input, fs=100, nperseg=32)
        np.testing.assert_array_equal(result.f_psd, frequencies)
        np.testing.assert_allclose(result[name], expected, rtol=1e-12)
        mask = (frequencies >= 10) & (frequencies <= 40)
        expected_median = np.median(expected[..., mask], axis=-1)
        expected_mad = (
            np.median(abs(expected[..., mask] - expected_median[..., None]), axis=-1)
            * 1.4826
        )
        np.testing.assert_allclose(median[name], expected_median)
        np.testing.assert_allclose(mad[name], expected_mad)
    assert result.temperature.u.unit == u.K**2 / u.Hz
    assert result.pressure.u.unit == u.Pa**2 / u.Hz
    assert result.attrs["n_samples"] == 128
    assert result.attrs["nperseg"] == 32


def test_analysis_uses_recorded_rate_without_evaluating_timestamps():
    @delayed
    def load_time():
        msg = "PSD must not evaluate timestamps"
        raise AssertionError(msg)

    source = _timestream().rename(time="sample").drop_vars("sample")
    source = source.assign_coords(
        time=(
            "sample",
            da.from_delayed(load_time(), shape=(64,), dtype=float),
            {"units": "ms"},
        )
    )
    view = source.kids.timestream
    assert view.f_smp.unit == u.Hz
    assert view.f_smp.to_value(u.Hz) == pytest.approx(100)
    analysis = PSDAnalysis(nperseg=16)
    tree = analysis(view)
    result = tree.kids.psd
    frequencies, expected = welch(source.I.values, fs=100, nperseg=16)
    np.testing.assert_allclose(result.f_psd, frequencies)
    np.testing.assert_allclose(result.psd.I.compute(), expected)
    assert analysis(view, output=tree) is tree
    assert tree.kids.psd.data_source is result.data_source


def test_timestream_requires_recorded_rate():
    source = _timestream()
    source.attrs.clear()
    with pytest.raises(ValueError, match="Missing required field 'f_smp'"):
        _ = source.kids.timestream


def test_complex_and_real_series_share_two_sided_grid():
    time = np.arange(128) / 100
    source = xr.Dataset(
        {
            "voltage": ("time", np.exp(2j * np.pi * 10 * time)),
            "temperature": ("time", np.sin(2 * np.pi * 5 * time)),
        },
        coords={"time": time},
    )
    result = PSDAnalysis(nperseg=32).make_psd(source, f_smp=100 << u.Hz)
    for name in source:
        frequencies, expected = welch(
            source[name].values, fs=100, nperseg=32, return_onesided=False
        )
        np.testing.assert_allclose(result.f_psd, frequencies)
        np.testing.assert_allclose(result[name], expected, atol=1e-15)


def test_subtree_independence_and_export(tmp_path):
    source = _timestream()
    tree = xr.DataTree.from_dict(
        {"instrument/timestream": source, "elsewhere/_input": xr.Dataset({"value": 7})}
    )
    untouched = tree["elsewhere"].copy(deep=False)
    scope = tree["instrument"]
    analysis = PSDAnalysis(nperseg=16)
    result = analysis(scope.kids.timestream, output=tree, parent_namespace="instrument")
    assert result is tree
    xr.testing.assert_identical(result["elsewhere"], untouched)
    assert scope.kids.psd.psd_analysis == analysis
    assert PSDView.from_namespace(scope.kids.psd.data_source).psd_analysis == analysis
    detached = scope.copy(deep=False)
    assert analysis(detached.kids.timestream, output=detached) is detached
    assert analysis(detached.kids.timestream, output=detached, force=True) is detached
    xr.testing.assert_identical(
        detached.kids.psd.data_source, scope.kids.psd.data_source
    )
    target = tmp_path / "spectra.nc"
    scope.kids.psd.data_source.copy(deep=False).to_netcdf(target)
    with xr.open_datatree(target) as restored:
        view = PSDView.from_namespace(restored)
        assert view.psd_analysis == analysis
        assert view.signal_names == ("I", "Q")
        xr.testing.assert_equal(view.psd, detached.kids.psd.psd)


def test_root_coordinates_do_not_constrain_new_spectra():
    source = (
        _timestream()
        .assign(old_psd=("f_psd", [1.0, 2.0]))
        .assign_coords(f_psd=[0.0, 50.0])
    )
    result = PSDAnalysis(nperseg=16)(source.kids.timestream)
    assert not result.dataset.variables
    xr.testing.assert_identical(result["_input"].to_dataset(), source)
    assert result.kids.psd.f_psd.size == 9
    with pytest.raises(ValueError, match="not an analysis scope"):
        PSDAnalysis(nperseg=16)(
            source.kids.timestream, output=xr.DataTree(dataset=source)
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"nperseg": 1},
        {"noverlap": -1},
        {"nperseg": 16, "noverlap": 16},
        {"stat_f_lims": (5.0 << u.Hz, 1.0 << u.Hz)},
        {"psd_nperseg": 16},
    ],
)
def test_invalid_settings_rejected(settings):
    with pytest.raises(ValueError):
        PSDAnalysis(**settings)


@pytest.mark.parametrize(
    ("problem", "message"),
    [
        ("zero_rate", "positive"),
        ("short", "at least two"),
        ("overlap", "available segment"),
        ("empty_band", "no frequency bins"),
        ("nonnumeric", "numerical"),
    ],
)
def test_high_level_numerical_requirements(problem, message):
    source = _timestream(n_samples=32)
    settings = {"nperseg": 16}
    if problem == "zero_rate":
        source.attrs["f_smp"] = 0
    elif problem == "short":
        source = source.isel(time=slice(0, 1))
    elif problem == "overlap":
        source = source.isel(time=slice(0, 4))
        settings["noverlap"] = 8
    elif problem == "empty_band":
        settings["stat_f_lims"] = (60 << u.Hz, 70 << u.Hz)
    else:
        source["I"] = source.I.astype(str)
    with pytest.raises(ValueError, match=message):
        PSDAnalysis(**settings)(source.kids.timestream)


def test_view_respects_explicit_field_mapping():
    source = _timestream()
    analysis = PSDAnalysis(nperseg=16)
    result = analysis(source.kids.timestream).kids.psd.data_source
    original = PSDMapper.from_namespace(result)
    renamed = result.copy(deep=False)
    renamed["psd"].dataset = renamed["psd"].to_dataset().rename({"f_psd": "frequency"})
    renamed["psd"].attrs["rate"] = renamed["psd"].attrs.pop("f_smp")
    names = {"f_psd": "frequency", "f_smp": "rate"}
    mapper = PSDMapper(
        {
            field: replace(resolved, name=names.get(resolved.name, resolved.name))
            for field, resolved in original.mapped_fields.items()
        },
    )
    view = PSDView(renamed, mapper=mapper)
    assert view.mapper is mapper
    assert view.data_source is renamed
    assert view.psd_analysis == analysis
    assert view.f_smp == 100
    np.testing.assert_array_equal(view.f_psd, result["psd"].dataset.f_psd)
    np.testing.assert_array_equal(view.psd.I, result["psd"].dataset.I_psd)


def test_schema_paths_control_writing_reading_and_detached_export(
    monkeypatch, tmp_path
):
    from tolteca_kidsproc.accessors.kids import KidsMapper

    schema = replace(
        PSDMapper.schema,
        _namespace=SubpathMapping.path("diagnostics/spectrum"),
        psd_subpath=SubpathMapping.path("arrays"),
    )
    monkeypatch.setattr(PSDMapper, "schema", schema)
    monkeypatch.setattr(
        KidsMapper,
        "schema",
        replace(
            KidsMapper.schema,
            timestream_subpath=SubpathMapping.path(("arrays", "_input")),
        ),
    )
    source = _timestream()
    analysis = PSDAnalysis(nperseg=16)
    tree = xr.DataTree.from_dict({"measurement/arrays": source})
    scope = tree["measurement"]
    assert (
        analysis(scope.kids.timestream, output=tree, parent_namespace="measurement")
        is tree
    )
    assert "psd" not in scope.children
    assert analysis(scope.kids.timestream, output=scope) is scope
    expected = analysis(source.kids.timestream).kids.psd.psd
    for view in (
        scope.kids.psd,
        PSDView.from_namespace(scope["diagnostics/spectrum"]),
        PSDView.from_namespace(scope.kids.psd.data_source),
        scope.copy(deep=False).kids.psd,
    ):
        xr.testing.assert_equal(view.psd, expected)
    target = tmp_path / "custom.zarr"
    scope.copy(deep=False).to_zarr(target)
    with xr.open_datatree(target, engine="zarr") as reopened:
        xr.testing.assert_equal(reopened.kids.psd.psd, expected)


def test_view_respects_explicit_subtree_mapping():
    result = PSDAnalysis(nperseg=16)(_timestream().kids.timestream)
    result = result.kids.psd.data_source
    original = PSDMapper.from_namespace(result)
    schema = original.schema
    names = {
        schema._namespace: "diagnostics/spectrum",
        schema.psd_subpath: "arrays",
    }
    mapper = PSDMapper(
        {
            field: replace(
                resolved,
                name=names.get(field, resolved.name),
                parent_path="arrays"
                if resolved.parent_path == "psd"
                else resolved.parent_path,
            )
            for field, resolved in original.mapped_fields.items()
        },
    )
    source = xr.DataTree.from_dict(
        {
            "diagnostics/spectrum": xr.Dataset(attrs=result.attrs),
            "diagnostics/spectrum/arrays": result["psd"].to_dataset(),
        }
    )
    view = PSDView(source, mapper=mapper)
    xr.testing.assert_equal(view.psd, PSDView.from_namespace(result).psd)


@pytest.mark.parametrize("missing", ["config", "frequency", "spectra"])
def test_view_requires_analysis_results(missing):
    result = PSDAnalysis(nperseg=16)(_timestream().kids.timestream)
    result = result.kids.psd.data_source
    if missing == "config":
        result.attrs.clear()
    elif missing == "frequency":
        result["psd"].dataset = result["psd"].to_dataset().drop_vars("f_psd")
    else:
        result["psd"].dataset = result["psd"].to_dataset().drop_vars(["I_psd", "Q_psd"])
    with pytest.raises(ValueError):
        PSDView.from_namespace(result)


def test_module_workflow_examples():
    import doctest

    from tolteca_kidsproc.analysis import psd

    result = doctest.testmod(psd, raise_on_error=True)
    assert result.attempted > 0


def test_exported_input_view_supports_recomputation(tmp_path):
    source = _timestream().chunk({"time": 16})
    analysis = PSDAnalysis(nperseg=16)
    direct = analysis(source.kids.timestream)
    target = tmp_path / "wrapped-input.zarr"
    direct.to_zarr(target)
    with xr.open_datatree(target, engine="zarr") as reopened:
        result = analysis(reopened.kids.timestream, output=reopened, force=True)
        assert result is reopened
        xr.testing.assert_identical(
            result.kids.psd.data_source, direct.kids.psd.data_source
        )
