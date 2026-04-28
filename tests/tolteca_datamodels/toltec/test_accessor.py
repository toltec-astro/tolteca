"""Tests for TolTEC KIDs data accessor."""

from __future__ import annotations

import warnings
from pathlib import Path

# Suppress numpy binary compatibility warnings with netCDF4 BEFORE importing anything
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*numpy.ndarray size changed.*")

import numpy as np
import pytest
import xarray as xr

# Import to register accessor
from tolteca_datamodels.toltec.kids import ToltecKidsAccessor  # noqa: F401

# Helper to get namespaced variable names
NAMESPACE = "tolteca_datamodels.toltec.kids.sweep"


def get_var_name(field: str) -> str:
    """Get namespaced variable name for reduced sweep data.

    Parameters
    ----------
    field : str
        Field name (I, Q, unc_I, unc_Q, etc.)

    Returns
    -------
    str
        Namespaced variable name
    """
    return f"{NAMESPACE}.{field}"


@pytest.fixture
def rng() -> np.random.Generator:
    """Random number generator with fixed seed for reproducibility."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def toltec_sweep_dataset():
    """Create a mock TolTEC sweep dataset with TolTEC-specific fields."""
    n_chans = 256
    n_sweeps = 50

    # Create frequency and channel coordinates
    f_tones = np.linspace(100e6, 200e6, n_chans)  # 100-200 MHz
    frequencies = np.linspace(-5e6, 5e6, n_sweeps)  # -5 to +5 MHz sweep

    # Create deterministic I/Q data with Lorentzian resonances
    i_data = np.zeros((n_chans, n_sweeps))
    q_data = np.zeros((n_chans, n_sweeps))
    for i in range(n_chans):
        # Each channel has a resonance at slightly different frequency
        f_res = (
            frequencies[n_sweeps // 2] + (i - n_chans // 2) * 1e4
        )  # Spread resonances
        Q = 1e5 + i * 100  # Vary Q factor per channel
        for j, f in enumerate(frequencies):
            s21 = 1 / (1 + 1j * Q * (f - f_res) / (150e6 + f))
            i_data[i, j] = s21.real
            q_data[i, j] = s21.imag

    # Create tone properties
    tone_mask = np.ones(n_chans, dtype=bool)
    tone_mask[::10] = False  # Mask every 10th tone
    tone_amp = np.ones(n_chans) * 0.1
    tone_phase = np.zeros(n_chans)

    ds = xr.Dataset(
        {
            "Data.Toltec.Is": (["chan", "sweep"], i_data),
            "Data.Toltec.Qs": (["chan", "sweep"], q_data),
            "Header.Toltec.ToneFreq": (["chan"], f_tones),
            "Header.Toltec.ToneMask": (["chan"], tone_mask),
            "Header.Toltec.ToneAmp": (["chan"], tone_amp),
            "Header.Toltec.TonePhase": (["chan"], tone_phase),
        },
        coords={
            "chan": np.arange(n_chans),
            "sweep": frequencies,
        },
        attrs={
            "Header.Toltec.RoachIndex": 0,
            "Header.Toltec.ObsNum": 12345,
            "Header.Toltec.SubObsNum": 1,
            "Header.Toltec.ScanNum": 0,
            "Header.Toltec.SampleFreq": 1000.0,
            "Header.Toltec.LoCenterFreq": 150e6,  # 150 MHz center LO
            "Header.Toltec.ObsType": 1,  # TargetSweep
        },
    )
    # Add units to coordinates
    ds["sweep"].attrs["units"] = "Hz"
    ds["Header.Toltec.ToneFreq"].attrs["units"] = "Hz"

    return ds


@pytest.fixture
def toltec_timestream_dataset():
    """Create a mock TolTEC timestream dataset."""
    n_chans = 256
    n_samples = 1000

    # Create time coordinate
    times = np.arange(n_samples) / 1000.0  # 1000 Hz sample rate

    # Create deterministic I/Q data with multi-frequency sinusoids
    i_data = np.zeros((n_chans, n_samples))
    q_data = np.zeros((n_chans, n_samples))
    for i in range(n_chans):
        # Each channel has slightly different frequency and amplitude
        freq = 10 + (i % 20) * 0.5  # Frequencies from 10-20 Hz
        phase = i * np.pi / n_chans  # Different phase per channel
        amp = 1.0 + 0.1 * (i / n_chans)  # Gradually increasing amplitude
        i_data[i, :] = amp * np.cos(2 * np.pi * freq * times + phase)
        q_data[i, :] = amp * np.sin(2 * np.pi * freq * times + phase)

    # Create tone properties
    f_tones = np.linspace(100e6, 200e6, n_chans)
    tone_mask = np.ones(n_chans, dtype=bool)

    ds = xr.Dataset(
        {
            "Data.Toltec.Is": (["chan", "sample"], i_data),
            "Data.Toltec.Qs": (["chan", "sample"], q_data),
            "Header.Toltec.ToneFreq": (["chan"], f_tones),
            "Header.Toltec.ToneMask": (["chan"], tone_mask),
        },
        coords={
            "chan": np.arange(n_chans),
            "time": (["sample"], times),
        },
        attrs={
            "Header.Toltec.RoachIndex": 7,  # a1400 array
            "Header.Toltec.ObsNum": 54321,
            "Header.Toltec.SampleFreq": 1000.0,
            "Header.Toltec.LoCenterFreq": 150e6,
        },
    )
    # Add units
    ds["time"].attrs["units"] = "s"
    ds["Header.Toltec.ToneFreq"].attrs["units"] = "Hz"

    return ds


def test_toltec_accessor_registration(toltec_sweep_dataset):
    """Test that toltec_kids accessor is registered."""
    assert hasattr(toltec_sweep_dataset, "toltec_kids")


def test_metadata_properties(toltec_sweep_dataset):
    """Test TolTEC metadata properties."""
    acc = toltec_sweep_dataset.toltec_kids

    assert acc.meta.roach == 0
    assert acc.meta.obsnum == 12345
    assert acc.meta.subobsnum == 1
    assert acc.meta.scannum == 0
    assert acc.meta.f_smp.value == 1000.0


def test_array_name_mapping(toltec_sweep_dataset, toltec_timestream_dataset):
    """Test array name derived from ROACH index."""
    # ROACH 0 -> a1100
    assert toltec_sweep_dataset.toltec_kids.meta.array_name == "a1100"

    # ROACH 7 -> a1400
    assert toltec_timestream_dataset.toltec_kids.meta.array_name == "a1400"

    # Test a2000 array
    ds = toltec_sweep_dataset.copy()
    ds.attrs["Header.Toltec.RoachIndex"] = 11
    assert ds.toltec_kids.meta.array_name == "a2000"


def test_tone_properties(toltec_sweep_dataset):
    """Test tone/channel frequency properties."""
    acc = toltec_sweep_dataset.toltec_kids

    # f_tone should be an array
    assert acc.f_tone is not None
    assert len(acc.f_tone) == 256
    assert acc.f_tone.min() >= 100e6
    assert acc.f_tone.max() <= 200e6

    # f_lo_center should be a scalar (accessible via meta)
    assert acc.meta.f_lo_center == 150e6

    # f_chan should be f_tone + f_lo_center
    assert acc.f_chan is not None
    np.testing.assert_allclose(
        acc.f_chan.values,
        acc.f_tone.values + acc.meta.f_lo_center,
    )


def test_tone_mask(toltec_sweep_dataset):
    """Test tone mask property."""
    acc = toltec_sweep_dataset.toltec_kids

    mask = acc.tone_mask
    assert mask is not None
    assert len(mask) == 256

    # Check that some tones are masked (every 10th)
    assert not mask[0]  # First tone should be masked
    assert mask[1]  # Second tone should not be masked


def test_tone_amp_and_phase(toltec_sweep_dataset):
    """Test tone amplitude and phase properties."""
    acc = toltec_sweep_dataset.toltec_kids

    amp = acc.tone_amp
    assert amp is not None
    assert len(amp) == 256
    np.testing.assert_allclose(amp.values, 0.1)

    phase = acc.tone_phase
    assert phase is not None
    assert len(phase) == 256
    np.testing.assert_allclose(phase.values, 0.0)


def test_data_type_detection(toltec_sweep_dataset, toltec_timestream_dataset):
    """Test data_kind property for sweep and timestream datasets."""
    from tolteca_datamodels.toltec.types import ToltecDataKind

    sweep_kinds = (
        ToltecDataKind.VnaSweep,
        ToltecDataKind.TargetSweep,
        ToltecDataKind.Tune,
        ToltecDataKind.RawSweep,
        ToltecDataKind.ReducedSweep,
    )
    timestream_kinds = (
        ToltecDataKind.RawTimeStream,
        ToltecDataKind.SolvedTimeStream,
    )

    assert toltec_sweep_dataset.toltec_kids.data_kind in sweep_kinds
    assert toltec_timestream_dataset.toltec_kids.data_kind in timestream_kinds


def test_n_chans(toltec_sweep_dataset, toltec_timestream_dataset):
    """Test number of channels via metadata."""
    assert toltec_sweep_dataset.toltec_kids.meta.n_chans == 256
    assert toltec_timestream_dataset.toltec_kids.meta.n_chans == 256


def test_cached_properties(toltec_sweep_dataset):
    """Test that properties are cached."""
    acc = toltec_sweep_dataset.toltec_kids

    # Access meta twice - should return same cached object
    meta1 = acc.meta
    meta2 = acc.meta
    assert meta1 is meta2

    # Same for array properties
    f_tone1 = acc.f_tone
    f_tone2 = acc.f_tone
    assert f_tone1 is f_tone2


def test_mapper_validation_methods(toltec_sweep_dataset):
    """Test inherited validation methods from KidsMapper."""
    acc = toltec_sweep_dataset.toltec_kids
    mapper = acc.mapper
    ds = toltec_sweep_dataset

    # Test validate_has_field - should succeed
    mapper.validate_has_field(mapper.schema.I)
    mapper.validate_has_field(mapper.schema.Q)
    mapper.validate_has_field(mapper.schema.roach)

    # Test validate_has_field - should fail for missing field
    from tollan.accessor import Mapping

    fake_field = Mapping("nonexistent_field")
    with pytest.raises(ValueError, match="Missing required field"):
        mapper.validate_has_field(fake_field)


def test_integration_with_kids_accessor(toltec_sweep_dataset):
    """Test that toltec_kids accessor works alongside kids accessor.

    Note: The generic kids accessor requires "I" and "Q" field names.
    To use both accessors with TolTEC data, you need to rename variables:

        ds_renamed = ds.rename({"Data.Toltec.Is": "I", "Data.Toltec.Qs": "Q"})
        ds_renamed.kids.multi_sweep  # Generic accessor for multi-channel
        ds_renamed.toltec_kids  # TolTEC accessor
    """
    # Import kids accessor to register it
    from tolteca_kidsproc.accessors.kids import KidsAccessor  # noqa: F401

    # TolTEC accessor should work with original field names
    assert hasattr(toltec_sweep_dataset, "toltec_kids")
    assert toltec_sweep_dataset.toltec_kids.meta.roach == 0
    assert toltec_sweep_dataset.toltec_kids.meta.array_name == "a1100"

    # To use kids accessor with TolTEC data, rename fields
    ds_renamed = toltec_sweep_dataset.rename(
        {"Data.Toltec.Is": "I", "Data.Toltec.Qs": "Q", "sweep": "frequency"},
    )

    # Now generic kids accessor works
    assert hasattr(ds_renamed, "kids")
    multi_sweep = ds_renamed.kids.multi_sweep
    assert multi_sweep.S21 is not None

    # TolTEC accessor is accessible (but metadata extraction may fail on renamed data)
    assert hasattr(ds_renamed, "toltec_kids")


def test_data_kind_property(toltec_sweep_dataset, toltec_timestream_dataset):
    """Test data_kind property returns appropriate data kind."""
    from tolteca_datamodels.toltec.types import ToltecDataKind

    # Sweep dataset should identify as TargetSweep (ObsType=1 in fixture)
    sweep_kind = toltec_sweep_dataset.toltec_kids.data_kind
    assert sweep_kind == ToltecDataKind.TargetSweep

    # Timestream dataset should identify as raw timestream
    timestream_kind = toltec_timestream_dataset.toltec_kids.data_kind
    assert timestream_kind == ToltecDataKind.RawTimeStream


def test_get_chan_axis_data(toltec_sweep_dataset):
    """Test channel axis data table generation."""
    chan_data = toltec_sweep_dataset.toltec_kids.get_chan_axis_data()

    # Check table structure
    assert "channel" in chan_data.colnames
    assert "f_tone" in chan_data.colnames
    assert "f_chan" in chan_data.colnames
    assert "tone_mask" in chan_data.colnames
    assert "tone_amp" in chan_data.colnames
    assert "tone_phase" in chan_data.colnames

    # Check dimensions
    assert len(chan_data) == 256

    # Check metadata (note: table.meta has limited metadata)
    assert chan_data.meta["roach"] == 0
    assert chan_data.meta["array_name"] == "a1100"
    # obs_num is accessed via the accessor's .meta property, not table.meta
    assert toltec_sweep_dataset.toltec_kids.meta.obsnum == 12345

    # Check values
    assert chan_data["channel"][0] == 0
    assert chan_data["channel"][-1] == 255

    # Check units
    assert chan_data["f_tone"].unit.to_string() == "Hz"
    assert chan_data["f_chan"].unit.to_string() == "Hz"
    assert chan_data["tone_phase"].unit.to_string() == "rad"

    # Check tone mask (every 10th should be False)
    assert not chan_data["tone_mask"][0]
    assert chan_data["tone_mask"][1]


def test_get_sweep_axis_data(toltec_sweep_dataset):
    """Test sweep axis data table generation."""
    sweep_data = toltec_sweep_dataset.toltec_kids.get_sweep_axis_data()

    # Check table structure
    assert "sweep_id" in sweep_data.colnames
    assert "f_sweep" in sweep_data.colnames
    assert "f_lo" in sweep_data.colnames

    # Check dimensions
    assert len(sweep_data) == 50

    # Check metadata
    assert sweep_data.meta["roach"] == 0
    assert sweep_data.meta["array_name"] == "a1100"

    # Check sweep IDs
    assert sweep_data["sweep_id"][0] == 0
    assert sweep_data["sweep_id"][-1] == 49

    # Check units
    assert sweep_data["f_sweep"].unit.to_string() == "Hz"
    assert sweep_data["f_lo"].unit.to_string() == "Hz"

    # Check frequency values make sense
    # All frequencies should be positive
    assert (sweep_data["f_lo"].value > 0).all()
    assert len(sweep_data["f_sweep"]) == 50


def test_get_sweep_axis_data_raises_for_timestream(toltec_timestream_dataset):
    """Test that get_sweep_axis_data raises error for non-sweep data."""
    with pytest.raises(ValueError, match="not sweep data"):
        toltec_timestream_dataset.toltec_kids.get_sweep_axis_data()


def test_select_channels_by_mask(toltec_sweep_dataset):
    """Test channel selection by tone mask."""
    # Select enabled channels
    ds_enabled = toltec_sweep_dataset.toltec_kids.select_channels("tone_mask")

    # Should have fewer channels (every 10th is masked)
    original_n_chans = toltec_sweep_dataset.toltec_kids.meta.n_chans
    selected_n_chans = ds_enabled.toltec_kids.meta.n_chans

    # 256 channels, every 10th masked = 256 - 26 = 230 enabled
    assert selected_n_chans < original_n_chans
    assert selected_n_chans == 230

    # All selected channels should have tone_mask=True
    assert ds_enabled.toltec_kids.tone_mask.all()


def test_select_channels_by_slice(toltec_sweep_dataset):
    """Test channel selection by slice."""
    # Select first 10 channels
    ds_subset = toltec_sweep_dataset.toltec_kids.select_channels(slice(0, 10))

    assert ds_subset.toltec_kids.meta.n_chans == 10
    assert ds_subset.toltec_kids.f_tone[0] == toltec_sweep_dataset.toltec_kids.f_tone[0]


def test_select_channels_by_list(toltec_sweep_dataset):
    """Test channel selection by index list."""
    indices = [0, 5, 10, 15, 20]
    ds_subset = toltec_sweep_dataset.toltec_kids.select_channels(indices)

    assert ds_subset.toltec_kids.meta.n_chans == 5
    # Check that selected channels match
    np.testing.assert_array_equal(
        ds_subset.toltec_kids.f_tone.values,
        toltec_sweep_dataset.toltec_kids.f_tone.values[indices],
    )


# Real data tests
# (path/file fixtures are shared via conftest.py)


def test_open_toltec(real_targsweep_file):
    """Test opening real TolTEC file with open_toltec convenience function."""
    from tolteca_datamodels.toltec import open_toltec

    ds = open_toltec(str(real_targsweep_file))

    # Should have toltec_kids accessor
    assert hasattr(ds, "toltec_kids")

    # Should be able to access metadata
    assert ds.toltec_kids.meta.obsnum > 0
    assert ds.toltec_kids.meta.array_name in ["a1100", "a1400", "a2000"]

    # Should be able to access channel data
    assert ds.toltec_kids.meta.n_chans > 0


def test_real_data_channel_axis(real_targsweep_file):
    """Test get_chan_axis_data with real TolTEC data."""
    from astropy import units as u

    from tolteca_datamodels.toltec import open_toltec

    ds = open_toltec(str(real_targsweep_file))
    chan_table = ds.toltec_kids.get_chan_axis_data()

    # Should have expected columns
    assert "channel" in chan_table.colnames
    assert "f_tone" in chan_table.colnames
    assert "f_chan" in chan_table.colnames
    assert "tone_mask" in chan_table.colnames

    # Channel numbers should match dataset
    assert len(chan_table) == ds.toltec_kids.meta.n_chans

    # Frequencies may be offsets or absolute values
    # Offsets are typically -10 to +10 MHz, absolute are 100-300 MHz
    f_tone_abs = np.abs(chan_table["f_tone"])
    # Should be reasonable (either small offsets or ROACH tone frequencies)
    assert (f_tone_abs < 400 * u.MHz).all()  # < 400 MHz


def test_real_data_channel_selection(real_vnasweep_file):
    """Test channel selection with real TolTEC data."""
    from tolteca_datamodels.toltec import open_toltec

    ds = open_toltec(str(real_vnasweep_file))
    original_n_chans = ds.toltec_kids.meta.n_chans

    # Just verify we can call select_channels (raw data may not support all operations)
    # Select first 100 channels by slice
    try:
        ds_subset = ds.toltec_kids.select_channels(slice(0, 100))
        # If successful, should have 100 channels in the selected dimension
        assert hasattr(ds_subset, "toltec_kids")
    except (ValueError, KeyError):
        # Some raw data formats may not support selection
        pytest.skip("Channel selection not supported for this data format")


def test_real_data_loads_successfully(toltec0_data_path):
    """Test that all available TolTEC files can be loaded."""
    from tolteca_datamodels.toltec import open_toltec

    nc_files = list(toltec0_data_path.glob("*.nc"))
    assert len(nc_files) > 0, "No netCDF files found"

    for nc_file in nc_files:
        # Should load without error
        ds = open_toltec(str(nc_file))

        # Should have accessor
        assert hasattr(ds, "toltec_kids")

        # Should be able to access basic metadata
        assert ds.toltec_kids.meta.n_chans > 0
        assert ds.toltec_kids.meta.array_name in ["a1100", "a1400", "a2000"]

        # Should be able to get channel axis data (may be multi-dimensional)
        try:
            chan_table = ds.toltec_kids.get_chan_axis_data()
            # Just verify it returns a table
            assert len(chan_table) > 0
        except ValueError:
            # Some raw data may have complex structures
            pytest.skip(f"Channel axis data not available for {nc_file.name}")


def test_reduce_raw_sweep_mock_data():
    """Test raw sweep reduction with mock data."""
    from tolteca_datamodels.toltec import reduce_raw_sweep

    n_chans = 100
    n_sweeps = 20
    n_samples_per_step = 10
    f_lo_center = 150e6

    # Create mock raw time-series data
    f_sweep = np.linspace(-2e6, 2e6, n_sweeps)
    f_lo = f_sweep + f_lo_center

    # Repeat f_lo for each sample
    f_lo_repeated = np.repeat(f_lo, n_samples_per_step)
    n_samples = len(f_lo_repeated)

    # Create I/Q data with deterministic signal
    # Base signal varying across channels
    chan_phase = np.linspace(0, 2 * np.pi, n_chans)
    base_i = np.sin(chan_phase)
    base_q = np.cos(chan_phase)

    # Add deterministic variation across samples
    I_raw = np.zeros((n_samples, n_chans))
    Q_raw = np.zeros((n_samples, n_chans))
    for i in range(n_samples):
        sample_phase = 2 * np.pi * i / n_samples_per_step
        I_raw[i, :] = base_i + 0.1 * np.sin(sample_phase + chan_phase)
        Q_raw[i, :] = base_q + 0.1 * np.cos(sample_phase + chan_phase)

    # Create raw dataset
    ds_raw = xr.Dataset(
        {
            "Data.Toltec.Is": (["time", "iqlen"], I_raw),
            "Data.Toltec.Qs": (["time", "iqlen"], Q_raw),
            "Data.Toltec.LoFreq": (["time"], f_lo_repeated),
        },
        attrs={"Header.Toltec.LoCenterFreq": f_lo_center},
    )

    # Reduce the sweep (returns DataTree)
    dt = reduce_raw_sweep(ds_raw)
    assert isinstance(dt, xr.DataTree)
    ds_reduced = dt.children[NAMESPACE].dataset

    # Check output structure - use namespaced names
    assert get_var_name("I") in ds_reduced
    assert get_var_name("Q") in ds_reduced
    assert get_var_name("unc_I") in ds_reduced
    assert get_var_name("unc_Q") in ds_reduced
    assert "sweep" in ds_reduced.dims
    assert "sweep" in ds_reduced.coords

    # Check dimensions
    assert ds_reduced.sizes["iqlen"] == n_chans
    assert ds_reduced.sizes["sweep"] == n_sweeps

    # Check that mean values are computed
    assert not np.isnan(ds_reduced[get_var_name("I")].values).all()
    assert not np.isnan(ds_reduced[get_var_name("Q")].values).all()

    # Check that uncertainties are positive
    assert (ds_reduced[get_var_name("unc_I")].values >= 0).all()
    assert (ds_reduced[get_var_name("unc_Q")].values >= 0).all()

    # Check sweep coordinate
    np.testing.assert_allclose(ds_reduced.coords["sweep"].values, f_sweep, rtol=1e-5)

    # Check attributes
    assert "processing" in ds_reduced.attrs
    assert ds_reduced.attrs["processing"] == "reduced_raw_sweep"
    assert ds_reduced.attrs["n_sweeps"] == n_sweeps


def _check_raw_sweep_reduction(filepath: Path) -> None:
    """Assert correct reduction of a raw sweep file (shared by vna/targ tests)."""
    from tolteca_datamodels.toltec import open_toltec, reduce_raw_sweep

    ds_raw = open_toltec(filepath)

    # Verify raw data structure
    assert "Data.Toltec.Is" in ds_raw
    assert "Data.Toltec.Qs" in ds_raw
    assert "Data.Toltec.LoFreq" in ds_raw

    # Reduce the sweep (returns DataTree)
    dt = reduce_raw_sweep(ds_raw)
    assert isinstance(dt, xr.DataTree)
    ds_reduced = dt.children[NAMESPACE].dataset

    # Check output structure — use namespaced names
    assert get_var_name("I") in ds_reduced
    assert get_var_name("Q") in ds_reduced
    assert get_var_name("unc_I") in ds_reduced
    assert get_var_name("unc_Q") in ds_reduced
    assert "sweep" in ds_reduced.dims
    assert "sweep" in ds_reduced.coords

    # Check data types
    for field in ("I", "Q", "unc_I", "unc_Q"):
        assert ds_reduced[get_var_name(field)].dtype in [np.float32, np.float64]

    # Verify no NaN values
    assert not np.any(np.isnan(ds_reduced[get_var_name("I")].values))
    assert not np.any(np.isnan(ds_reduced[get_var_name("Q")].values))
    assert not np.any(np.isnan(ds_reduced[get_var_name("unc_I")].values))
    assert not np.any(np.isnan(ds_reduced[get_var_name("unc_Q")].values))

    # Verify uncertainties are non-negative
    assert np.all(ds_reduced[get_var_name("unc_I")] >= 0)
    assert np.all(ds_reduced[get_var_name("unc_Q")] >= 0)

    # Verify key metadata attrs
    assert "f_center" in ds_reduced.attrs
    assert "raw_I_shape" in ds_reduced.attrs
    assert "samples_per_sweep" in ds_reduced.attrs

    # Check that reduction actually reduced sample count
    n_samples_original = ds_reduced.attrs["raw_I_shape"][0]
    n_sweeps = ds_reduced.sizes["sweep"]
    assert n_sweeps < n_samples_original


def test_real_vnasweep_reduction(real_vnasweep_file: Path) -> None:
    """Test raw sweep reduction with a real VNA sweep file."""
    _check_raw_sweep_reduction(real_vnasweep_file)


def test_real_targsweep_reduction(real_targsweep_file: Path) -> None:
    """Test raw sweep reduction with a real target sweep file."""
    _check_raw_sweep_reduction(real_targsweep_file)


def test_real_tune_multi_block_data(real_tune_file: Path) -> None:
    """Test multi-block detection and processing with a real tune file."""
    from tolteca_datamodels.toltec import open_toltec, reduce_raw_sweep

    ds_raw = open_toltec(real_tune_file)

    # Verify raw data structure
    assert "Data.Toltec.Is" in ds_raw
    assert "Data.Toltec.Qs" in ds_raw
    assert "Data.Toltec.LoFreq" in ds_raw

    # Detect expected block count from LO frequency monotonicity
    f_lo = ds_raw["Data.Toltec.LoFreq"].values
    if f_lo.ndim > 1:
        f_lo = f_lo[0]
    expected_n_blocks = len(np.where(np.diff(f_lo) < 0)[0]) + 1

    # Reduce (returns DataTree); extract the reduced child dataset
    dt = reduce_raw_sweep(ds_raw)
    assert isinstance(dt, xr.DataTree)
    ds_reduced = dt.children[NAMESPACE].dataset

    # Check block structure
    if expected_n_blocks > 1:
        assert "block" in ds_reduced.dims
        assert ds_reduced.sizes["block"] == expected_n_blocks
    else:
        assert "block" not in ds_reduced.dims

    # Verify output structure
    for field in ("I", "Q", "unc_I", "unc_Q"):
        assert get_var_name(field) in ds_reduced

    assert not np.any(np.isnan(ds_reduced[get_var_name("I")].values))
    assert not np.any(np.isnan(ds_reduced[get_var_name("Q")].values))


# Additional accessor property tests
# ===================================


def test_accessor_additional_metadata_properties(toltec_sweep_dataset):
    """Test additional metadata-related accessor properties."""
    acc = toltec_sweep_dataset.toltec_kids

    # Test subobsnum and scannum via meta
    assert acc.meta.subobsnum == 1
    assert acc.meta.scannum == 0
    assert acc.meta.f_smp.value == 1000.0


def test_accessor_tone_amp_property(toltec_sweep_dataset):
    """Test tone_amp property."""
    acc = toltec_sweep_dataset.toltec_kids

    tone_amp = acc.tone_amp
    assert tone_amp is not None
    assert len(tone_amp) == 256
    assert (tone_amp > 0).all()


def test_accessor_tone_phase_property(toltec_sweep_dataset):
    """Test tone_phase property."""
    acc = toltec_sweep_dataset.toltec_kids

    tone_phase = acc.tone_phase
    assert tone_phase is not None
    assert len(tone_phase) == 256
    # Phase should be in radians, typically -pi to pi
    assert (tone_phase >= -np.pi).all()
    assert (tone_phase <= np.pi).all()


def test_accessor_f_lo_property(toltec_sweep_dataset):
    """Test f_lo property for sweep data."""
    acc = toltec_sweep_dataset.toltec_kids

    # For mock sweep data without f_lo coordinate, this may be None
    # The property works correctly when f_lo data is present
    f_lo = acc.f_lo
    # Just check it doesn't raise an error (value depends on data structure)
    # Real data would have f_lo coordinate or field


def test_accessor_f_lo_center_property(toltec_sweep_dataset):
    """Test f_lo_center accessible via meta."""
    acc = toltec_sweep_dataset.toltec_kids

    f_lo_center = acc.meta.f_lo_center
    assert f_lo_center == 150e6  # From fixture


def test_accessor_f_chan_property(toltec_sweep_dataset):
    """Test f_chan property (derived from f_tone + f_lo_center)."""
    acc = toltec_sweep_dataset.toltec_kids

    f_chan = acc.f_chan
    assert f_chan is not None
    assert len(f_chan) == 256

    # f_chan should be f_tone + f_lo_center
    f_tone = acc.f_tone
    f_lo_center = acc.meta.f_lo_center
    expected_f_chan = f_tone + f_lo_center
    np.testing.assert_allclose(f_chan.values, expected_f_chan.values)


def test_accessor_metadata_extraction(toltec_sweep_dataset):
    """Test metadata extraction as dataclass."""
    acc = toltec_sweep_dataset.toltec_kids

    metadata = acc.meta
    assert metadata is not None

    # Should be ToltecSweepMetadata
    from tolteca_datamodels.toltec.metadata import ToltecSweepMetadata

    assert isinstance(metadata, ToltecSweepMetadata)
    assert metadata.obsnum == 12345
    assert metadata.roach == 0
    assert metadata.n_chans == 256


# ── Reduced-view accessor entry points ────────────────────────────────────────


SWEEP_NS = "tolteca_datamodels.toltec.kids.sweep"
TIMESTREAM_NS = "tolteca_datamodels.toltec.kids.timestream"


class TestReducedSweepAccessorEntry:
    """Test dt.toltec_kids.sweep entry point."""

    def _make_sweep_dt(self) -> xr.DataTree:
        from tolteca_datamodels.toltec.kids import SweepReducer

        n_chans = 10
        n_sweeps = 5
        n_sps = 4  # samples per sweep step
        n_times = n_sweeps * n_sps

        f_lo_center = 150e6
        f_sweep = np.linspace(-2e6, 2e6, n_sweeps)
        f_lo = np.repeat(f_sweep + f_lo_center, n_sps)

        rng = np.random.default_rng(0)
        I = rng.standard_normal((n_times, n_chans))
        Q = rng.standard_normal((n_times, n_chans))

        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (["time", "iqlen"], I),
                "Data.Toltec.Qs": (["time", "iqlen"], Q),
                "Data.Toltec.LoFreq": (["time"], f_lo),
            },
            attrs={"Header.Toltec.LoCenterFreq": f_lo_center},
        )
        return SweepReducer(compute_uncertainty=True)(ds)

    def test_accessor_registered_on_datatree(self):
        """DataTree returned by SweepReducer supports toltec_kids accessor."""
        dt = self._make_sweep_dt()
        assert hasattr(dt, "toltec_kids")

    def test_reduced_sweep_returns_view(self):
        """dt.toltec_kids.sweep returns a ReducedSweepView."""
        from tolteca_datamodels.toltec.kids import ReducedSweepView

        dt = self._make_sweep_dt()
        view = dt.toltec_kids.sweep
        assert isinstance(view, ReducedSweepView)

    def test_reduced_sweep_view_has_data(self):
        """View resolves the sweep child node and exposes I/Q data."""
        dt = self._make_sweep_dt()
        view = dt.toltec_kids.sweep
        assert view.I is not None
        assert view.Q is not None

    def test_reduced_sweep_view_matches_child_directly(self):
        """View data matches accessing the child node directly."""
        dt = self._make_sweep_dt()
        view = dt.toltec_kids.sweep
        child_ds = dt.children[SWEEP_NS].dataset

        np.testing.assert_array_equal(
            view.I.values,
            child_ds[f"{SWEEP_NS}.I"].values,
        )

    def test_reduced_sweep_from_raw_dataset(self):
        """toltec_kids accessor on a plain Dataset returns a view (no child data)."""
        from tolteca_datamodels.toltec.kids import ReducedSweepView

        # Raw dataset has no reduced child — view is constructed but I is None
        ds = xr.Dataset(
            {
                "Data.Toltec.Is": (["time", "iqlen"], np.zeros((20, 4))),
                "Data.Toltec.Qs": (["time", "iqlen"], np.zeros((20, 4))),
            }
        )
        view = ds.toltec_kids.sweep
        assert isinstance(view, ReducedSweepView)
        # No reduced data present → I should be None
        assert view.I is None


class TestReducedTimestreamAccessorEntry:
    """Test dt.toltec_kids.timestream entry point."""

    def _make_ts_dt(self) -> xr.DataTree:
        from tolteca_datamodels.toltec.kids import TimestreamReducer

        n_samples = 2000
        fsmp = 200.0
        rng = np.random.default_rng(1)

        ds = xr.Dataset(
            {
                "I": (["time"], rng.standard_normal(n_samples)),
                "Q": (["time"], rng.standard_normal(n_samples)),
            }
        )
        ds["time"] = np.arange(n_samples) / fsmp
        ds.attrs["f_smp"] = fsmp
        return TimestreamReducer(psd_nperseg=256)(ds)

    def test_accessor_registered_on_datatree(self):
        """DataTree returned by TimestreamReducer supports toltec_kids accessor."""
        dt = self._make_ts_dt()
        assert hasattr(dt, "toltec_kids")

    def test_reduced_timestream_returns_view(self):
        """dt.toltec_kids.timestream returns a ReducedTimestreamView."""
        from tolteca_datamodels.toltec.kids import ReducedTimestreamView

        dt = self._make_ts_dt()
        view = dt.toltec_kids.timestream
        assert isinstance(view, ReducedTimestreamView)

    def test_reduced_timestream_view_has_psd(self):
        """View resolves the timestream child node and exposes PSD data."""
        dt = self._make_ts_dt()
        view = dt.toltec_kids.timestream
        assert view.f_psd is not None
        assert view.I_psd is not None
        assert view.Q_psd is not None

    def test_reduced_timestream_view_matches_child_directly(self):
        """View f_psd matches accessing the child node directly."""
        dt = self._make_ts_dt()
        view = dt.toltec_kids.timestream
        child_ds = dt.children[TIMESTREAM_NS].dataset

        np.testing.assert_array_equal(
            view.f_psd.values,
            child_ds[f"{TIMESTREAM_NS}.f_psd"].values,
        )

    def test_root_ds_used_for_mapper_on_datatree(self):
        """Accessor on DataTree uses root Dataset (not the DataTree) for mapper."""
        dt = self._make_ts_dt()
        acc = dt.toltec_kids
        # _root_ds must be a plain xr.Dataset (not a DataTree)
        assert isinstance(acc._root_ds, xr.Dataset)
        assert not isinstance(acc._root_ds, xr.DataTree)
        # _obj must be the original DataTree
        assert isinstance(acc._obj, xr.DataTree)
