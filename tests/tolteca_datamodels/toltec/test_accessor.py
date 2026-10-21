"""Tests for TolTEC KIDs data accessor."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from tolteca_datamodels.toltec.kids.core import ToltecKidsIOMapper

_OBS_HEADERS = {
    "Header.Toltec.Master": 0,
    "Header.Toltec.ObsNum": 12345,
    "Header.Toltec.SubObsNum": 0,
    "Header.Toltec.ScanNum": 0,
    "Header.Toltec.ObsStartTime": 1761902400,
    "Header.Toltec.ObsEndTime": 1761902460,
}


@pytest.fixture
def toltec_sweep_dataset(kids_readout_settings, kids_design_data):
    """Toy numerical sweep; not evidence of an acquisition file format."""
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
            **kids_design_data.data_vars,
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
            **kids_readout_settings,
            **_OBS_HEADERS,
            "Header.Toltec.RoachIndex": 0,
            "Header.Toltec.ObsNum": 12345,
            "Header.Toltec.SubObsNum": 1,
            "Header.Toltec.ScanNum": 0,
            "Header.Toltec.SampleFreq": 1000.0,
            "Header.Toltec.LoCenterFreq": 150e6,  # 150 MHz center LO
            "Header.Toltec.ObsType": 3,  # TargetSweep
            "Header.Kids.kind": "processed_sweep",  # One value per sweep step.
        },
    )
    # Add units to coordinates
    ds["sweep"].attrs["units"] = "Hz"
    ds["Header.Toltec.ToneFreq"].attrs["units"] = "Hz"

    return ds


@pytest.fixture
def toltec_timestream_dataset(kids_readout_settings, kids_design_data):
    """Toy numerical timestream; not evidence of an acquisition file format."""
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
            **kids_design_data.data_vars,
            "Data.Toltec.Is": (["chan", "time"], i_data),
            "Data.Toltec.Qs": (["chan", "time"], q_data),
            "Header.Toltec.ToneFreq": (["chan"], f_tones),
            "Header.Toltec.ToneMask": (["chan"], tone_mask),
            "Header.Toltec.ToneAmp": (["chan"], np.full(n_chans, 0.1)),
        },
        coords={
            "chan": np.arange(n_chans),
            "time": times,
            "numSweeps": [0],
        },
        attrs={
            **kids_readout_settings,
            **_OBS_HEADERS,
            "Header.Toltec.RoachIndex": 7,  # a1400 array
            "Header.Toltec.ObsNum": 54321,
            "Header.Toltec.ObsType": 0,  # Nominal, as in the reference timestream.
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

    assert acc.block_index.meta.roach == 0
    assert acc.block_index.meta.obsnum == 12345
    assert acc.block_index.meta.subobsnum == 1
    assert acc.block_index.meta.scannum == 0
    assert acc.block_index.meta.f_smp.value == 1000.0


def test_array_name_mapping(toltec_sweep_dataset, toltec_timestream_dataset):
    """Test array name derived from ROACH index."""
    # ROACH 0 -> a1100
    assert toltec_sweep_dataset.toltec_kids.block_index.meta.array_name == "a1100"

    # ROACH 7 -> a1400
    assert toltec_timestream_dataset.toltec_kids.meta.array_name == "a1400"

    # Test a2000 array
    ds = toltec_sweep_dataset.copy()
    ds.attrs["Header.Toltec.RoachIndex"] = 11
    assert ds.toltec_kids.block_index.meta.array_name == "a2000"


def test_tone_properties(toltec_sweep_dataset):
    """Test tone/channel frequency properties."""
    acc = toltec_sweep_dataset.toltec_kids

    # f_tone should be an array
    assert acc.f_tone is not None
    assert len(acc.f_tone) == 256
    assert acc.f_tone.min() >= 100e6
    assert acc.f_tone.max() <= 200e6

    # f_lo_center should be a scalar (accessible via meta)
    assert acc.block_index.meta.f_lo_center == 150e6

    # f_chan should be f_tone + f_lo_center
    assert acc.f_chan is not None
    np.testing.assert_allclose(
        acc.f_chan.values,
        acc.f_tone.values + acc.block_index.meta.f_lo_center,
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


def test_n_chans(toltec_sweep_dataset, toltec_timestream_dataset):
    """Test number of channels via metadata."""
    assert toltec_sweep_dataset.toltec_kids.block_index.meta.n_chans == 256
    assert toltec_timestream_dataset.toltec_kids.meta.n_chans == 256


def test_cached_properties(toltec_sweep_dataset):
    """Test that properties are cached."""
    acc = toltec_sweep_dataset.toltec_kids

    # Access meta twice - should return same cached object
    meta1 = acc.block_index.meta
    meta2 = acc.block_index.meta
    assert meta1 is meta2

    # Same for array properties
    f_tone1 = acc.f_tone
    f_tone2 = acc.f_tone
    assert f_tone1 is f_tone2


def test_mapper_validation_methods(toltec_sweep_dataset):
    """Test inherited validation methods from KidsMapper."""
    acc = toltec_sweep_dataset.toltec_kids
    mapper = acc.mapper

    # Test validate_has_field - should succeed
    mapper.validate_has_field(mapper.schema.I)
    mapper.validate_has_field(mapper.schema.Q)
    mapper.validate_has_field(mapper.schema.readout_code)

    # Test validate_has_field - should fail for missing field
    from tollan.accessor import NameMapping

    fake_field = NameMapping("nonexistent_field")
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
    assert toltec_sweep_dataset.toltec_kids.block_index.meta.roach == 0
    assert toltec_sweep_dataset.toltec_kids.block_index.meta.array_name == "a1100"

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

    # The toy sweep stores one value per step, not raw acquisition samples.
    sweep_kind = toltec_sweep_dataset.toltec_kids.data_kind
    assert sweep_kind == ToltecDataKind.reduced_sweep

    # Timestream dataset should identify as raw timestream
    timestream_kind = toltec_timestream_dataset.toltec_kids.data_kind
    assert timestream_kind == ToltecDataKind.raw_timestream


def test_get_chan_axis_table(toltec_sweep_dataset):
    """Test channel axis data table generation."""
    chan_data = toltec_sweep_dataset.toltec_kids.get_chan_axis_table()

    # Check table structure
    assert "chan" in chan_data.colnames
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
    assert toltec_sweep_dataset.toltec_kids.block_index.meta.obsnum == 12345

    # Check values
    assert chan_data["chan"][0] == 0
    assert chan_data["chan"][-1] == 255

    # Check units
    assert chan_data["f_tone"].unit.to_string() == "Hz"
    assert chan_data["f_chan"].unit.to_string() == "Hz"
    assert chan_data["tone_phase"].unit.to_string() == "rad"

    # Check tone mask (every 10th should be False)
    assert not chan_data["tone_mask"][0]
    assert chan_data["tone_mask"][1]


def test_get_probe_axis_table(toltec_sweep_dataset):
    """Test sweep axis data table generation."""
    sweep_data = toltec_sweep_dataset.toltec_kids.get_probe_axis_table()

    # Check table structure
    assert "step" in sweep_data.colnames
    assert "f_sweep" in sweep_data.colnames
    assert "f_lo" in sweep_data.colnames

    # Check dimensions
    assert len(sweep_data) == 50

    # Check metadata
    assert sweep_data.meta["roach"] == 0
    assert sweep_data.meta["array_name"] == "a1100"

    # Check sweep IDs
    assert sweep_data["step"][0] == 0
    assert sweep_data["step"][-1] == 49

    # Check units
    assert sweep_data["f_sweep"].unit.to_string() == "Hz"
    assert sweep_data["f_lo"].unit.to_string() == "Hz"

    # Check frequency values make sense
    # All frequencies should be positive
    assert (sweep_data["f_lo"].value > 0).all()
    assert len(sweep_data["f_sweep"]) == 50


def test_get_probe_axis_table_for_timestream(toltec_timestream_dataset):
    """A constant-LO timestream has one probe dwell spanning all samples."""
    axis = toltec_timestream_dataset.toltec_kids.get_probe_axis_table()
    assert len(axis) == 1
    assert axis["step"][0] == 0
    assert axis["n_samples"][0] == toltec_timestream_dataset.sizes["time"]
    assert axis["sample_start"][0] == 0
    assert axis["sample_end"][0] == toltec_timestream_dataset.sizes["time"]


def test_select_channels_by_mask(toltec_sweep_dataset):
    """Test channel selection by tone mask."""
    # Select enabled channels
    ds_enabled = toltec_sweep_dataset.toltec_kids.select_channels("tone_mask")

    # Should have fewer channels (every 10th is masked)
    original_n_chans = toltec_sweep_dataset.toltec_kids.block_index.meta.n_chans
    selected_n_chans = ds_enabled.toltec_kids.block_index.meta.n_chans

    # 256 channels, every 10th masked = 256 - 26 = 230 enabled
    assert selected_n_chans < original_n_chans
    assert selected_n_chans == 230

    # All selected channels should have tone_mask=True
    assert ds_enabled.toltec_kids.tone_mask.all()


def test_select_channels_by_slice(toltec_sweep_dataset):
    """Test channel selection by slice."""
    # Select first 10 channels
    ds_subset = toltec_sweep_dataset.toltec_kids.select_channels(slice(0, 10))

    assert ds_subset.toltec_kids.block_index.meta.n_chans == 10
    assert ds_subset.toltec_kids.f_tone[0] == toltec_sweep_dataset.toltec_kids.f_tone[0]


def test_select_channels_by_list(toltec_sweep_dataset):
    """Test channel selection by index list."""
    indices = [0, 5, 10, 15, 20]
    ds_subset = toltec_sweep_dataset.toltec_kids.select_channels(indices)

    assert ds_subset.toltec_kids.block_index.meta.n_chans == 5
    # Check that selected channels match
    np.testing.assert_array_equal(
        ds_subset.toltec_kids.f_tone.values,
        toltec_sweep_dataset.toltec_kids.f_tone.values[indices],
    )


# Real data tests
# (path/file fixtures are shared via conftest.py)


def test_open_toltec(real_targsweep_file):
    """Test opening real TolTEC file with open_toltec_kids convenience function."""
    from tolteca_datamodels.toltec import open_toltec_kids

    with open_toltec_kids(real_targsweep_file) as ds:
        # Should have toltec_kids accessor
        assert hasattr(ds, "toltec_kids")

        # Should be able to access metadata
        assert ds.toltec_kids.meta.obsnum > 0
        assert ds.toltec_kids.meta.array_name in [
            "a1100",
            "a1400",
            "a2000",
        ]

        # Should be able to access channel data
        assert ds.toltec_kids.meta.n_chans > 0


def test_real_data_channel_axis(real_targsweep_file):
    """Test get_chan_axis_table with real TolTEC data."""
    from astropy import units as u

    from tolteca_datamodels.toltec import open_toltec_kids

    with open_toltec_kids(real_targsweep_file) as ds:
        chan_table = ds.toltec_kids.get_chan_axis_table()

        # Should have expected columns
        assert "chan" in chan_table.colnames
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
    from tolteca_datamodels.toltec import open_toltec_kids

    with open_toltec_kids(real_vnasweep_file) as ds:
        original_n_chans = ds.toltec_kids.meta.n_chans
        ds_subset = ds.toltec_kids.select_channels(slice(0, 100))
        mapper = ds_subset.toltec_kids.mapper
        chan_dim = mapper.get_name(mapper.schema.n_chans)
        assert chan_dim is not None
        selected_count = min(100, original_n_chans)
        assert ds_subset.sizes[chan_dim] == selected_count
        np.testing.assert_array_equal(ds_subset[chan_dim], np.arange(selected_count))
        assert ds.toltec_kids.meta.n_chans == original_n_chans


def test_real_mapper_get_var_preserves_lazy_backend(real_vnasweep_file):
    """Mapper resolution and labeled reads leave the large raw I array lazy."""
    with xr.open_dataset(real_vnasweep_file) as ds:
        raw_i = ds["Data.Toltec.Is"]
        assert not raw_i.variable._in_memory

        mapper = ToltecKidsIOMapper.from_data_source(ds)
        mapped_i = mapper.get_arr(ds, mapper.schema.I)

        assert mapped_i.variable is raw_i.variable
        assert not raw_i.variable._in_memory


def test_real_data_loads_successfully(toltec0_data_path):
    """Test that all available TolTEC files can be loaded."""
    from tolteca_datamodels.toltec import open_toltec_kids

    nc_files = list(toltec0_data_path.glob("*.nc"))
    assert len(nc_files) > 0, "No netCDF files found"

    for nc_file in nc_files:
        with open_toltec_kids(nc_file) as ds:
            meta = ds.toltec_kids.meta
            assert meta.n_chans > 0
            assert meta.array_name in ["a1100", "a1400", "a2000"]


def test_accessor_f_lo_property(toltec_sweep_dataset):
    """A processed sweep coordinate is not a native acquisition LO stream."""
    assert toltec_sweep_dataset.toltec_kids.f_lo is None


def test_processed_frequency_axis_uses_the_supplied_mapping(toltec_sweep_dataset):
    """A mapped frequency array takes precedence over the default coordinate."""
    from dataclasses import replace

    from tolteca_datamodels.toltec.kids.core import ToltecKidsAccessor

    ds = toltec_sweep_dataset
    expected = ds.toltec_kids.get_probe_axis_table()["f_lo"]
    alternate = ds.assign_coords(selected_frequency=ds.sweep).assign_coords(
        sweep=np.zeros(ds.sizes["sweep"])
    )
    mapper = ToltecKidsIOMapper.from_data_source(alternate)
    mapper = ToltecKidsIOMapper(
        {
            field: replace(mapped, name="selected_frequency")
            if field == mapper.schema.f_sweep
            else mapped
            for field, mapped in mapper.mapped_fields.items()
        }
    )
    actual = ToltecKidsAccessor(alternate, mapper=mapper).get_probe_axis_table()["f_lo"]
    np.testing.assert_array_equal(actual, expected)


def test_accessor_metadata_reads_sweep_counts_from_data_variables(
    toltec_sweep_dataset,
):
    """Mapped header variables provide sweep counts."""
    ds = toltec_sweep_dataset.drop_vars("sweep")
    ds["Data.Toltec.LoFreq"] = ("sweep", toltec_sweep_dataset["sweep"].values + 150e6)
    ds["Header.Toltec.NumSweepSteps"] = xr.DataArray(50)
    ds["Header.Toltec.NumSamplesPerSweepStep"] = xr.DataArray(4)

    index = ds.toltec_kids.block_index

    assert index.index.n_sweepsteps.item() == 50
    assert index.meta.n_sweepreps == 4
