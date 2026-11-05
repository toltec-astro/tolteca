"""Per-block axis values preserve the acquisition and v2 reader conventions."""

import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from astropy import units as u
from pydantic import ValidationError

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.axes import (
    ToltecAxisMapper,
    ToltecAxisView,
)
from tolteca_datamodels.toltec.metadata import ToltecRawObsMetadata
from tolteca_datamodels.toltec.types import ToltecDataKind


def test_axis_selection_preserves_explicit_mapper():
    source = xr.Dataset(
        {
            "block": ("chan_entry", [0, 0, 1, 1]),
            "chan": ("chan_entry", [0, 1, 0, 1]),
            "f_tone": ("chan_entry", [1.0, 2.0, 3.0, 4.0], {"units": "MHz"}),
        }
    )
    mapper = ToltecAxisMapper.from_data_source(source)
    renamed = {"block": "acquisition_block", "f_tone": "tone_frequency"}
    mapper = ToltecAxisMapper(
        {
            field: replace(resolved, name=renamed.get(resolved.name, resolved.name))
            for field, resolved in mapper.mapped_fields.items()
        }
    )
    view = ToltecAxisView(source.rename(renamed), mapper=mapper).select_block(1)
    assert view.mapper is mapper
    assert view.axis_table is view.axis_table
    assert view.axis_table.colnames == ["block", "chan", "f_tone"]
    assert view.axis_table["block"].tolist() == [1, 1]
    assert view.axis_table["f_tone"].to_value(u.MHz).tolist() == [3.0, 4.0]
    assert np.shares_memory(view.axis_table["f_tone"], source["f_tone"].data)


@pytest.mark.parametrize("suffix", ["vnasweep", "targsweep", "tune", ""])
def test_axes_equal_v2(data_lmt_path, tmp_path, suffix):
    interpreter = os.environ.get("TOLTECA_V2_PYTHON")
    if not interpreter:
        pytest.skip("Set TOLTECA_V2_PYTHON for cross-environment parity")
    pattern = f"*1456*_{suffix}.nc" if suffix else "*_09_27_57.nc"
    (path,) = (data_lmt_path / "toltec/tcs/toltec0").glob(pattern)
    reference = tmp_path / "axes.npz"
    subprocess.run(  # noqa: S603 -- explicit local comparison interpreter
        [
            interpreter,
            str(Path(__file__).with_name("v2_axis_reference.py")),
            str(path),
            str(reference),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    chan_names = {
        "chan": "id",
        "tone_mask": "mask_tone",
        "tone_amp": "amp_tone",
        "tone_phase": "phase_tone",
    }
    with open_toltec_kids(path) as ds, np.load(reference) as expected:
        meta = ds.toltec_kids.meta
        assert isinstance(meta, ToltecRawObsMetadata)
        assert (
            meta.obs_type
            == {"vnasweep": "vna", "targsweep": "targ", "tune": "tune", "": "nominal"}[
                suffix
            ]
        )
        assert meta.array == 0
        index = ds.toltec_kids.block_index
        for block in range(index.meta.n_blocks):
            chan = ds.toltec_kids.get_chan_axis_table(block)
            assert np.all(chan["block"] == block)
            for name in set(chan.colnames) - {"block"}:
                np.testing.assert_array_equal(
                    np.asarray(chan[name]),
                    expected[f"{block}_chan_{chan_names.get(name, name)}"],
                )
            assert chan["f_tone"].unit == u.Hz
            assert chan["tone_phase"].unit == u.rad
            if ds.toltec_kids.data_kind & ToltecDataKind.raw_sweep:
                sweep = ds.toltec_kids.get_probe_axis_table(block)
                assert np.all(sweep["block"] == block)
                for name in set(sweep.colnames) - {"block"}:
                    old_name = "id" if name == "step" else name
                    np.testing.assert_array_equal(
                        np.asarray(sweep[name]),
                        expected[f"{block}_sweep_{old_name}"],
                    )
                assert sum(sweep["n_samples"]) == int(
                    index.index.sample_end[block] - index.index.sample_start[block]
                )


def test_actual_repetitions_and_units(real_targsweep_file):
    with open_toltec_kids(real_targsweep_file) as source:
        # A partial last step changes observed counts, not the programmed header.
        ds = source.isel(time=slice(0, -3))
        for name in ("Header.Toltec.ToneFreq", "Data.Toltec.LoFreq"):
            ds[name] = ds[name] / 1e6
            ds[name].attrs["units"] = "MHz"
        meta = ds.toltec_kids.block_index.meta
        assert meta.probe_axis_table is not None
        assert meta.chan_axis_table is not None
        assert meta.n_sweepreps == 10
        assert meta.probe_axis_table["n_samples"][-1] == 7
        assert meta.probe_axis_table["sample_end"][-1] == ds.sizes["time"]
        np.testing.assert_array_equal(
            meta.chan_axis_table["f_tone"].to_value(u.Hz),
            source["Header.Toltec.ToneFreq"].values[0],
        )
        np.testing.assert_array_equal(
            meta.probe_axis_table["f_lo"].to_value(u.Hz),
            np.unique(source["Data.Toltec.LoFreq"].values),
        )


@pytest.mark.parametrize("axis", ["chan", "probe"])
@pytest.mark.parametrize("block", [-1, 1])
def test_native_axis_table_rejects_invalid_block(real_targsweep_file, axis, block):
    with open_toltec_kids(real_targsweep_file) as ds:
        reader = ds.toltec_kids
        assert reader.block_index.meta.n_blocks == 1
        with pytest.raises(ValueError, match="out of range"):
            getattr(reader, f"get_{axis}_axis_table")(block)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("master", "other"),
        ("master_code", 3),
        ("master_code", True),
        ("obs_type", "raw_sweep"),
        ("obs_type_code", 5),
        ("obs_type_code", True),
        ("array", 3),
        ("array", True),
        ("array_name", "other"),
    ],
)
def test_acquisition_codes_are_constrained(real_targsweep_file, name, value):
    with open_toltec_kids(real_targsweep_file) as ds:
        meta = ds.toltec_kids.meta
        assert isinstance(meta, ToltecRawObsMetadata)
        values = meta.model_dump(exclude_computed_fields=True)
    values[name] = value
    with pytest.raises(ValidationError, match=name):
        ToltecRawObsMetadata(**values)


def test_native_channel_selection_keeps_tone_axis_aligned(real_targsweep_file):
    with open_toltec_kids(real_targsweep_file) as ds:
        selected = ds.toltec_kids.select_channels([0, 5, 10])
        assert selected.sizes["time"] == ds.sizes["time"]
        assert selected.sizes["iqlen"] == selected.sizes["toneFreqLen"] == 3
        chan = selected.toltec_kids.get_chan_axis_table(0)
        np.testing.assert_array_equal(
            chan["f_tone"].value,
            ds["Header.Toltec.ToneFreq"].values[0, [0, 5, 10]],
        )


@pytest.mark.parametrize("legacy_header", [False, True])
def test_channel_axes_require_recorded_tone_amplitudes(
    real_targsweep_file, legacy_header
):
    with open_toltec_kids(real_targsweep_file) as source:
        name = "Header.Toltec.ToneAmp"
        ds = (
            source.rename({name: "Header.Toltec.ToneAmps"})
            if legacy_header
            else source.drop_vars(name)
        )
        if legacy_header:
            np.testing.assert_array_equal(
                ds.toltec_kids.get_chan_axis_table(0)["tone_amp"],
                source.toltec_kids.get_chan_axis_table(0)["tone_amp"],
            )
        else:
            with pytest.raises(KeyError):
                _ = ds.toltec_kids.block_index
