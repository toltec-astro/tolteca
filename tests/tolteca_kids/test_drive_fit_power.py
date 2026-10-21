"""Power closure checks independent of the legacy commit normalization."""

import numpy as np
import pytest

from tolteca_kids.drive_fit.power import tone_power_fractions
from tolteca_kids.drive_fit_commit import compute_ampcor


def commit(attenuation, fraction, frequency=None, **kwargs):
    return compute_ampcor(
        attenuation,
        np.zeros(len(attenuation)),
        fraction,
        output_frequency=np.arange(len(attenuation)) * 1e6
        if frequency is None
        else frequency,
        adrv_max=30,
        fill_percentile=50,
        reference_percentile=0,
        reference_step=0.25,
        **kwargs,
    )


def forward_power(amplitudes):
    # Independent original RoachTonePower equation, with arbitrary DAC total 0 dBm.
    return 10 * np.log10(amplitudes**2) - 10 * np.log10(np.sum(amplitudes**2))


def test_two_tone_fixed_total_power_closure():
    result = commit([10, 20], [0.5, 0.5])
    requested = forward_power(np.ones(2)) - [10, 20]
    unquantized = forward_power(result["ampcor"]) - result["adrv_exact"]
    np.testing.assert_allclose(unquantized, requested, atol=1e-12)
    assert result["adrv_exact"] == pytest.approx(12.59637310505756)
    assert result["adrv_ref_total"] == 12.5
    np.testing.assert_allclose(result["power_error_db"], 0.09637310505756)
    np.testing.assert_allclose(result["ampcor"], [1, np.sqrt(0.1)])


def test_reference_full_comb_survives_selection_and_changed_target_count():
    # Two of four equal reference tones remain: each retains 1/4 of the old total.
    result = commit([10, 10], [0.25, 0.25])
    assert result["adrv_exact"] == pytest.approx(10 + 10 * np.log10(2))
    np.testing.assert_allclose(
        forward_power(result["ampcor"]) - result["adrv_exact"],
        10 * np.log10(0.25) - 10,
    )


@pytest.mark.parametrize("attenuation", [0, 10, 10.25, 15.75])
def test_uniform_scaling_and_unchanged_comb(attenuation):
    amplitudes = np.array([1.0, 0.5, 0.25])
    frequencies = np.array([1e6, 2e6, 3e6])
    fraction = tone_power_fractions(amplitudes, frequencies)
    np.testing.assert_array_equal(
        fraction, tone_power_fractions(amplitudes * 7, frequencies)
    )
    result = commit([attenuation] * 3, fraction)
    np.testing.assert_allclose(result["ampcor"], amplitudes)
    assert result["adrv_ref_total"] == attenuation


def test_dac_duplicates_masks_and_conflicting_requests():
    frequency = [1e6, 1e6, 2e6, 3e6]
    fraction = tone_power_fractions(
        [100, 1, 1, 30], frequency, [True, True, True, False]
    )
    np.testing.assert_array_equal(fraction, [0.5, 0.5, 0.5, 0])
    result = commit(
        [10, 10, 20, 10], fraction, frequency, output_mask=[True, True, True, False]
    )
    assert result["adrv_exact"] == pytest.approx(12.59637310505756)
    assert np.isnan(result["power_error_db"][-1])
    with pytest.raises(ValueError, match="sharing a DAC bin"):
        commit([10, 20, 10], [0.5, 0.5, 0.5], [1e6, 1e6, 2e6])
    with pytest.raises(ValueError, match="positive reference power"):
        commit([10, 10], [0, 1])
    with pytest.raises(ValueError, match="exceeds the fixed DAC total"):
        commit([0, 0], [0.8, 0.8])


def test_fit_subset_persists_full_reference_normalization(
    drive_fit_calibration_sequence,
):
    from tolteca_kids.drive_fit.computation import _fit, _read_sweep
    from tolteca_kids.drive_fit.step import DriveFitStepConfig

    sweep = _read_sweep(drive_fit_calibration_sequence[0], [0])
    axis = sweep.meta.chan_axis_table
    expected = tone_power_fractions(
        axis["tone_amp"], axis["f_tone"].value, axis["tone_mask"]
    )
    result = _fit(
        [sweep], np.array([sweep.meta.atten_drive]), DriveFitStepConfig(channels=[0])
    )
    assert result.data.power_fraction.item() == expected[0]
    assert result.data.power_fraction.item() < 1
    assert result.data.attrs["reference_comb_channels"] == 685
