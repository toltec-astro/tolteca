"""Welch windows, segment sizes and in-band signal peaks."""

import numpy as np
import pytest
from astropy import units as u
from scipy.signal import welch

from tolteca_kidsproc.accessors import make_kids_dataset
from tolteca_kidsproc.analysis import PSDAnalysis


@pytest.mark.parametrize("window", ["hann", "hamming", "blackman"])
@pytest.mark.parametrize("nperseg", [256, 512, 2048])
def test_windows_and_segments_match_scipy(window, nperseg):
    rng = np.random.default_rng(13)
    source = make_kids_dataset(
        rng.normal(size=(3, 4096)), rng.normal(size=(3, 4096)), f_smp=1000 << u.Hz
    )
    analysis = PSDAnalysis(window=window, nperseg=nperseg)
    result = analysis(source.kids.multi_timestream)
    view = result.kids.psd
    f, expected = welch(source.I.values, fs=1000, window=window, nperseg=nperseg)
    np.testing.assert_array_equal(view.f_psd, f)
    np.testing.assert_allclose(view.psd.I, expected, rtol=1e-12)
    assert view.psd_median.I.dims == ("chan",)
    assert view.psd_analysis == analysis


def test_selected_band_contains_in_band_tone():
    time = np.arange(10000) / 1000
    signal = sum(np.sin(2 * np.pi * f * time) for f in (5, 50, 200))
    source = make_kids_dataset(signal, np.zeros_like(signal), f_smp=1000 << u.Hz)
    view = PSDAnalysis(nperseg=2048, stat_f_lims=(20 << u.Hz, 100 << u.Hz))(
        source.kids.timestream
    ).kids.psd
    band = view.psd.I.sel(f_psd=slice(20, 100))
    peak = band.argmax("f_psd").compute().item()
    assert 45 < band.f_psd[peak].item() < 55
    np.testing.assert_allclose(view.psd_median.I, np.median(band))
