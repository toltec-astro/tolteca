import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table

from tolteca_kids.roach_tone import (
    RoachToneProps,
    RoachTonePropsMetadata,
    TlalocEtcDataStore,
)


def test_roach_tone_props():
    tbl = Table()
    tbl["f_tone"] = np.arange(-10, 10, dtype=float)
    tbl["amp_tone"] = 0.5
    tbl["phase_tone"] = np.zeros((len(tbl),), dtype=float)
    tbl["mask_tone"] = 1
    tbl["mask_tone"][:5] = 0

    rtp = RoachToneProps(tbl, f_lo=1.0)

    assert isinstance(rtp.meta, RoachTonePropsMetadata)
    assert rtp.meta.table_validated
    assert rtp.meta.f_lo == 1.0 << u.Hz

    assert rtp.n_chans == len(tbl)
    assert rtp.n_tones == 15
    np.testing.assert_array_equal(rtp.mask, tbl["mask_tone"])
    np.testing.assert_array_equal(rtp.f_tones, tbl["f_tone"][5:] << u.Hz)
    with rtp.no_mask():
        np.testing.assert_array_equal(rtp.mask, tbl["mask_tone"])
        np.testing.assert_array_equal(rtp.f_tones, tbl["f_tone"] << u.Hz)


def test_tlaloc_etc():
    with tempfile.TemporaryDirectory() as _tmp:
        tmp = Path(_tmp)
        path = tmp / "tlaloc_etc"
        tlaloc = TlalocEtcDataStore.create(path=path)
        assert tlaloc.path.samefile(path)

        # make sample data
        tbl = Table()
        tbl["f_tone"] = range(-10, 10)
        tbl["amp_tone"] = 0.5
        tbl["phase_tone"] = RoachToneProps.make_random_phases(len(tbl))
        tbl["mask_tone"] = 1
        tbl["mask_tone"][:5] = 0
        tlaloc.write_tone_props(tbl, roach=1, f_lo=100)
        # TODO: reenable these
        # assert paths == {
        #     "targ_phases": path / "toltec1/random_phases.dat",
        #     "targ_amps": path / "toltec1/default_targ_amps.dat",
        #     "targ_freqs": path / "toltec1/targ_freqs.dat",
        #     "targ_mask": path / "toltec1/default_targ_masks.dat",
        # }
        rtp = tlaloc.read_tone_props(roach=1)
        assert rtp.meta.f_lo == 100 << u.Hz
