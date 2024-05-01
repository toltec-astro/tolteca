import astropy.units as u
import numpy as np
from astropy.table import Table

from tolteca_kids.roach_tone import RoachToneProps, RoachTonePropsMetadata


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
    pass
    # some tests

    # test_path = "test_tone_utils_tlaloc_etc"
    # tlaloc = TlalocEtcDataStore.create(test_path, exist_ok=True)
    #
    # # make sample data
    #
    # tbl = Table()
    #
    # tbl["f_comb"] = range(-10, 10)
    # tbl["amp_tone"] = 0.5
    # tbl["phase_tone"] = tlaloc.make_random_phases(len(tbl))
    # tbl["mask_tone"] = 1
    # tbl["mask_tone"][:5] = 0
    #
    # paths = tlaloc.write_tone_prop_table(tbl, nw=1, flo=100)
    # print(paths)
    #
    # rtp = tlaloc.get_tone_props(nw=1)
    # print(rtp)
    # print(rtp.table)
    #
    # paths2 = tlaloc.write_tone_prop_table(rtp, nw=2)
    #
    # paths3 = tlaloc.backup_tone_prop_files(nw=2)
    #
    # rtp2 = rtp[:2]
    # print(rtp2)
    # print(rtp2.table)
    # tlaloc.write_tone_prop_table(rtp2, nw=2)
    #
