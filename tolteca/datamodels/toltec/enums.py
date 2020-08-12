#! /usr/bin/env python

from enum import IntEnum
from ..io.toltec.kidsdata import KidsDataKind  # noqa: F401
from ..io.toltec.table import TableKind  # noqa: F401

__all__ = ['KidsDataKind', 'TableKind', 'RawObsMaster', 'RawObsType']


class RawObsMaster(IntEnum):
    """The is in line with the ``toltec/master`` table."""

    TCS = 0
    ICS = 1
    CLIP = 2

    @property
    def desc(self):
        return {
            0: 'The Telescope Control System.',
            1: 'The Instrument Control System.',
            2: 'The ROACH Manager.'
            }.get(self.value, None)


class RawObsType(IntEnum):
    """The is in line with the ``toltec/obstype`` table."""

    Nominal = 0
    Timestream = 1
    VNA = 2
    TARG = 3
    TUNE = 4

    @property
    def desc(self):
        return {
            0: 'Nominal observation.',
            1: 'Time stream probed at a set of tones.',
            2: 'Blind sweep in the full frequency range.',
            3: 'Sweep around a set of tones.',
            4: 'Two consecutive target sweeps with reduction.'
            }.get(self.value, None)
