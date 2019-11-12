#! /usr/bin/env python

"""This module implements the container classes for KIDs data."""

from astropy.nddata import NDDataRef
from ..common.timestream import TimeStreamMixin
from .sweep_mixin import SweepMixin


class Sweep(SweepMixin, NDDataRef):
    """Container for frequency sweep data."""
    pass


class VnaSweep(Sweep):
    pass


class TargetSweep(Sweep):
    pass


class TimeStream(TimeStreamMixin, NDDataRef):
    """Container for time stream data."""
    pass


class RawTimeStream(TimeStream):
    """Container for the raw in-phase (i) and quadrature (q) data."""
    pass


class SolvedTimeStream(TimeStream):
    """Container for the solved detuning (x) and loss (r) data."""
    pass
