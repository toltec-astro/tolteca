#!/usr/bin/env python

from .lmt import lmt_info
from .toltec import toltec_info


__all__ = ['lmt_info', 'toltec_info', 'instru_info', 'get_instru_info']


instru_info = {v['instru']: v for v in [lmt_info, toltec_info]}
"""The dict of facts and constants for various instruments."""


def get_instru_info(instru):
    """Return dict of facts and constants for instrument `instru`."""

    if instru not in instru_info:
        raise ValueError(f"invalid instrument key {instru}")
    return instru_info[instru]
