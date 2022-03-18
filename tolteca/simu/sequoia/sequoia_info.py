#!/usr/bin/env python


from copy import deepcopy
from ..lmt import lmt_info
from ...common.sequoia import sequoia_info as _sequoia_info


__all__ = ['sequoia_info']


def _make_extended_sequoia_info(sequoia_info):
    """Extend the sequoia_info dict with array properties related to
    the simulator.
    """
    sequoia_info = deepcopy(sequoia_info)

    # add lmt info as sequoia site_info
    sequoia_info['site'] = lmt_info
    return sequoia_info


sequoia_info = _make_extended_sequoia_info(_sequoia_info)
"""The sequoia info dict with additional items related to simulator."""
