#!/usr/bin/env python


from tollan.utils.log import get_logger

from ..base import ProjModel
from .sequoia_info import sequoia_info


__all__ = ['SequoiaSkyProjModel']


class SequoiaSkyProjModel(ProjModel):
    """
    A model to transform SEQUOIA pixel positions to
    absolute world coordinates for given telescope bore sight target
    and time of obs.

    The output coordinate frame is a generic sky lon/lat frame which
    can represent any of the valid celestial coordinate frames supported,
    by specifying the ``evaluate_frame`` keyword argument.
    """

    logger = get_logger()

    observer = sequoia_info['site']['observer']

    def evaluate(self):
        return NotImplemented
