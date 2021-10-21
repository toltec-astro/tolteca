#!/usr/bin/env python


class OffsetMappingModel(object):
    """The base class for mapping models defined in offset coordinates."""
    pass


class TrajMappingModel(object):
    """The base class for mapping models in absolute coordinates."""
    pass


class OffsetTrajMappingModel(TrajMappingModel):
    """The class for trajectories rendered using `OffsetMappingModel`.
    """
    pass
