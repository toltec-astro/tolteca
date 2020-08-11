#! /usr/bin/env python


# the registry has to be loaded first in order to correctly reload the
# submodules
from .registry import open_file  # noqa: F401

# load the io classes
from .toltec import *  # noqa: F401, F403
