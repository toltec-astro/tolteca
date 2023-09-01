"""A submodule containing TolTEC data models."""


# register io handlers
from . import ncfile  # noqa: F401
from . import table  # noqa: F401
from .core import ToltecData  # noqa: F401
from .types import ToltecDataKind  # noqa: F401
