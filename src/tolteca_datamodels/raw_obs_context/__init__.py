"""Raw-observation contexts preserving recorded calibration lineage.

Construct these reference models without opening acquisition files. Product
references identify acquisitions; interface names select their logical items.
Derive sweep parents from the owning product's recorded calibration references.
Resolve those references through the owning product indexes when accessing data.
"""

from .artifact import ArtifactBinding
from .calibration import (
    ArrayFlux,
    ArrayPropTableCalItem,
    PhotometryCalItem,
    PointingCalItem,
    PointingOffset,
    SweepCalDataItem,
    SweepCalItem,
)
from .context import RawObsContext, RawObsDataItem
from .model import KidsModelBinding
from .stub import StubApplication, open_acquisition
from .types import RawObsProcRef, RawObsRef
from .workflow import prepare_run_inputs, read_run_indexes, read_run_inputs

__all__ = [
    "ArrayFlux",
    "ArrayPropTableCalItem",
    "ArtifactBinding",
    "KidsModelBinding",
    "PhotometryCalItem",
    "PointingCalItem",
    "PointingOffset",
    "RawObsContext",
    "RawObsDataItem",
    "RawObsProcRef",
    "RawObsRef",
    "StubApplication",
    "SweepCalDataItem",
    "SweepCalItem",
    "open_acquisition",
    "prepare_run_inputs",
    "read_run_indexes",
    "read_run_inputs",
]
