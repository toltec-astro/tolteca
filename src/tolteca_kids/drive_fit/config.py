"""Root runtime configuration for the drive-fit workflow."""

from pydantic import ConfigDict, Field
from tollan.config import FrozenBaseModel

from tolteca_kids.config import DataProdOutputConfig

from .plot import DriveFitPlotStepConfig
from .step import DriveFitStepConfig

__all__ = ["DriveFitConfig"]


class DriveFitConfig(FrozenBaseModel):
    """Workflow settings in the runtime's root ``drive_fit`` section.

    Parameters
    ----------
    fit : DriveFitStepConfig, optional
        Sequence selection, phase-fitting settings and scientific artifacts.
    plot : DriveFitPlotStepConfig, optional
        Saved-result diagnostics; disabled by default.
    output : DataProdOutputConfig, optional
        Product-store path, namespace, run naming, tag and DPDB registration.
        Disabled by default. Drive fits use the registered ``dp_drivefit`` type.
    select : str or None, optional
        Metadata query on discovered inputs. None retains every target/tune sweep.

    Examples
    --------
    >>> from tollan.config import RuntimeContext
    >>> from tolteca_kids.drive_fit import DriveFit
    >>> runtime = RuntimeContext({"drive_fit": {"fit": {"a_ref": 0.15}}})
    >>> DriveFit(runtime).config.fit.a_ref
    0.15
    """

    model_config = ConfigDict(extra="forbid")
    fit: DriveFitStepConfig = Field(default_factory=DriveFitStepConfig)
    plot: DriveFitPlotStepConfig = Field(
        default_factory=lambda: DriveFitPlotStepConfig(enabled=False)
    )
    output: DataProdOutputConfig = Field(default_factory=DataProdOutputConfig)
    select: str | None = None
