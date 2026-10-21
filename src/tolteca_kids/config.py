"""Configuration models for KIDs processing and product output."""

from pathlib import Path

from pydantic import ConfigDict, Field, model_validator
from tollan.config import FrozenBaseModel
from tollan.config.types import (
    AbsAnyPath,  # noqa: TC002 -- Pydantic resolves the annotation at runtime
)
from tollan.pipeline import StepConfig

from tolteca_datamodels.data_prod_namespace import (
    NamespacePath,  # noqa: TC001 -- Pydantic field
)
from tolteca_datamodels.data_prod_run import (
    PathComponent,
    RunRequest,
)
from tolteca_kidsproc.analysis import PSDAnalysis

from .kids_find import KidsFindConfig
from .quicklook import KidsQuicklook
from .sweep_check import SweepCheckConfig
from .sweep_check_plot import SweepCheckPlotConfig
from .tlaloc_output import TlalocOutputConfig
from .tone_prepare import TonePrepareConfig

__all__ = [
    "DataProdOutputConfig",
    "InputConfig",
    "KidsConfig",
    "SweepReduceConfig",
    "TimestreamReduceConfig",
]


class InputConfig(StepConfig):
    """Selection of inputs for a KIDs run.

    Parameters
    ----------
    select : str, optional
        Query on the combined, header-populated input table. Omitted by
        default, considering all rows.
    raise_on_errors : bool, optional
        Raise for selected inputs that cannot be processed. Default False
        skips these rows with debug logs. Filename diagnostics alone do not
        exclude an input with successfully read acquisition metadata.
    enabled : bool, optional
        Apply the query when True (default). False considers all rows;
        selected-input validation still applies.
    """

    model_config = ConfigDict(extra="forbid")
    select: str | None = None
    raise_on_errors: bool = False


class SweepReduceConfig(StepConfig):
    """Sweep-reduction configuration.

    Parameters
    ----------
    block : int, optional
        Block exposed by the result context. Default is -1, selecting the last
        block. Every sweep block in the input collection is reduced.
    force : bool, optional
        Whether to recompute existing reduced data. Default is False.
    compute_uncertainty : bool, default True
        Include the population standard deviation at each probe step.
    """

    model_config = ConfigDict(extra="forbid")
    block: int = Field(default=-1, ge=-1)
    force: bool = Field(default=False, description="Recompute existing reduced data.")
    compute_uncertainty: bool = True


class TimestreamReduceConfig(StepConfig):
    """Block selection and Python spectral reduction settings.

    Parameters
    ----------
    block : int, optional
        Block exposed by the result context; -1 selects the last block (default).
        Every timestream block in the input collection is reduced.
    force : bool, optional
        Recompute existing results when True; default False.
    psd : PSDAnalysis or None, optional
        Spectral settings. Defaults to PSDAnalysis(); None retains signals only.
    compute_r_x : bool, default True
        Include spectra of supplied r/x signals; does not solve raw I/Q.
    compute_amplitude_phase : bool, default True
        Include magnitude and unwrapped raw-phase spectra.
    """

    model_config = ConfigDict(extra="forbid")
    block: int = Field(default=-1, ge=-1)
    force: bool = False
    psd: PSDAnalysis | None = Field(default_factory=PSDAnalysis)
    compute_r_x: bool = True
    compute_amplitude_phase: bool = True


class DataProdOutputConfig(StepConfig):
    """Product-output configuration.

    Parameters
    ----------
    enabled : bool, optional
        Whether to write products. Default is False.
    registration : bool, optional
        Register published indexes through ``data_prod.index`` in DPDB. Default
        False. Registration failure retains the published files for retry.
    path : pathlib.Path, optional
        Output root. Relative paths, including the default ``Path(".")``, are
        resolved against the validation context's root_path when provided,
        otherwise against the current directory.
    namespace : str, optional
        Store namespace, default ``scratch``. Products are placed beneath their
        primary acquisition identifier and logical product identifier.
    run : RunRequest, optional
        Optional invocation UUID and readable name. An omitted UID is generated
        once per invocation. The name defaults to its UUID text, or accepts a
        literal name or counter policy. Reservations remain consumed after failure.
    tag : str or None, optional
        Reduction flavor to advance after publication, default ``main``.
        Labels such as ``ql-main`` are explicit; the namespace does not set them.
        None leaves all tags unchanged. Concurrent publication to the same
        product directory and tag fails.
    n_chans_per_chunk : int, optional
        Maximum channels per Zarr chunk for raw samples and reduced sweeps,
        default 16. Larger values favor reading many channels together;
        smaller values favor individual channels.
    n_samples_per_chunk : int, optional
        Maximum raw time samples per Zarr chunk, default 1024. Reduced sweep
        chunks retain the complete probe axis.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    registration: bool = False
    path: AbsAnyPath = Field(default=Path())
    namespace: NamespacePath = "scratch"
    run: RunRequest = Field(default_factory=RunRequest)
    tag: PathComponent | None = "main"
    n_chans_per_chunk: int = Field(default=16, ge=1)
    n_samples_per_chunk: int = Field(default=1024, ge=1)


class KidsConfig(FrozenBaseModel):
    """KIDs processing configuration.

    Parameters
    ----------
    input : InputConfig, optional
        Input-selection settings. If omitted, keep all prepared inputs.
    sweep_reduce : SweepReduceConfig, optional
        Sweep-reduction settings. If omitted, use their defaults.
    timestream_reduce : TimestreamReduceConfig, optional
        Block selection and Python Welch settings for raw timestreams.
    sweep_check : SweepCheckConfig, optional
        Sweep flags, despiking and D21 diagnostics. Disabled by default.
    sweep_check_plot : SweepCheckPlotConfig, optional
        Standalone diagnostic figures for saved checks. Disabled by default.
    kids_find : KidsFindConfig, optional
        Resonance finding and probe matching. Disabled by default.
    tone_prepare : TonePrepareConfig, optional
        Prepare the v2 next-tone files. Disabled by default.
    tlaloc_output : TlalocOutputConfig, optional
        Install the last sweep block's prepared files after enabled publication
        and registration succeed. Disabled by default.
    quicklook : KidsQuicklook, optional
        Acquisition feedback from completed sweep checks. Disabled by default.
    output : DataProdOutputConfig, optional
        Product-output settings. If omitted, output is disabled.
    """

    model_config = ConfigDict(extra="forbid")
    input: InputConfig = Field(default_factory=InputConfig)
    sweep_reduce: SweepReduceConfig = Field(default_factory=SweepReduceConfig)
    timestream_reduce: TimestreamReduceConfig = Field(
        default_factory=TimestreamReduceConfig
    )
    sweep_check: SweepCheckConfig = Field(
        default_factory=lambda: SweepCheckConfig(enabled=False)
    )
    sweep_check_plot: SweepCheckPlotConfig = Field(
        default_factory=lambda: SweepCheckPlotConfig(enabled=False)
    )
    kids_find: KidsFindConfig = Field(
        default_factory=lambda: KidsFindConfig(enabled=False)
    )
    tone_prepare: TonePrepareConfig = Field(default_factory=TonePrepareConfig)
    tlaloc_output: TlalocOutputConfig = Field(default_factory=TlalocOutputConfig)
    quicklook: KidsQuicklook = Field(default_factory=KidsQuicklook)
    output: DataProdOutputConfig = Field(default_factory=DataProdOutputConfig)

    @model_validator(mode="after")
    def _tone_prerequisites(self):
        if self.kids_find.enabled and not self.sweep_check.enabled:
            msg = "kids_find requires sweep_check"
            raise ValueError(msg)
        if self.tone_prepare.enabled and not self.kids_find.enabled:
            msg = "tone_prepare requires kids_find"
            raise ValueError(msg)
        if self.tlaloc_output.enabled and not self.tone_prepare.enabled:
            msg = "tlaloc_output requires tone_prepare"
            raise ValueError(msg)
        return self
