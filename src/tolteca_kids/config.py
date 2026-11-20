"""Configuration models for KIDs processing and product output."""

from pathlib import Path

from pydantic import ConfigDict, Field, model_validator
from tollan.config import FrozenBaseModel
from tollan.config.types import (
    AbsAnyPath,  # noqa: TC002 -- Pydantic resolves the annotation at runtime
    PathSegment,  # noqa: TC002 -- Pydantic resolves the annotation at runtime
    SubPath,  # noqa: TC002 -- Pydantic field type
)
from tollan.pipeline import StepConfig

from tolteca_datamodels.data_prod import (
    CounterRunNameGenerator,
    RunName,
    RunUid,
)
from tolteca_kidsproc.analysis import PSDAnalysis

from .context import KidsContextConfig
from .kids_find import KidsFindConfig
from .kids_find_plot import KidsFindPlotConfig
from .model_fit import ModelFitConfig
from .model_fit_plot import ModelFitPlotConfig
from .quicklook import KidsQuicklook
from .sweep_check import SweepCheckConfig
from .sweep_check_plot import SweepCheckPlotConfig
from .timestream_check import TimestreamCheckConfig
from .timestream_check_plot import TimestreamCheckPlotConfig
from .timestream_inspect import TimestreamInspectConfig
from .timestream_solve import TimestreamSolveConfig
from .tlaloc_output import TlalocOutputConfig
from .tone_prep import TonePrepConfig

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
    run_name : str or CounterRunNameGenerator or None, optional
        Namespace-local directory name or counter policy. None uses the execution
        UID. Reserved names remain consumed after failed outputs.
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
    namespace: SubPath = Field(default="scratch", validate_default=True)
    run_name: RunName | CounterRunNameGenerator | None = None
    tag: PathSegment | None = "main"
    n_chans_per_chunk: int = Field(default=16, ge=1)
    n_samples_per_chunk: int = Field(default=1024, ge=1)


class KidsConfig(FrozenBaseModel):
    """KIDs processing configuration.

    Parameters
    ----------
    run_uid : UUID or None, optional
        Caller-assigned execution identity. None generates one UID per invocation.
    run_meta : dict, optional
        Execution metadata, including optional orchestration correlation values.
    input : InputConfig, optional
        Input-selection settings. If omitted, keep all prepared inputs.
    model_fit : ModelFitConfig or None, optional
        Fit acquired sweep channels with the Python grouped fitter or explicitly
        select kidscpp for native target/Tune reports. None disables fitting.
        Automatic Python plans require saved KidsFind results. Python fits support
        VNA sweeps; fixed-probe recovery remains separately assessed.
    model_fit_plot : ModelFitPlotConfig or None, optional
        HTML summary, group and channel-resonator diagnostics from saved Python
        results. None disables plotting. Product output exports retained figures.
    context : KidsContextConfig or None, optional
        Raw inventories, sweep model selections and optional replay run. None
        retains uncalibrated processing. Requires timestream_solve.enabled.
    timestream_inspect : TimestreamInspectConfig or None, optional
        Calibrated native-window and slow-trend inspection policy. None omits
        inspection. Requires context and enabled calibrated solving.
    timestream_solve : TimestreamSolveConfig, optional
        Native I/Q solving with the context's pinned model. Disabled by default.
    sweep_reduce : SweepReduceConfig, optional
        Sweep-reduction settings. If omitted, use their defaults.
    timestream_reduce : TimestreamReduceConfig, optional
        Block selection and Python Welch settings for raw timestreams.
    timestream_check : TimestreamCheckConfig, optional
        Raw I/Q noise diagnostics on sweep and timestream blocks. Disabled by default.
    timestream_check_plot : TimestreamCheckPlotConfig, optional
        Figures from saved raw noise checks. Disabled by default.
    sweep_check : SweepCheckConfig, optional
        Sweep flags, despiking and D21 diagnostics. Disabled by default.
    sweep_check_plot : SweepCheckPlotConfig, optional
        Standalone diagnostic figures for saved checks. Disabled by default.
    kids_find : KidsFindConfig, optional
        Resonance finding and probe matching. Disabled by default.
    kids_find_plot : KidsFindPlotConfig, optional
        Finding and matching diagnostic figures. Disabled by default.
    tone_prep : TonePrepConfig, optional
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
    run_uid: RunUid | None = None
    run_meta: dict = Field(default_factory=dict)
    input: InputConfig = Field(default_factory=InputConfig)
    context: KidsContextConfig | None = None
    model_fit: ModelFitConfig | None = None
    model_fit_plot: ModelFitPlotConfig | None = None
    timestream_inspect: TimestreamInspectConfig | None = None
    timestream_solve: TimestreamSolveConfig = Field(
        default_factory=TimestreamSolveConfig
    )
    sweep_reduce: SweepReduceConfig = Field(default_factory=SweepReduceConfig)
    timestream_check: TimestreamCheckConfig = Field(
        default_factory=lambda: TimestreamCheckConfig(enabled=False),
        description="Optional raw I/Q noise check for sweep and timestream blocks.",
    )
    timestream_check_plot: TimestreamCheckPlotConfig = Field(
        default_factory=lambda: TimestreamCheckPlotConfig(enabled=False),
        description="Optional saved raw noise plots.",
    )
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
    kids_find_plot: KidsFindPlotConfig = Field(
        default_factory=lambda: KidsFindPlotConfig(enabled=False)
    )
    tone_prep: TonePrepConfig = Field(default_factory=TonePrepConfig)
    tlaloc_output: TlalocOutputConfig = Field(default_factory=TlalocOutputConfig)
    quicklook: KidsQuicklook = Field(default_factory=KidsQuicklook)
    output: DataProdOutputConfig = Field(default_factory=DataProdOutputConfig)

    @model_validator(mode="after")
    def _tone_prerequisites(self):
        if (self.context is not None) != self.timestream_solve.enabled:
            msg = (
                "Calibrated solving requires both kids.context "
                "and timestream_solve.enabled"
            )
            raise ValueError(msg)
        if (
            self.timestream_inspect is not None
            and self.timestream_inspect.enabled
            and not self.timestream_solve.enabled
        ):
            msg = "Timestream inspection requires calibrated solving and kids.context"
            raise ValueError(msg)
        if self.kids_find_plot.enabled and not self.kids_find.enabled:
            msg = "kids_find_plot requires kids_find"
            raise ValueError(msg)
        if self.kids_find.enabled and not self.sweep_check.enabled:
            msg = "kids_find requires sweep_check"
            raise ValueError(msg)
        if self.tone_prep.enabled and not self.kids_find.enabled:
            msg = "tone_prep requires kids_find"
            raise ValueError(msg)
        if self.tlaloc_output.enabled and not self.tone_prep.enabled:
            msg = "tlaloc_output requires tone_prep"
            raise ValueError(msg)
        return self
