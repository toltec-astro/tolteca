"""Typed invocation and artifact ownership for drive fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field, field_validator
from tollan.pipeline import Step, StepConfig, StepContext

from .artifacts import DriveFitArtifacts
from .computation import fit_sequence
from .result import DriveFitResult

if TYPE_CHECKING:
    import xarray as xr


class DriveFitStepConfig(StepConfig):
    """Scientific settings for one readout network's attenuation sequence.

    Parameters
    ----------
    n_adrvs_min : int, optional
        Minimum distinct attenuation settings, default 5.
    weight_window_Qr : float, optional
        Quality factor defining the Lorentz weighting width, default 12000.
    a_ref : float, optional
        Desired nonlinearity, default 0.2.
    channels : list of int or None, optional
        Recorded channel IDs in requested order. None fits all channels.
    phase_fit_method : {"analytic", "legacy"}, optional
        Centered analytic phase fitting (default) or the unchanged v2 solver.
    phase_fit_maxfev : int, optional
        Evaluation budget per analytic solve, default 200; ignored by legacy.
    phase_fit_retry : bool, optional
        Allow one alternate nonlinear start after failure or an out-of-range
        nonlinearity. Default False keeps runtime and scientific changes small.
        Ignored by legacy. A retry can use another phase_fit_maxfev evaluations.
    artifacts : DriveFitArtifacts, optional
        Scientific and tabular output specifications.
    """

    model_config = ConfigDict(extra="forbid")
    n_adrvs_min: int = Field(default=5, ge=2)
    weight_window_Qr: float = Field(  # noqa: N815 -- physical Qr convention.
        default=12000.0, gt=0, allow_inf_nan=False
    )
    a_ref: float = Field(default=0.2, allow_inf_nan=False)
    channels: list[Annotated[int, Field(ge=0)]] | None = None
    phase_fit_method: Literal["analytic", "legacy"] = "analytic"
    phase_fit_maxfev: int = Field(default=200, ge=1, strict=True)
    phase_fit_retry: bool = False
    artifacts: DriveFitArtifacts = Field(default_factory=DriveFitArtifacts)

    @field_validator("channels")
    @classmethod
    def _validate_channels(cls, value):
        if value is not None and (not value or len(value) != len(set(value))):
            msg = "channels must be nonempty and contain no duplicates"
            raise ValueError(msg)
        return value


class DriveFitStepContext(StepContext["DriveFitStep", DriveFitStepConfig]):
    """Fit execution record with a live scientific result and typed artifacts."""

    data: DriveFitResult | None = Field(default=None, exclude=True)


class DriveFitStep(Step[DriveFitStepConfig, DriveFitStepContext]):
    """Fit one prepared interface and install its scientific results.

    Call the step on an interface returned by :func:`prepare_drive_fit_inputs`.
    Raw files close before phase fitting starts. Numerical failures remain in
    the per-power result flags; invalid sequence inputs fail the invocation.
    """

    @staticmethod
    def result_path(*, alias: str | None = None) -> str:
        """Return the result node name for a fit invocation.

        Parameters
        ----------
        alias : str or None, optional
            Invocation label; None selects the primary fit.

        Returns
        -------
        str
            Child name within the interface tree.
        """
        return "drive_fit" if alias is None else f"drive_fit:{alias}"

    @classmethod
    def view(cls, data: xr.DataTree, *, alias: str | None = None) -> DriveFitResult:
        """Access live or saved scientific results without running a fit.

        Parameters
        ----------
        data : xarray.DataTree
            Fitted interface tree. Keep reopened storage open until values load.
        alias : str or None, optional
            Fit invocation label; None selects the primary result.

        Returns
        -------
        DriveFitResult
            View of the saved scientific dataset.
        """
        return DriveFitResult(data[cls.result_path(alias=alias)].to_dataset())

    @classmethod
    def run(cls, data: xr.DataTree, context: DriveFitStepContext) -> None:
        """Fit a sequence and retain its results and export artifacts.

        Parameters
        ----------
        data : xarray.DataTree
            One prepared interface with native locations in ``inputs/source``.
        context : DriveFitStepContext
            Scientific settings and destination for results and artifacts.

        Returns
        -------
        None
            Results are installed in the tree and exposed through context.data.
        """
        path = cls.result_path(alias=context.config.alias)
        if path in data.children:
            del data[path]
        result = fit_sequence(
            data["inputs"].to_dataset()["source"].values.tolist(), context.config
        )
        data[path] = result.data
        context.data = cls.view(data, alias=context.config.alias)
        context.artifacts = context.config.artifacts(data, context.data)
