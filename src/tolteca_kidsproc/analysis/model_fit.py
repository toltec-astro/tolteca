"""Physical fixed-group resonance fitting through generic KIDs sweep views."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import xarray as xr
from pydantic import ConfigDict, Field
from tollan.accessor import NameMapping, SubpathMapping
from tollan.accessor.xarray import (
    Analysis,
    AnalysisMapper,
    AnalysisSchemaMixin,
    AnalysisView,
    ensure_analysis_scope,
)

from .._model_tables import make_calibration
from ..accessors.views import SweepView
from ..calibration import (
    calibration_components,
    calibration_pair_model,
    evaluate_calibration,
    group_covariance,
    validate_calibration,
)
from ..fitting import fit_kids_model

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt
    from lmfit import Parameters

__all__ = [
    "KidsModelAnalysis",
    "KidsModelMapper",
    "KidsModelSchema",
    "KidsModelView",
]


@dataclass
class KidsModelSchema(AnalysisSchemaMixin):
    """Namespace and compact result locations for physical resonance fitting."""

    _namespace: SubpathMapping = SubpathMapping.path(__name__)
    analysis: NameMapping = NameMapping.attr(f"{__name__}.analysis")
    fit_subpath: SubpathMapping = SubpathMapping.path("fit")
    samples_subpath: SubpathMapping = SubpathMapping.path("samples")
    joint_subpath: SubpathMapping = SubpathMapping.path("joint")
    chan_prop_subpath: SubpathMapping = SubpathMapping.path("chan_prop")


class KidsModelMapper(AnalysisMapper[KidsModelSchema]):
    """Locate the four calibration datasets in an analysis namespace."""


class KidsModelAnalysis(Analysis[KidsModelMapper]):
    """Fit a fixed segment plan with physical resonances and channel gains.

    Parameters
    ----------
    gain_family : {"affine", "delay"}, default "affine"
        Complex affine gain or exponential amplitude slope plus delay.
    initialization : {"rational", "supplied"}, default "rational"
        First pole seed. Invalid rational seeds use the supplied plan states,
        with the fallback recorded. D21/S21 candidate finding precedes this call.
    max_nfev : int, default 2000
        Lmfit evaluation limit per fitting group.
    qr_bounds : pair of float, default (500,100000)
        Loaded-Q bounds, applied in fitted loss coordinates.
    rank_rtol : float, default 1e-8
        Relative singular-value cutoff in scaled optimizer coordinates.
    nonlinear : bool, default False
        Fit a shared nonlinearity a_nl per resonance. This describes sweep
        shape, not fixed-power
        timestream response or multitone nonlinear dynamics.
    nonlinear_branch : {"unique", "increasing", "decreasing"}, default "unique"
        Unique response below bifurcation, or a monotonic sweep that starts
        below/above resonance. The caller supplies this history. Directional
        fits retain local covariance, conditional on the selected branch.
    a_nl_max : float, default 0.75
        Positive finite nonlinearity upper bound. Above 0.75, supply an explicit
        directional branch with nonlinear=True. Zero-drive frequency bounds
        can differ from the measured dip location.

    Notes
    -----
    The only optimizer is lmfit. Each call recomputes all supplied groups.
    Numerical work is generic S21 analysis; no instrument setup matching,
    programmed phase normalization or automatic detector finding is performed.
    """

    model_config = ConfigDict(extra="forbid")

    gain_family: Literal["affine", "delay"] = Field(
        default="affine", description="Physical per-channel gain family."
    )
    initialization: Literal["rational", "supplied"] = Field(
        default="rational",
        description=(
            "Pole initializer; invalid rational seeds record a supplied-state fallback."
        ),
    )
    max_nfev: int = Field(
        default=2000, ge=1, description="Maximum lmfit function evaluations per group."
    )
    qr_bounds: tuple[float, float] = Field(
        default=(500.0, 100000.0),
        strict=False,
        description="Positive increasing loaded-Q bounds; accepts a YAML/JSON pair.",
    )
    rank_rtol: float = Field(
        default=1e-8,
        gt=0,
        lt=1,
        description="Relative rank cutoff in scaled optimizer coordinates.",
    )
    nonlinear: bool = Field(
        default=False,
        description=(
            "Fit one shared stationary softening parameter a_nl per resonance."
        ),
    )

    nonlinear_branch: Literal["unique", "increasing", "decreasing"] = Field(
        default="unique",
        description="Stationary response history for nonlinear sweeps.",
    )
    a_nl_max: float = Field(
        default=0.75,
        gt=0,
        allow_inf_nan=False,
        description="Upper bound on fitted a_nl.",
    )

    compute_fit = staticmethod(fit_kids_model)

    def make_fit(  # noqa: PLR0913 -- Measured inputs and optional fit constraints.
        self,
        sweep_view: SweepView,
        plan: xr.Dataset,
        *,
        mask: npt.ArrayLike | None = None,
        sigma: npt.ArrayLike | None = None,
        probe_frequency: npt.ArrayLike | None = None,
        initial_params: Mapping[str, Parameters] | None = None,
    ) -> xr.DataTree:
        """Fit a segment plan and return the four compact result datasets.

        Parameters
        ----------
        sweep_view : SweepView or MultiSweepView
            Generic measured means with aligned frequency and I/Q arrays. Samples
            occupy the last axis. Frequencies are converted to Hz; signal units
            must match. Fitting loads arrays; keep source files open until return.
        plan : xarray.Dataset
            One ``row`` axis with ``group_id,chan,resonator_id,sample_start,
            sample_stop,fr_initial,r_initial``. Frequencies are Hz unless units
            are declared; loss is dimensionless ``1/(2*Qr)``. Intervals are
            original zero-based half-open sample positions. Optional paired
            ``fr_search_min,fr_search_max`` give bounds in Hz. Repeated resonance
            states must agree. A resonator belongs to one result group.
        mask : array_like of bool, optional
            Inclusion mask with signal shape, intersected with unioned segments
            and finite-data validity. Excluded samples retain their predictions.
        sigma : array_like, optional
            Positive standard deviations of I/Q means in signal units, with
            shape ``S21.shape + (2,)``. Omit for equal weighting and residual-scaled
            covariance. Quadrature/sample correlations are not supported yet.
        probe_frequency : array_like, optional
            Actual probe frequencies in Hz, in acquired channel order. NaN means
            unbound. Missing probes remain missing; no nominal value is invented.
            A supplied probe inside its window is also the fixed gain origin.
        initial_params : mapping of str to lmfit.Parameters, optional
            Group-keyed starts and constraints from the same model coordinates
            and scales. Fixed values, bounds and expressions are retained.
            Omitted groups use the configured initializer. Unknown groups fail.

        Returns
        -------
        xarray.DataTree
            Standalone ``fit,samples,joint,chan_prop`` datasets. Failed optimizer
            outcomes are retained. A converged sweep remains unassessed for
            fixed-probe recovery unless the measurement count establishes failure.
            Channels without plan rows remain present in ``chan_prop``.

        Raises
        ------
        ValueError
            Plan identity, bounds, sample selection, data or weights are invalid.
        """
        return make_calibration(
            sweep_view,
            plan,
            gain_family=self.gain_family,
            initialization=self.initialization,
            max_nfev=self.max_nfev,
            qr_bounds=self.qr_bounds,
            rank_rtol=self.rank_rtol,
            nonlinear=self.nonlinear,
            nonlinear_branch=self.nonlinear_branch,
            a_nl_max=self.a_nl_max,
            mask=mask,
            sigma=sigma,
            probe_frequency=probe_frequency,
            initial_params=initial_params,
        )

    def __call__(  # noqa: PLR0913 -- Measured inputs and analysis destination.
        self,
        sweep_view: SweepView,
        plan: xr.Dataset,
        *,
        mask: npt.ArrayLike | None = None,
        sigma: npt.ArrayLike | None = None,
        probe_frequency: npt.ArrayLike | None = None,
        initial_params: Mapping[str, Parameters] | None = None,
        output: xr.DataTree | None = None,
        parent_namespace: str | None = None,
    ) -> xr.DataTree:
        """Compute and install the analysis namespace with its configuration.

        Parameters
        ----------
        sweep_view : SweepView or MultiSweepView
            Measured sweep; see ``make_fit`` for shape, units and lifetime.
        plan : xarray.Dataset
            Explicit channel-resonator segment plan described by ``make_fit``.
        mask, sigma, probe_frequency : array_like, optional
            Sample inclusion, I/Q uncertainties and actual probes; see ``make_fit``.
        initial_params : mapping of str to lmfit.Parameters, optional
            Group-keyed starts and constraints; see ``make_fit``.
        output : xarray.DataTree, optional
            Destination scope. Without it, preserve input under ``_input``.
        parent_namespace : str, optional
            Existing scope inside output. Requires an output tree.

        Returns
        -------
        xarray.DataTree
            The destination tree, or a standalone scope. Read its result through
            ``.kids.model_fit``. Every call recomputes the supplied plan.

        Raises
        ------
        TypeError
            The input is not a sweep view or output is not a DataTree.
        ValueError
            Namespace selection or numerical inputs are invalid.
        """
        if not isinstance(sweep_view, SweepView):
            msg = "KidsModelAnalysis requires a generic sweep view"
            raise TypeError(msg)
        if output is not None and not isinstance(output, xr.DataTree):
            msg = "Fit output must be a DataTree"
            raise TypeError(msg)
        if output is None and parent_namespace is not None:
            msg = "Selecting a parent namespace requires output"
            raise ValueError(msg)
        scope = ensure_analysis_scope(
            sweep_view.data_source if output is None else output,
            parent_namespace=parent_namespace,
        )
        mapper = self._mapper_cls.from_data_source(scope)

        def compute():
            result = self.make_fit(
                sweep_view,
                plan,
                mask=mask,
                sigma=sigma,
                probe_frequency=probe_frequency,
                initial_params=initial_params,
            )
            return {
                getattr(mapper.schema, f"{name}_subpath"): result[name].to_dataset()
                for name in ("fit", "samples", "joint", "chan_prop")
            }

        self._call(scope, mapper=mapper, force=True, compute=compute)
        return scope if output is None else output

    def _get_output_subpaths(self, *, mapper):
        return tuple(
            getattr(mapper.schema, f"{name}_subpath")
            for name in ("fit", "samples", "joint", "chan_prop")
        )


class KidsModelView(AnalysisView[KidsModelMapper, KidsModelAnalysis]):
    """Read compact physical fit results and their producing configuration.

    Parameters
    ----------
    data_source : xarray.DataTree
        Scope containing the analysis namespace. ``from_namespace`` accepts a
        selected, detached or reopened namespace. Keep backing files open while
        reading lazy arrays or evaluating responses.
    mapper : KidsModelMapper, optional
        Analysis namespace and dataset mappings.
    """

    def _dataset(self, name):
        return self.mapper.get_node(
            self.data_source, getattr(self.mapper.schema, f"{name}_subpath")
        ).to_dataset()

    def _tables(self):
        return xr.DataTree.from_dict(
            {
                name: self._dataset(name)
                for name in ("fit", "samples", "joint", "chan_prop")
            }
        )

    @property
    def fit(self) -> xr.Dataset:
        """Channel-resonator-segment rows with tied physical parameter copies."""
        return self._dataset("fit")

    @property
    def samples(self) -> xr.Dataset:
        """Original sample identities, measured IQ, predictions and residuals."""
        return self._dataset("samples")

    @property
    def joint(self) -> xr.Dataset:
        """Packed full physical covariance, solver diagnostics and seed records."""
        return self._dataset("joint")

    @property
    def chan_prop(self) -> xr.Dataset:
        """Acquired channel order, selected group, probe binding and recovery flags."""
        return self._dataset("chan_prop")

    @property
    def fit_analysis(self) -> KidsModelAnalysis:
        """Saved producing configuration."""
        return self._analysis_snapshot

    def validate(self):
        """Validate shared copies; raise ValueError on inconsistent parameters."""
        validate_calibration(self._tables())

    def components(self):
        """Build a response-by-component working view.

        Returns
        -------
        xarray.Dataset
            Padded components with resonator IDs and a validity mask. Several
            segments for one association produce one component slot. Gain arrays
            have only the response axis. Shared values are validated first.
        """
        return calibration_components(self._tables())

    def pair_model(self, group_id, chan, resonator_id):
        """Select a channel-resonator view without assigning a primary.

        Parameters
        ----------
        group_id, chan, resonator_id : scalar
            Explicit group, acquired-channel and resonator labels. Selection
            validates stored parameter copies and loads the model coefficients.
            Keep backing files open until this call returns.

        Returns
        -------
        KidsSweepWithAffineGain or KidsSweepWithDelayGain
            Detached model snapshot with ``eval``, ``rotate`` and ``derotate``.
            Its prediction contains the selected resonance alone. Derotation
            retains neighbours in measured IQ. Use ``predict`` for the complete
            group/channel response and ``covariance`` for joint uncertainty.
            Selecting a pair does not change channel properties or fit status.

        Raises
        ------
        KeyError
            The association does not exist.
        ValueError
            Stored calibration relations or parameter copies disagree.
        """
        return calibration_pair_model(self._tables(), group_id, chan, resonator_id)

    def predict(self, group_id, chan, frequency):
        """Evaluate physical responses at specified probe frequencies.

        Parameters
        ----------
        group_id, chan, frequency : array_like
            Broadcastable group labels, acquired channel labels and positive
            frequencies in Hz. Each group/channel must exist in the fit table.

        Returns
        -------
        numpy.ndarray
            Complex predictions in measured signal units, with the broadcast
            input shape. Validation loads the required result columns.

        Raises
        ------
        KeyError
            The group/channel response does not exist.
        ValueError
            Frequency is invalid or shared parameter copies disagree.
        """
        return evaluate_calibration(self._tables(), group_id, chan, frequency)

    def covariance(self, group_id):
        """Read one full physical covariance block.

        Parameters
        ----------
        group_id : str
            Joint fitting group label.

        Returns
        -------
        xarray.DataArray
            Square covariance with physical parameter labels and unit coordinates.
            Cross-group uncertainty is not represented by this local block.

        Raises
        ------
        KeyError
            The group does not exist.
        """
        return group_covariance(self._tables(), group_id)
