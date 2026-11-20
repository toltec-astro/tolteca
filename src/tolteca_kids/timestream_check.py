"""Raw I/Q noise diagnostics for one sweep or timestream block."""

from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import xarray as xr
from pydantic import ConfigDict, Field
from tollan.accessor import NameMapping, Schema, SubpathMapping
from tollan.accessor.xarray import XarrayAccessorBase, XarrayMapper
from tollan.pipeline import Step, StepConfig, StepContext, StepFailure, StepFailureKind
from tollan.utils.xarray import reduce_median_mad

from tolteca_kidsproc.analysis import PSDAnalysis, PSDView

from ._reduction import calculation_settings
from .timestream import TimestreamView


class TimestreamCheckConfig(StepConfig):
    """Raw noise settings shared by sweep and timestream blocks.

    Parameters
    ----------
    psd : PSDAnalysis, optional
        Welch and band-statistic settings. Defaults to 512-sample segments and
        the inclusive 5-20 Hz band, with the other PSDAnalysis defaults.
    force : bool, default False
        Recompute a saved check, including after changing the input samples.
    """

    model_config = ConfigDict(extra="forbid")
    psd: PSDAnalysis = Field(
        default_factory=lambda: PSDAnalysis(
            nperseg=512, stat_f_lims=(5 << u.Hz, 20 << u.Hz)
        ),
        description="Welch spectra and frequency-band statistics of raw I/Q.",
    )
    force: bool = Field(default=False, description="Recompute saved noise results.")


@dataclass
class TimestreamCheckSchema(Schema):
    """Saved noise spectra, normalization and channel frequencies."""

    config: NameMapping = NameMapping.attr("tolteca_kids.timestream_check.config")
    normalized: SubpathMapping = SubpathMapping.path("normalized")
    Sphi: NameMapping = normalized / NameMapping.var("Sphi")
    Sphi_median: NameMapping = normalized / NameMapping.var("Sphi_median")
    Sphi_mad_std: NameMapping = normalized / NameMapping.var("Sphi_mad_std")
    mean_iq_power: NameMapping = normalized / NameMapping.var("mean_iq_power")
    band_mask: NameMapping = normalized / NameMapping.var("band_mask")
    f_chan: NameMapping = normalized / NameMapping.var("f_chan")
    chan: NameMapping = normalized / NameMapping.var("chan")
    chan_dim: NameMapping = normalized / NameMapping.dim("chan")


class TimestreamCheckMapper(XarrayMapper[TimestreamCheckSchema]):
    """Access to saved raw noise diagnostics."""


class TimestreamCheckView(XarrayAccessorBase[TimestreamCheckMapper]):
    """Saved I/Q spectra and normalized noise for one block.

    Parameters
    ----------
    data_source : xarray.DataTree
        Check namespace returned by ``TimestreamCheck.view``. Keep the source
        open while reading lazy arrays. Raw samples are not needed to plot it.
    mapper : TimestreamCheckMapper, optional
        Field mapping; omitted by default, resolving against the supplied node.
    """

    @property
    def data_source(self):
        """Check namespace with saved scientific results."""
        return self._data_source

    @property
    def config(self):
        """Effective settings that produced the saved result."""
        return TimestreamCheckConfig.model_validate_json(
            self.mapper.get_value(self.data_source, self.mapper.schema.config)
        )

    @property
    def psd(self) -> PSDView:
        """I/Q spectra, band statistics and optional common spectra."""
        return PSDView(self.data_source)

    @property
    def normalized(self) -> xr.Dataset:
        """Sphi, its band statistics, normalization and channel frequencies.

        Sphi is (I PSD + Q PSD) / mean(I² + Q²), with units Hz⁻¹ for density
        scaling. It is not the spectrum of the unwrapped phase.
        """
        return self.mapper.get_node(
            self.data_source, self.mapper.schema.normalized
        ).to_dataset()


class TimestreamCheckContext(StepContext["TimestreamCheck", TimestreamCheckConfig]):
    """Live result view and execution status for one checked block."""

    data: TimestreamCheckView | None = Field(default=None, exclude=True)


class TimestreamCheck(Step[TimestreamCheckConfig, TimestreamCheckContext]):
    """Measure raw I/Q noise in one sweep or timestream block.

    Uses every recorded sample in the block, before sweep averaging or despiking.
    The input must already have complete, uniformly sampled data and a recorded
    sampling rate. The step installs computed results without changing raw data.
    """

    @classmethod
    def result_path(cls, *, alias: str | None = None) -> str:
        """Return a check's block-relative path.

        Parameters
        ----------
        alias : str or None, optional
            Invocation alias. None selects the primary check.

        Returns
        -------
        str
            Subtree name for this check.
        """
        return "timestream_check" if alias is None else f"timestream_check_{alias}"

    @classmethod
    def view(
        cls, data: xr.DataTree, *, alias: str | None = None
    ) -> TimestreamCheckView:
        """Read a block's saved noise check.

        Parameters
        ----------
        data : xarray.DataTree
            Working or reopened block containing a completed check.
        alias : str or None, optional
            Invocation alias. None selects the primary check.

        Returns
        -------
        TimestreamCheckView
            Lazy result view independent of live execution contexts.
        """
        return TimestreamCheckView(data[cls.result_path(alias=alias)])

    @classmethod
    def run(cls, data: xr.DataTree, context: TimestreamCheckContext) -> None:
        """Compute and retain one block's raw noise diagnostics.

        Parameters
        ----------
        data : xarray.DataTree
            Single block created by BlockCreate, retaining its raw samples.
        context : TimestreamCheckContext
            Settings, status and destination for the live result view.

        Returns
        -------
        None
            The block retains computed arrays and effective settings. Missing
            raw samples record a missing-input failure.
        """
        cfg = context.config
        path = cls.result_path(alias=cfg.alias)
        if not cfg.force and path in data.children:
            previous = cls.view(data, alias=cfg.alias)
            if calculation_settings(previous.config) == calculation_settings(cfg):
                context.data = previous
                return
        view = TimestreamView(data)
        if view.I is None or view.Q is None:
            context.failures.append(
                StepFailure(
                    kind=StepFailureKind.missing_input,
                    message="Timestream check requires recorded raw I/Q samples.",
                    details={"path": data.path},
                )
            )
            return
        # Select only I/Q through the facade; the generic analysis receives its
        # own typed time-domain view, with the recorded sampling-rate metadata.
        signals = xr.Dataset({"I": view.I, "Q": view.Q}, attrs=view.data_source.attrs)
        signals = signals.chunk(
            {view.chan.dims[0]: cfg.psd.batch_size, view.time.dims[0]: -1}
        )
        source = signals.kids.multi_timestream
        result = xr.DataTree()
        cfg.psd(source, output=result)
        psd = PSDView(result)
        dim = source.time.dims[0]
        power = (source.I.astype(float) ** 2 + source.Q.astype(float) ** 2).mean(dim)
        if source.I.u.unit is not None:
            power = power.u.set(source.I.u.unit**2)
        sphi = ((psd.psd.I + psd.psd.Q) / power).u.set(
            u.Hz**-1 if cfg.psd.scaling == "density" else u.dimensionless_unscaled
        )
        band = xr.ones_like(psd.f_psd, dtype=bool)
        if cfg.psd.stat_f_lims is not None:
            low, high = (bound.to_value(u.Hz) for bound in cfg.psd.stat_f_lims)
            band = (psd.f_psd >= low) & (psd.f_psd <= high)
        median, mad = reduce_median_mad(
            sphi.rename("Sphi").isel(f_psd=band), dim="f_psd"
        )
        axis = data.toltec_kids.block.chan_axis
        frequency = axis.mapper.get_arr(axis.data_source, axis.mapper.schema.f_chan)
        result["normalized"] = xr.DataTree(
            xr.Dataset(
                {
                    "Sphi": sphi,
                    "Sphi_median": median.Sphi,
                    "Sphi_mad_std": mad.Sphi,
                    "mean_iq_power": power,
                    "band_mask": band,
                    "f_chan": xr.DataArray(
                        frequency.data,
                        dims=view.chan.dims,
                        coords={view.chan.dims[0]: view.chan},
                        attrs=frequency.attrs,
                    ),
                }
            )
        )
        result.attrs["tolteca_kids.timestream_check.config"] = cfg.model_dump_json()
        # Finish computation while the input is open; publication and plotting
        # can then use the result without retaining a raw-file handle.
        data[path] = result.compute()
        context.data = cls.view(data, alias=cfg.alias)
