"""Python timestream reduction in the acquisition/block tree."""

from __future__ import annotations

# ruff: noqa: E741 -- Preserve physical I/Q names.
import numpy as np
import xarray as xr
from astropy import units as u
from pydantic import ConfigDict, Field
from tollan.pipeline import PIPELINE_CONTEXT_KEY, Step, StepContext

from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kidsproc.analysis import PSDAnalysis, PSDMapper

from ._reduction import calculation_settings, ensure_blocks, select_block
from .blocks import NODE_TYPE
from .config import TimestreamReduceConfig
from .timestream import TimestreamView, ToltecTimestreamMapper

__all__ = ["TimestreamReduce", "TimestreamReduceContext"]


class TimestreamReduceContext(StepContext["TimestreamReduce", TimestreamReduceConfig]):
    """Execution settings and selected acquisition block."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    data: TimestreamView | None = Field(
        default=None,
        exclude=True,
        description="Live block signals and spectra; excluded from JSON export.",
    )
    block: int = Field(default=0, description="Selected block index after reduction.")


class TimestreamReduce(Step[TimestreamReduceConfig, TimestreamReduceContext]):
    """Reduce timestream blocks with optional Python spectral estimation."""

    @classmethod
    def run(cls, data: xr.DataTree, context: TimestreamReduceContext) -> None:
        """Reduce all timestream blocks and publish the selected block's view.

        Parameters
        ----------
        data : xarray.DataTree
            Acquisition, collection, block, or native data at the root. Native
            data is moved to the acquisition ``raw`` leaf before reduction.
        context : TimestreamReduceContext
            Execution settings; ``block`` records the selected acquisition ID and
            ``data`` publishes its signal and spectral view.

        Returns
        -------
        None
            Reduced blocks and lazy spectral graphs are attached to the working
            tree. Export or computation evaluates the spectra.

        Raises
        ------
        ValueError
            An input block is not a timestream or native conversion would
            overwrite an unrelated existing node.
        """
        ensure_blocks(data)
        selected = select_block(data, context.config.block)
        context.block = selected.meta.block
        blocks = (
            [selected]
            if data.attrs.get(NODE_TYPE) == "block"
            else data.toltec_kids.blocks
        )
        for block in blocks:
            if not block.data_kind & ToltecDataKind.timestream:
                msg = "TimestreamReduce requires a timestream block"
                raise ValueError(msg)
            if context.config.alias is None:
                cls._reduce_scope(block.tree, context.config)
            else:
                cls._run_alias(block, context)
        context.data = cls.view(data, block=context.block, alias=context.config.alias)

    @classmethod
    def _run_alias(cls, block, context):
        """Install one named reduction without replacing the primary results."""
        target = block.tree
        result_path = f"timestream_{context.config.alias}"
        source = block.detach()
        psd_mapper = PSDMapper.from_data_source(source)
        psd_path = psd_mapper.namespace_path
        alias_psd_path = f"{psd_path}_{context.config.alias}"
        if result_path in source.children:
            source["timestream"] = source[result_path]
        if alias_psd_path in source.children:
            source[psd_path] = source[alias_psd_path]
        cls._reduce_scope(source, context.config)
        target[result_path] = source["timestream"]
        if context.config.psd is not None:
            target[alias_psd_path] = source[psd_path]
        elif alias_psd_path in target.children:
            del target[alias_psd_path]

    @classmethod
    def view(
        cls, data: xr.DataTree, *, block: int, alias: str | None = None
    ) -> TimestreamView:
        """Access time signals and spectra for the invocation's selected block.

        Parameters
        ----------
        data : xarray.DataTree
            Working or reopened product tree.
        block : int
            Acquisition block ID recorded by the reduction context.
        alias : str or None, optional
            Invocation label. None selects the primary reduction.

        Returns
        -------
        TimestreamView
            Fresh view of the selected block's available time and PSD domains.
        """
        node = select_block(data, block).tree
        if alias is None:
            return TimestreamView(node)
        source = node.toltec_kids.block.detach()
        source["timestream"] = source[f"timestream_{alias}"]
        mapper = PSDMapper.from_data_source(source)
        path = mapper.namespace_path
        if f"{path}_{alias}" in source.children:
            source[path] = source[f"{path}_{alias}"]
        elif path in source.children:
            del source[path]
        return TimestreamView(source)

    @classmethod
    def _reduce_scope(cls, scope, config):
        view = TimestreamView(scope)
        mapper = ToltecTimestreamMapper.from_defaults()
        config_key = mapper.get_name(mapper.schema.reduction_config)
        unchanged = view.data_source.attrs.get(config_key) == calculation_settings(
            config
        )
        source = view.data_source
        signals = cls._prepare_signals(source)
        signals.attrs[config_key] = calculation_settings(config)
        if "meta" in scope.attrs:
            signals.attrs["meta"] = scope.attrs["meta"]
        signals.attrs["kids_node_type"] = "timestream"
        psd_mapper = PSDMapper.from_data_source(scope)
        try:
            saved = psd_mapper.get_namespace(scope)
        except KeyError:
            saved = None
        if config.psd is not None:
            analysis = _ToltecPSDAnalysis.model_validate(config.psd.model_dump())
            analysis(
                cls._spectral_signals(signals, config).kids.timestream,
                output=scope,
                force=config.force or not unchanged,
            )
        elif saved is not None:
            del scope[psd_mapper.namespace_path]
        if config.force or not unchanged:
            scope["timestream"] = xr.DataTree(dataset=signals)

    @staticmethod
    def _prepare_signals(ds):
        """Prepare recorded samples on shared time and channel coordinates."""
        mapper = ToltecTimestreamMapper.from_data_source(ds)
        mapper.validate_has_field(mapper.schema.I)
        mapper.validate_has_field(mapper.schema.Q)
        I = mapper.get_arr(ds, mapper.schema.I)
        Q = mapper.get_arr(ds, mapper.schema.Q)
        if I.dims != Q.dims or I.shape != Q.shape:
            msg = "I and Q must have the same dimensions and shape"
            raise ValueError(msg)
        time_dim = mapper.get_arr(ds, mapper.schema.time).dims[0]
        if time_dim not in I.dims:
            msg = f"Could not find time dimension {time_dim!r}"
            raise ValueError(msg)
        mapper.validate_has_field(mapper.schema.f_smp)
        names = [mapper.get_name(mapper.schema.I), mapper.get_name(mapper.schema.Q)]
        names.extend(
            mapper.get_name(field)
            for field in (mapper.schema.r, mapper.schema.x)
            if mapper.has_arr(ds, field)
        )
        signals = ds[names].copy(deep=False)
        signals.attrs.pop(PIPELINE_CONTEXT_KEY, None)
        time = mapper.get_arr(ds, mapper.schema.time)
        if time.u.unit is None:
            time = time.u.set("s")
        signals = signals.assign_coords(time=time)
        if mapper.schema.chan_dim in mapper:
            chan_dim = mapper.get_name(mapper.schema.chan_dim)
            if chan_dim != "chan":
                signals = signals.rename({chan_dim: "chan"})
        return signals

    @staticmethod
    def _spectral_signals(signals, config):
        """Select recorded observables and derive lazy magnitude and phase."""
        mapper = ToltecTimestreamMapper.from_data_source(signals)
        fields = [mapper.schema.I, mapper.schema.Q]
        if config.compute_r_x:
            fields.extend(
                field for field in (mapper.schema.r, mapper.schema.x) if field in mapper
            )
        signals = signals[[mapper.get_name(field) for field in fields]]
        I = mapper.get_arr(signals, mapper.schema.I)
        Q = mapper.get_arr(signals, mapper.schema.Q)
        time_dim = mapper.get_arr(signals, mapper.schema.time).dims[0]
        if config.compute_amplitude_phase:
            chunks = (
                {dim: config.psd.batch_size for dim in I.dims if dim != time_dim}
                if config.psd is not None
                else {}
            )
            signals = signals.chunk({**chunks, time_dim: -1})
            I, Q = (
                signals[mapper.get_name(mapper.schema.I)],
                signals[mapper.get_name(mapper.schema.Q)],
            )
            signals["amplitude"] = xr.apply_ufunc(
                np.hypot, I.astype(float), Q.astype(float), dask="allowed"
            )
            signals["amplitude"].attrs = dict(I.attrs)
            phase = np.arctan2(Q, I).chunk({time_dim: -1})
            signals["phase"] = (
                xr.apply_ufunc(
                    np.unwrap,
                    phase,
                    input_core_dims=[[time_dim]],
                    output_core_dims=[[time_dim]],
                    dask="parallelized",
                    output_dtypes=[float],
                )
                .u.unset()
                .u.set(u.rad)
            )
        return signals


class _ToltecPSDAnalysis(PSDAnalysis):
    """Apply the shared PSD workflow to the prepared TolTEC observables."""

    def _get_signals(self, timestream_view):
        return timestream_view.data_source
