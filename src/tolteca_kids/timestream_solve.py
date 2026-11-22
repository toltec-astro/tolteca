"""Solve selected timestream samples with their pinned same-setup sweep model."""

from __future__ import annotations

import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import xarray as xr
from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from tollan.pipeline import Step, StepConfig, StepContext
from tollan.utils.fileloc import fileloc
from tollan.utils.yaml import yaml_dump

from tolteca_datamodels.raw_obs_context import (  # noqa: TC001 -- Pydantic evaluates these annotations
    KidsModelBinding,
    SweepCalItem,
)
from tolteca_kidsproc.timestream import solve_timestream

from .model_calibration import calibration_accumlen
from .model_fit import ModelFit, open_fit
from .timestream_model import solve_fitted_groups

__all__ = [
    "TimestreamSolve",
    "TimestreamSolveConfig",
    "TimestreamSolveContext",
    "solve_calibrated_timestream",
]


class TimestreamSolveConfig(StepConfig):
    """Solver settings for contexts with an exact fitted model.

    Parameters
    ----------
    enabled : bool, optional
        Enable calibrated solving, default False. Requires kids.context.
    backend : {'python', 'kidscpp'}, optional
        Lazy selected-array recovery (default python), or the native subprocess
        for explicit acquisition-stub applications. Neither backend discovers a
        calibration or silently substitutes a different model.
    executable : str, optional
        Kidscpp executable or path, default ``kidscpp``. Invoked without a shell.
    execution : {'seq', 'thr', 'omp'}, optional
        Native execution backend, default ``seq``.
    threads : int, optional
        Native worker count, default 1.
    chunk_samples : int, optional
        Numerical time chunk length, default 4096 samples.
    """

    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    backend: Literal["python", "kidscpp"] = "python"
    executable: str = "kidscpp"
    execution: Literal["seq", "thr", "omp"] = "seq"
    threads: int = Field(default=1, ge=1)
    chunk_samples: int = Field(default=4096, ge=1)

    @model_validator(mode="after")
    def _native_options(self):
        if self.backend == "python" and (
            self.executable != "kidscpp" or self.execution != "seq" or self.threads != 1
        ):
            msg = "Executable, execution and threads require backend='kidscpp'"
            raise ValueError(msg)
        return self


class TimestreamSolveContext(StepContext["TimestreamSolve", TimestreamSolveConfig]):
    """Recorded model binding and solve diagnostics for one block."""

    source: str = ""
    sweep_source: str = ""
    model_source: str = ""
    stubs: list[dict[str, str]] = Field(default_factory=list)
    model: KidsModelBinding | None = None
    native_log: str = ""
    _calibration: SweepCalItem | None = PrivateAttr(default=None)
    _model_index: object = PrivateAttr(default=None)


class TimestreamSolve(Step[TimestreamSolveConfig, TimestreamSolveContext]):
    """Solve one raw block after checking its detector state against its sweep.

    Parameters
    ----------
    config : TimestreamSolveConfig
        Selected solver backend and execution settings.
    source, sweep_source, model_source : str or pathlib.Path, optional
        Local timestream, calibration acquisition and exact fitted ECSV paths.
        Required when running; may be omitted when reconstructing settings only.
    stubs : sequence of dict, optional
        Ordered explicit filepath/mode applications resolved from the saved
        context. Empty selects the fitted ECSV directly. Native reading validates
        target headers and applies each model group before solving.
    calibration : SweepCalItem or None, optional
        Recorded sweep with the exact model bound during input preparation.
        None requires supplying the binding before running.
    model_index : DataProdIndex or None, optional
        Exact model product supplied by input preparation. Python fits are read
        as complete groups from this product and checked against their ECSV.
        None supports the independent-channel table API only.
    """

    def __init__(  # noqa: PLR0913 -- exact calibration binding and optional native model application
        self,
        config,
        *,
        source=None,
        sweep_source=None,
        model_source=None,
        calibration=None,
        stubs=(),
        model_index=None,
    ):
        super().__init__(config)
        self._calibration: SweepCalItem | None = calibration
        self._model_index = model_index
        self._inputs = {
            "source": "" if source is None else str(source),
            "sweep_source": "" if sweep_source is None else str(sweep_source),
            "model_source": "" if model_source is None else str(model_source),
            "stubs": list(stubs),
        }

    def create_context(self, data):
        """Create the block's execution record with its pinned calibration."""
        context = super().create_context(data)
        context._calibration = self._calibration
        context._model_index = self._model_index
        context.model = (
            None if self._calibration is None else self._calibration.data_items[0].model
        )
        for key, value in self._inputs.items():
            setattr(context, key, value)
        return context

    @classmethod
    def run(cls, data, context):
        """Attach complete physical groups, or independent calibrated r/x arrays.

        Parameters
        ----------
        data : xarray.DataTree
            One complete acquired timestream block with its raw source open.
        context : TimestreamSolveContext
            Pinned calibration, local input paths and solver execution settings.

        Returns
        -------
        None
            Python results remain lazy until computed or persisted with the raw
            source open. Native results are loaded before temporary files close.
            DataProdOutput persists the resulting tree.
            Python ModelFit products use ``solved_groups`` with detector and
            probe identities; legacy independent tables use ``solved``.

        Raises
        ------
        ValueError
            The sweep/model identity, detector settings or native output axes
            differ from the selected block. Missing settings also fail.
        subprocess.CalledProcessError
            The native solver fails; no solved node is installed.
        """
        block = data.toltec_kids.block
        calibration = context._calibration
        if calibration is None:
            msg = "TimestreamSolve requires a resolved sweep calibration"
            raise ValueError(msg)
        if context.config.backend == "python":
            if context.stubs:
                msg = "Explicit acquisition stubs require backend='kidscpp'"
                raise ValueError(msg)
            if context._model_index is not None:
                from .product import (  # noqa: PLC0415 -- configuration cycle
                    open_sweep_product,
                )

                with open_sweep_product(
                    context._model_index, interface=block.meta.interface
                ) as product:
                    target = calibration.data_items[0]
                    scope = product.toltec_kids.blocks[target.block].tree
                    if ModelFit.result_path() in scope.children:
                        _solve_python_groups(data, context, scope)
                        return
            solved = solve_calibrated_timestream(
                block,
                calibration,
                sweep_source=context.sweep_source,
                model_source=context.model_source,
                chunk_samples=context.config.chunk_samples,
            )
            # The saved block contract uses sample,chan; the public view uses
            # chan,time and retains the original acquisition sample IDs.
            samples = solved["sample"].values
            solved = solved.drop_vars("sample").rename(time="sample")
            solved = solved.transpose("sample", "chan", ...)
            solved["sample_index"] = ("sample", samples)
            data["solved"] = xr.DataTree(dataset=solved)
            return
        target = calibration.data_items[0]
        with open_fit(context.sweep_source, context.model_source) as (_, sweep):
            if (sweep.meta.uid_raw_obs, sweep.meta.interface, sweep.meta.block) != (
                calibration.raw_obs.uid,
                target.interface,
                target.block,
            ):
                msg = "Model source differs from the pinned calibration"
                raise ValueError(msg)
            _validate_detector_state(sweep, block)
        row = block.parent._row(block.meta.block)
        start, end = int(row.sample_start), int(row.sample_end)
        with TemporaryDirectory(prefix="tolteca-solve-") as directory:
            output = Path(directory) / "solved.nc"
            command = [
                context.config.executable,
                "solve",
                str(fileloc(context.source).path),
                "--blocks",
                str(block.meta.block),
                "--samples",
                str(start),
                str(end),
                "--output",
                str(output),
                "--execution",
                context.config.execution,
                "--threads",
                str(context.config.threads),
                "--chunk_samples",
                str(context.config.chunk_samples),
            ]
            command.extend(
                ["--stubs", yaml_dump(context.stubs)]
                if context.stubs
                else ["--model_params", str(fileloc(context.model_source).path)]
            )
            result = subprocess.run(  # noqa: S603 -- configured executable, literal argv
                command, capture_output=True, text=True, check=False
            )
            context.native_log = result.stdout + result.stderr
            result.check_returncode()
            with xr.open_dataset(output) as native:
                if (
                    native.attrs.get("contract") != "kids.reduction/1"
                    or native.attrs.get("kind") != "SolvedTimeStream"
                ):
                    msg = "Unsupported native solved-timestream contract"
                    raise ValueError(msg)
                if not np.array_equal(
                    native.chan, block.chan_axis.data_source.chan
                ) or not np.array_equal(native.sample_index, np.arange(start, end)):
                    msg = "Native solved axes differ from the selected block"
                    raise ValueError(msg)
                solved = native.load()
        solved.attrs["calibration"] = calibration.model_dump(mode="json")
        data["solved"] = xr.DataTree(dataset=solved)


def _solve_python_groups(data, context, fitted):
    """Bind the full scientific fit to its ECSV and the recorded sweep state."""
    from .model_calibration import calibration_report  # noqa: PLC0415

    block, calibration = data.toltec_kids.block, context._calibration
    target = calibration.data_items[0]
    with open_fit(context.sweep_source, context.model_source) as (report, sweep):
        if (sweep.meta.uid_raw_obs, sweep.meta.interface, sweep.meta.block) != (
            calibration.raw_obs.uid,
            target.interface,
            target.block,
        ):
            msg = "Model source differs from the pinned calibration"
            raise ValueError(msg)
        _validate_detector_state(sweep, block)
        view = ModelFit.view(fitted)
        quality = fitted[f"{ModelFit.result_path()}/quality"].to_dataset().load()
        # The full product and native report must describe the very same fit.
        exported = calibration_report(view, sweep, quality=quality)
        if (
            exported.meta != report.meta
            or exported.colnames != report.colnames
            or any(
                not np.array_equal(exported[name], report[name], equal_nan=True)
                for name in exported.colnames
            )
        ):
            msg = "Saved Python fit differs from its bound model report"
            raise ValueError(msg)
        normalization = calibration_accumlen(block) / calibration_accumlen(sweep)
        solved = solve_fitted_groups(
            block.timestream.data_source[["I", "Q"]],
            view,
            quality,
            normalization=normalization,
            chunk_samples=context.config.chunk_samples,
        )
        solved.attrs.update(
            calibration=calibration.model_dump(mode="json"),
            calibration_source=context.model_source,
        )
        data["solved_groups"] = solved


def solve_calibrated_timestream(  # noqa: PLR0913 -- Bound calibration plus explicit array selection.
    block,
    calibration,
    *,
    sweep_source,
    model_source,
    channels=None,
    samples=None,
    chunk_samples=4096,
):
    """Solve a selected raw view with an exact same-setup calibration.

    Parameters
    ----------
    block : ToltecBlockView
        Open target timestream block. The backing source must remain open until
        the returned arrays are computed or persisted.
    calibration : SweepCalItem
        Pinned calibration acquisition, interface, block and model product.
    sweep_source, model_source : str or pathlib.Path
        Resolved calibration NetCDF and exact kids.fit/1 ECSV artifact. Binding
        and detector settings are validated before selecting any signal samples.
    channels : sequence of int or None, optional
        Original channel labels, in requested order. None selects all channels.
    samples : slice or sequence of int or None, optional
        Positions within this block, applied with isel before numerical loading.
    chunk_samples : int, optional
        Numerical time chunk size, default 4096.

    Returns
    -------
    xarray.Dataset
        Lazy r/x/flags with original chan/time/sample coordinates, exact model
        binding and explicit data/model accumulation normalization.

    Raises
    ------
    ValueError
        The calibration or detector setup differs, or normalization is invalid.
    """
    if len(calibration.data_items) != 1 or calibration.data_items[0].model is None:
        msg = "Require one pinned calibration interface and model"
        raise ValueError(msg)
    target = calibration.data_items[0]
    with open_fit(sweep_source, model_source) as (table, sweep):
        if (sweep.meta.uid_raw_obs, sweep.meta.interface, sweep.meta.block) != (
            calibration.raw_obs.uid,
            target.interface,
            target.block,
        ):
            msg = "Model source differs from the pinned calibration"
            raise ValueError(msg)
        _validate_detector_state(sweep, block)
        normalization = calibration_accumlen(block) / calibration_accumlen(sweep)
        signals = block.timestream.data_source[["I", "Q"]]
        if channels is not None:
            signals = signals.sel(chan=channels)
        if samples is not None:
            signals = signals.isel(time=samples)
        result = solve_timestream(
            signals, table, normalization=normalization, chunk_samples=chunk_samples
        )
        result.attrs.update(
            calibration=calibration.model_dump(mode="json"),
            calibration_source=str(model_source),
            calibration_accumlen=calibration_accumlen(sweep),
        )
    return result


def _validate_detector_state(sweep, timestream):
    """Require the exact recorded tone comb and drive/readout settings."""
    for name in ("roach", "f_lo_center", "atten_drive", "atten_sense"):
        left, right = getattr(sweep.meta, name), getattr(timestream.meta, name)
        if left is None or right is None or left != right:
            msg = f"Calibration detector setting differs or is missing: {name}"
            raise ValueError(msg)
    for name in ("chan", "f_tone", "f_chan", "tone_amp", "tone_phase", "tone_mask"):
        left, right = sweep.chan_axis_table[name], timestream.chan_axis_table[name]
        if not np.array_equal(left, right):
            msg = f"Calibration tone setting differs: {name}"
            raise ValueError(msg)
