"""Fit acquired-channel models with a Python or native pipeline backend."""

from __future__ import annotations

import json
import subprocess
from contextlib import contextmanager, nullcontext
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
import xarray as xr  # noqa: TC002 -- Pydantic and Sphinx resolve annotations
from astropy.table import Table
from pydantic import ConfigDict, Field, PrivateAttr, model_validator
from tollan.accessor.xarray import ensure_datatree
from tollan.config.types import AbsAnyPath  # noqa: TC002 -- Pydantic field type
from tollan.pipeline import ArtifactContext, Step, StepConfig, StepContext
from tollan.utils.fileloc import fileloc

from tolteca_datamodels.toltec import open_toltec_kids
from tolteca_datamodels.toltec.types import ToltecDataKind
from tolteca_kidsproc.analysis import (  # noqa: TC001 -- Pydantic and Sphinx resolve annotations
    KidsModelAnalysis,
    KidsModelView,
)
from tolteca_kidsproc.model_flags import ModelFitFlag

from .block_create import BlockCreate
from .blocks import (
    ToltecBlockView,  # noqa: TC001 -- Sphinx resolves public annotations.
)
from .model_calibration import (
    PHYSICAL_SCHEMAS,
    calibration_accumlen,
    calibration_report,
)
from .model_fit_artifacts import KIDSCPP_FILES, ModelFitArtifacts
from .model_fit_quality import ModelFitQualityConfig
from .power_setup import PowerSetupConfig  # noqa: TC001 -- Pydantic field type.
from .tlaloc_output import TlalocKidsModel

_MODEL_PARAMETERS = (
    "fp",
    "Qr",
    "Qc",
    "fr",
    "A",
    "normI",
    "normQ",
    "slopeI",
    "slopeQ",
    "interceptI",
    "interceptQ",
)
_MODEL_SCHEMA = {
    "name": "S21WithGainLinTrend",
    "version": 1,
    "parameters": [
        {"name": name, "unit": "Hz" if name in {"fp", "fr"} else ""}
        for name in _MODEL_PARAMETERS
    ],
}


@contextmanager
def open_fit(source, report):
    """Validate a native report with its source kept open for the consumer."""
    table = Table.read(report, format="ascii.ecsv")
    _validate_report(table)
    with open_toltec_kids(source) as ds:
        tree = BlockCreate()(ensure_datatree(ds))
        blocks = tree.toltec_kids.blocks
        block_id = table.meta.get("block")
        if type(block_id) is not int or not 0 <= block_id < len(blocks):
            msg = "Native fit must identify an existing acquisition block"
            raise ValueError(msg)
        block = blocks[block_id]
        if not block.data_kind & ToltecDataKind.sweep or block.meta.obs_type == "vna":
            msg = "Native fit requires a target or Tune sweep"
            raise ValueError(msg)
        _validate_binding(table, block)
        yield table, block


def _validate_report(table):
    from .model_joint import validate_joint_models  # noqa: PLC0415

    schema = table.meta.get("model_schema")
    supported = [_MODEL_SCHEMA, *PHYSICAL_SCHEMAS.values()]
    supported.extend(
        {**schema, "version": 1, "parameters": schema["parameters"][:-1]}
        for schema in PHYSICAL_SCHEMAS.values()
    )
    if (
        table.meta.get("schema") != "kids.fit/1"
        or schema not in supported
        or table.meta.get("model") != schema["name"]
    ):
        msg = "Fit requires kids.fit/1 with a supported model schema"
        raise ValueError(msg)
    parameters = {p["name"]: p["unit"] for p in schema["parameters"]}
    required = {"f_out", "f_in", "flag", "chan", *parameters}
    if not required.issubset(table.colnames):
        msg = "Native fit is missing frequency, model, flag or channel columns"
        raise ValueError(msg)
    _validate_report_columns(table, required, parameters)
    flags = np.asarray(table["flag"])
    if "joint_models" in table.meta:
        validate_joint_models(table.meta["joint_models"], table["chan"], table["f_in"])
    if np.any((flags < 0) | (flags > np.iinfo(np.uint32).max)):
        msg = "Native fit flags must fit an unsigned 32-bit value"
        raise ValueError(msg)
    for name in ("f_out", "f_in", *parameters):
        values = np.asarray(table[name])
        bad = np.isinf(values) | (
            np.isnan(values)
            & (
                (flags & (ModelFitFlag.fit_unusable | ModelFitFlag.fit_joint_required))
                == 0
            )
        )
        if name in {"f_out", "f_in"}:
            bad |= ~np.isfinite(values) | (values <= 0)
        if bad.any():
            msg = f"Native fit column {name} contains invalid values"
            raise ValueError(msg)


def _validate_report_columns(table, required, parameters):
    for name in required:
        column = table[name]
        integer = name in {"flag", "chan"}
        expected = np.dtype("int64" if integer else "float64")
        if column.dtype != expected or np.ma.getmaskarray(column).any():
            msg = f"Native fit column {name} must be unmasked {expected}"
            raise ValueError(msg)
        unit = "Hz" if name in {"f_out", "f_in"} else parameters.get(name, "")
        if str(column.unit or "") != unit:
            msg = f"Native fit column {name} has incorrect units"
            raise ValueError(msg)


def _validate_binding(table, block):
    meta = block.meta
    expected = {
        "Header.Toltec.ObsNum": meta.obsnum,
        "Header.Toltec.SubObsNum": meta.subobsnum,
        "Header.Toltec.ScanNum": meta.scannum,
        "Header.Toltec.RoachIndex": meta.roach,
    }
    if any(table.meta.get(key) != value for key, value in expected.items()):
        msg = "Native fit acquisition identity differs from the source"
        raise ValueError(msg)
    probes = block.chan_axis_table
    if len(table) != meta.n_chans or not np.array_equal(table["chan"], probes["chan"]):
        msg = "Native fit must retain every acquired channel in its original order"
        raise ValueError(msg)
    if not np.array_equal(table["f_in"], probes["f_chan"].to_value("Hz")):
        msg = "Native fit input frequencies differ from the acquired channel binding"
        raise ValueError(msg)
    accumlen = table.meta.get("Header.Toltec.AccumLen")
    if (
        not isinstance(accumlen, (int, float))
        or not np.isfinite(accumlen)
        or accumlen <= 0
        or accumlen != calibration_accumlen(block)
    ):
        msg = "Native fit accumulation length differs from its acquisition"
        raise ValueError(msg)
    statuses = table.meta.get("acquisition_status", {}).get("blocks", [])
    selected = [status for status in statuses if status.get("block") == meta.block]
    if len(selected) != 1 or selected[0].get("state") != "complete":
        msg = "Native fit requires a completed source block"
        raise ValueError(msg)


class ModelFitConfig(StepConfig):
    """
    Grouped Python fitting or an explicitly selected native invocation.

    Parameters
    ----------
    backend : {'python', 'kidscpp'}, optional
        Python grouped physical fitting (default), or the existing native
        per-channel GainLinTrend fitter. These are different model contracts.
    analysis : KidsModelAnalysis or None, optional
        Python lmfit settings. None uses the analysis defaults. Native runs
        cannot supply Python analysis settings.
    quality : ModelFitQualityConfig, optional
        Python fit warning thresholds. Masks are saved separately from solver
        results and do not authorize timestream solving.
    power_setup : PowerSetupConfig or None, optional
        Independent acquired-comb power diagnostics and external references.
        None disables them. Requires recorded attenuation and the complete comb.
    fit_n_fwhm : float, optional
        Half-width of each candidate's sample window in its measured FWHM,
        default 2. Used only when constructing a plan from KidsFind.
    use_uncertainty : bool, optional
        Weight by repeat standard deviation divided by sqrt(repeat count),
        default True. Assumes independent repeats. False uses equal weights.
    finding_alias, check_alias : str or None, optional
        Saved finding and quality-check selections. None selects each primary
        result. Explicit plans do not require finding results.
    executable : pathlib.Path or None, optional
        Required for backend='kidscpp'; forbidden for Python. Relative paths
        resolve from the runtime.
    threads : int, optional
        Native worker count, at least one; defaults to one.
    execution : {'seq', 'thr', 'omp'}, optional
        Native execution policy; defaults to sequential.
    retain_processed : bool, optional
        Retain the native processed sweep NetCDF in addition to fit tables,
        command and log. Defaults to False.
    artifacts : ModelFitArtifacts, optional
        Writers for the common model table and optional Kidscpp diagnostics.
    """

    model_config = ConfigDict(extra="forbid")
    backend: Literal["python", "kidscpp"] = "python"
    analysis: KidsModelAnalysis | None = None
    quality: ModelFitQualityConfig = Field(default_factory=ModelFitQualityConfig)
    power_setup: PowerSetupConfig | None = None
    fit_n_fwhm: float = Field(default=2.0, gt=0, allow_inf_nan=False)
    use_uncertainty: bool = True
    finding_alias: str | None = None
    check_alias: str | None = None
    executable: AbsAnyPath | None = None
    threads: int = Field(default=1, ge=1)
    execution: Literal["seq", "thr", "omp"] = "seq"
    retain_processed: bool = False
    artifacts: ModelFitArtifacts = Field(default_factory=ModelFitArtifacts)

    @model_validator(mode="after")
    def validate_execution(self):
        """Reject configurations that mix Python and native execution settings."""
        python_fields = {
            "analysis",
            "fit_n_fwhm",
            "use_uncertainty",
            "finding_alias",
            "check_alias",
        }
        if self.backend == "kidscpp" and self.executable is None:
            msg = "The kidscpp backend requires executable"
            raise ValueError(msg)
        if self.backend == "kidscpp" and (
            self.analysis is not None
            or self.power_setup is not None
            or self.quality != ModelFitQualityConfig()
            or any(
                getattr(self, name) != type(self).model_fields[name].default
                for name in python_fields - {"analysis"}
            )
        ):
            msg = "Python fitting settings cannot accompany executable"
            raise ValueError(msg)
        if self.backend == "python" and (
            self.executable is not None
            or self.threads != 1
            or self.execution != "seq"
            or self.retain_processed
        ):
            msg = "Native execution settings require backend='kidscpp'"
            raise ValueError(msg)
        return self


class ModelFitContext(StepContext["ModelFit", ModelFitConfig]):
    """Backend execution record; Python results reside in the block tree."""

    source: str = ""
    directory: str = ""
    sample_end: int = 0
    _plan: xr.Dataset | None = PrivateAttr(default=None)


class ModelFit(Step[ModelFitConfig, ModelFitContext]):
    """
    Fit acquired channels in one measured sweep block.

    Parameters
    ----------
    config : ModelFitConfig or dict, optional
        Python analysis or native invocation settings. Omitted uses Python.
    plan : xarray.Dataset or None, optional
        Explicit kidsproc association-segment plan for this block. None builds
        a plan from saved KidsFind groups. The step copies supplied plans.
    source : str or pathlib.Path or FileLoc, optional
        Local acquisition containing this block. Required for native execution.
    directory : str or pathlib.Path or None, optional
        Existing native output directory for acquisition hooks. None uses a
        temporary directory and retains file contents for product output.
    sample_end : int or None, optional
        Exclusive completed sample boundary in the acquisition. None derives
        it from the block index. Hooks supply the bound acquisition snapshot.
    """

    def __init__(
        self, config=None, *, plan=None, source=None, directory=None, sample_end=None
    ):
        super().__init__(
            ModelFitConfig()
            if config is None
            else ModelFitConfig.model_validate(config)
        )
        if self.config.backend == "kidscpp" and plan is not None:
            msg = "Explicit segment plans require Python fitting"
            raise ValueError(msg)
        self._plan = None if plan is None else plan.copy(deep=True)
        self.source = "" if source is None else str(fileloc(source).path)
        self.directory = "" if directory is None else str(directory)
        self.sample_end = sample_end

    def create_context(self, data):
        context = super().create_context(data)
        context._plan = self._plan
        context.source, context.directory, context.sample_end = (
            self.source,
            self.directory,
            self.sample_end
            if self.sample_end is not None
            else (
                int(
                    data.toltec_kids.block.parent._row(
                        data.toltec_kids.block.meta.block
                    ).sample_end
                )
                if self.config.backend == "kidscpp"
                else 0
            ),
        )
        return context

    @classmethod
    def run(cls, data, context):
        block = data.toltec_kids.block
        if not block.data_kind & ToltecDataKind.sweep:
            msg = "Model fitting requires a sweep block"
            raise ValueError(msg)
        if context.config.backend == "python":
            from ._model_fit import fit_python  # noqa: PLC0415 -- step-config cycle

            result = fit_python(block, context.config, context._plan)
            report = calibration_report(
                result.kids.model_fit, block, quality=result["quality"].dataset
            )
            _validate_report(report)
            _validate_binding(report, block)
            data["tlaloc_kids_model"] = TlalocKidsModel.from_table(report).data
            stream = StringIO()
            report.write(stream, format="ascii.ecsv")
            context.artifacts = ArtifactContext()
            context.artifacts.add(
                context.config.artifacts.model, stream.getvalue().encode()
            )
            data[cls.result_path(alias=context.config.alias)] = result
            return
        if block.meta.obs_type == "vna":
            msg = "Model fitting requires a complete target or Tune sweep"
            raise ValueError(msg)
        if not context.source:
            msg = "Native model fitting requires a source acquisition"
            raise ValueError(msg)
        area = (
            nullcontext(context.directory)
            if context.directory
            else TemporaryDirectory(prefix="tolteca-fit-")
        )
        with area as directory:
            cls._fit(data, context, Path(directory))

    @classmethod
    def result_path(cls, *, alias: str | None = None) -> str:
        """Return the block-relative Python result path.

        Parameters
        ----------
        alias : str or None, optional
            Invocation label. None selects the primary result.

        Returns
        -------
        str
            Result subtree containing the segment plan and generic analysis.
        """
        return "model_fit" if alias is None else f"model_fit_{alias}"

    @classmethod
    def view(
        cls, data: xr.DataTree, *, block: int = -1, alias: str | None = None
    ) -> KidsModelView:
        """Read Python fit results from a working or reopened acquisition.

        Parameters
        ----------
        data : xarray.DataTree
            Block, acquisition or block collection with saved Python results.
        block : int, optional
            Block ID; -1 selects the last block when data is an acquisition.
        alias : str or None, optional
            ModelFit invocation alias, by default None.

        Returns
        -------
        KidsModelView
            Generic compact fit view. Keep its store open for lazy access.
            Use pair_model with explicit channel and resonator labels to derotate.

        Raises
        ------
        KeyError
            The selected Python result is absent, including native-only runs.
        """
        from ._reduction import select_block  # noqa: PLC0415

        scope = select_block(data, block).tree
        return scope[cls.result_path(alias=alias)].kids.model_fit

    @classmethod
    def peak_links(
        cls, data: xr.DataTree, *, block: int = -1, alias: str | None = None
    ) -> Table:
        """Read original D21 peak links to fitted components.

        Parameters
        ----------
        data : xarray.DataTree
            Block or acquisition containing saved Python ModelFit results.
        block : int, optional
            Block ID; -1 selects the last block of an acquisition.
        alias : str or None, optional
            ModelFit invocation alias, by default None.

        Returns
        -------
        astropy.table.Table
            Materialized peak_id, group_id, resonator_id, status and reason.
            Rejected peaks have masked component keys. Colliding peaks retain
            candidate keys with unresolved status. A linked correspondence is
            finding provenance, not a detector identification or fit-quality pass.

        Raises
        ------
        KeyError
            Links are absent. Native fits and supplied segment plans do not
            establish D21 provenance. Recompute older automatic finding fits.
        """
        from ._model_peak_links import read_peak_links  # noqa: PLC0415
        from ._reduction import select_block  # noqa: PLC0415

        scope = select_block(data, block).tree
        return read_peak_links(scope[f"{cls.result_path(alias=alias)}/peak_links"])

    @staticmethod
    def bind_analysis(
        block: ToltecBlockView, analysis: KidsModelAnalysis
    ) -> tuple[KidsModelAnalysis, dict[str, object]]:
        """Bind generic analysis settings to a block's recorded sweep history.

        Parameters
        ----------
        block : tolteca_kids.blocks.ToltecBlockView
            Original or reopened block with acquired probe sample intervals.
            Detector samples need not be available.
        analysis : KidsModelAnalysis
            Requested numerical settings. Nonlinear fits above the unique-root
            domain resolve the default branch from acquisition order. An explicit
            directional branch must agree with that order.

        Returns
        -------
        analysis : KidsModelAnalysis
            Resolved settings. Default linear and unique-domain fits are unchanged.
        history : dict
            Compact acquisition provenance to store as ``sweep_history`` on the
            returned native model scope. Unavailable history is recorded for
            fits that do not need direction. Sorted frequency is not evidence.

        Raises
        ------
        ValueError
            A directional fit lacks monotonic recorded history, contradicts it,
            or its reduced LO coordinates differ from the acquired probe order.
        """
        from ._model_history import bind_analysis  # noqa: PLC0415

        return bind_analysis(block, analysis)

    @staticmethod
    def make_plan(sweep, finding, *, fit_n_fwhm=2.0, mask=None, frequency_bounds=None):
        """Construct one block's groups and segments from measured frequency support.

        Parameters
        ----------
        sweep : tolteca_kids.sweep.ReducedSweepView
            Measured sweep with original channel labels and frequency samples.
        finding : tolteca_kids.kids_find_view.KidsFindView
            Saved candidates; their idx and group columns supply local resonance
            identities and initial groups. Intersecting support joins groups;
            original group labels remain in ``finding_group_id``. Probe matches
            do not select a primary.
        fit_n_fwhm : float, optional
            Window half-width in each candidate's measured linewidth, default 2.
        mask : array_like of bool, optional
            Usable samples with the sweep shape. Channel/group responses need at
            least three usable samples across their combined windows.
        frequency_bounds : array_like, shape (n_candidates, 2), optional
            Explicit lower and upper support in Hz, aligned with ``detected``.
            Overrides the linewidth windows; permits asymmetric wings and
            externally determined positional uncertainty. Bounds must contain
            each initial frequency. They select samples, not hard profile cutoffs.

        Returns
        -------
        xarray.Dataset
            Association-segment rows with original half-open sample positions,
            fr and loss seeds. Every overlapping acquired channel contributes.
            Frequency labels need not be increasing; disjoint runs remain separate.
            Each usable channel/sample belongs to at most one fit group.
            Group labels are local and may change when support is enlarged.
            No supported candidates produce an empty plan with an explicit
            status. Planning can retain this gap in a sequence; fitting requires
            a nonempty plan.

        Raises
        ------
        ValueError
            Candidates, frequency grids, window width or sample mask are invalid.
        """
        from ._model_fit import make_plan  # noqa: PLC0415

        return make_plan(
            sweep,
            finding,
            fit_n_fwhm=fit_n_fwhm,
            mask=mask,
            frequency_bounds=frequency_bounds,
        )

    @classmethod
    def _fit(cls, data, context, directory):
        block = data.toltec_kids.block
        config = context.config
        report = directory / "kids_model.ecsv"
        command = [
            str(config.executable),
            "fit",
            context.source,
            "--blocks",
            str(block.meta.block),
            "--output",
            str(report),
            "--rolling",
            "--completed_through",
            str(context.sample_end),
            "--block_output",
            str(directory / "blocks.ecsv"),
            "--chan_output",
            str(directory / "channels.ecsv"),
            "--threads",
            str(config.threads),
            "--execution",
            config.execution,
        ]
        if config.retain_processed:
            command.extend(["--processed_output", str(directory / "processed.nc")])
        (directory / "command.json").write_text(json.dumps(command, indent=2))
        with (directory / "native.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)  # noqa: S603 -- literal argument vector
        table = Table.read(report, format="ascii.ecsv")
        _validate_report(table)
        if table.meta.get("block") != block.meta.block:
            msg = "Native fit returned a different block"
            raise ValueError(msg)
        _validate_binding(table, block)
        data["tlaloc_kids_model"] = TlalocKidsModel.from_table(table).data
        context.artifacts = ArtifactContext()
        context.artifacts.add(config.artifacts.model, report.read_bytes())
        context.artifacts.add(
            config.artifacts.kidscpp,
            {
                name: (directory / name).read_bytes()
                for name in KIDSCPP_FILES
                if (directory / name).is_file()
            },
        )
